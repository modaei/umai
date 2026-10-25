from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.clients.shutdown_client import DOWNTIME_CODES
from umai.config.settings import Settings
from umai.rag.vector_store import OperationalContextStoreError


logger = logging.getLogger(__name__)


class WellTimelineClientError(RuntimeError):
    """Raised when a well timeline query is invalid or cannot be completed."""


class OperationalContextSearchStore(Protocol):
    """Minimal RAG-store interface needed to enrich well timelines."""

    def search(
        self,
        query: str,
        site_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
        entity_name: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        ...


class WellTimelineClient:
    def __init__(
        self,
        engine: Engine,
        max_rows_per_source: int = 100,
        operational_context_store: OperationalContextSearchStore | None = None,
    ):
        self.engine = engine
        self.max_rows_per_source = max_rows_per_source
        self.operational_context_store = operational_context_store

    @classmethod
    def from_settings(cls, settings: Settings) -> "WellTimelineClient":
        settings.validate_database()
        url = URL.create(
            drivername="mysql+pymysql",
            username=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
        )
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        )
        return cls(engine)

    def get_well_timeline(
        self,
        site_id: int,
        well_name: str,
        start_date: str,
        end_date: str,
        context_query: str | None = None,
    ) -> dict[str, Any]:
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if start_day > end_day:
            raise WellTimelineClientError("start_date cannot be after end_date.")

        well = self._find_well(site_id, well_name)
        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day, time.min) + timedelta(days=1)

        events = []
        events.extend(self._well_tests(well["well_id"], start_time, end_time))
        events.extend(self._well_fluids(well["well_id"], start_time, end_time))
        events.extend(self._shutdowns(well["well_id"], start_day, end_day))
        events.extend(self._well_history(well["well_id"], start_time, end_time))
        events.extend(self._chart_notes(well["well_id"], start_time, end_time))
        events.extend(
            self._general_notes(site_id, well["well_name"], start_day, end_day)
        )
        events.extend(
            self._work_orders(site_id, well["well_name"], start_time, end_time)
        )
        events.extend(self._alarms(site_id, well["well_name"], start_time, end_time))
        events.extend(
            self._operational_context_events(
                site_id=site_id,
                well_name=well["well_name"],
                context_query=context_query,
                start_day=start_day,
                end_day=end_day,
                existing_events=events,
            )
        )

        events.sort(key=lambda item: item["time"])
        return {
            "site_id": site_id,
            "well": f"Well - {well['well_name']}",
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "event_count": len(events),
            "events": events,
        }

    def _find_well(self, site_id: int, well_name: str) -> dict[str, Any]:
        normalized_name = _normalize_well_name(well_name)
        query = text(
            """
            SELECT id AS well_id, name AS well_name
            FROM wells
            WHERE site_id = :site_id
                AND (
                    name = :exact_well_name
                    OR name LIKE :well_name_contains
                    OR name LIKE :well_name_suffix
                    OR name LIKE :hartzog_unit_name
                )
            ORDER BY
                CASE
                    WHEN name = :exact_well_name THEN 0
                    WHEN name = :hartzog_unit_exact THEN 1
                    WHEN name LIKE :well_name_suffix THEN 2
                    ELSE 3
                END,
                name
            LIMIT 1
            """
        )
        rows = self._execute(
            query,
            site_id=site_id,
            well_name_contains=f"%{normalized_name}%",
            well_name_suffix=f"% {normalized_name}",
            hartzog_unit_name=f"%HARTZOG DRAW UNIT {normalized_name}%",
            hartzog_unit_exact=f"HARTZOG DRAW UNIT {normalized_name}",
            exact_well_name=normalized_name,
        )
        if not rows:
            raise WellTimelineClientError(f"Well not found: {well_name}")
        return rows[0]

    def _well_tests(
        self, well_id: int, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT time, oil, water, gas, pip, m_temp, amps, tbgp, csgp, runtime, comments
            FROM well_tests
            WHERE well_id = :well_id
                AND time >= :start_time
                AND time < :end_time
            ORDER BY time
            LIMIT :limit
            """
        )
        return [
            self._event(
                row["time"],
                "well_test",
                "Well test",
                self._non_empty_fields(
                    row,
                    (
                        "oil",
                        "water",
                        "gas",
                        "pip",
                        "m_temp",
                        "amps",
                        "tbgp",
                        "csgp",
                        "runtime",
                        "comments",
                    ),
                ),
            )
            for row in self._execute(
                query,
                well_id=well_id,
                start_time=start_time,
                end_time=end_time,
            )
        ]

    def _well_fluids(
        self, well_id: int, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT time, level, comments
            FROM well_fluids
            WHERE well_id = :well_id
                AND time >= :start_time
                AND time < :end_time
            ORDER BY time
            LIMIT :limit
            """
        )
        return [
            self._event(
                row["time"],
                "well_fluid",
                "Fluid level",
                self._non_empty_fields(row, ("level", "comments")),
            )
            for row in self._execute(
                query,
                well_id=well_id,
                start_time=start_time,
                end_time=end_time,
            )
        ]

    def _shutdowns(
        self, well_id: int, start_day: date, end_day: date
    ) -> list[dict[str, Any]]:
        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day, time.min) + timedelta(days=1)
        query = text(
            """
            SELECT `start` AS start, `end` AS end, downtime_code, comments
            FROM well_shutdowns
            WHERE well_id = :well_id
                AND `start` < :end_time
                AND (
                    `end` IS NULL
                    OR `end` >= :start_time
                )
            ORDER BY `start`
            LIMIT :limit
            """
        )
        events = []
        for row in self._execute(
            query,
            well_id=well_id,
            start_time=start_time,
            end_time=end_time,
        ):
            code = row.get("downtime_code")
            details = {
                "downtime_code": code,
                "downtime_reason": DOWNTIME_CODES.get(code, code) if code else None,
                "comments": row.get("comments"),
                "start": row.get("start"),
                "end": row.get("end"),
            }

            events.append(
                self._event(
                    row["start"],
                    "shutdown",
                    "Well shutdown",
                    {k: v for k, v in details.items() if v is not None},
                )
            )
        return events

    def _well_history(
        self, well_id: int, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT changed_at, property, old_value, new_value, source
            FROM well_histories
            WHERE well_id = :well_id
                AND changed_at >= :start_time
                AND changed_at < :end_time
            ORDER BY changed_at
            LIMIT :limit
            """
        )
        return [
            self._event(
                row["changed_at"],
                "well_history",
                "Well history",
                self._non_empty_fields(
                    row, ("property", "old_value", "new_value", "source")
                ),
            )
            for row in self._execute(
                query,
                well_id=well_id,
                start_time=start_time,
                end_time=end_time,
                optional=True,
            )
        ]

    def _chart_notes(
        self, well_id: int, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT id AS source_id, x_axis_value, chart_name, note
            FROM chart_notes
            WHERE object_id = :well_id
                AND object_type IN ('RodPumpOilWell', 'OilWell', 'WaterWell')
                AND x_axis_value >= :start_value
                AND x_axis_value < :end_value
            ORDER BY x_axis_value
            LIMIT :limit
            """
        )
        return [
            self._event(
                row["x_axis_value"],
                "chart_note",
                "Chart note",
                self._non_empty_fields(row, ("source_id", "chart_name", "note")),
            )
            for row in self._execute(
                query,
                well_id=well_id,
                start_value=start_time.strftime("%Y-%m-%d %H:%M"),
                end_value=end_time.strftime("%Y-%m-%d %H:%M"),
                optional=True,
            )
        ]

    def _operational_context_events(
        self,
        site_id: int,
        well_name: str,
        context_query: str | None,
        start_day: date,
        end_day: date,
        existing_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add RAG matches to the timeline without failing the structured timeline.

        The structured timeline queries a fixed set of MySQL tables. The RAG
        index can contain additional operational text sources, so it is used as
        an enrichment layer. Duplicate records are skipped when a structured
        event already has the same source type and source id.
        """
        if self.operational_context_store is None:
            return []

        existing_source_keys = _event_source_keys(existing_events)
        search_query = context_query or f"operational history context for {well_name}"
        try:
            result = self.operational_context_store.search(
                query=search_query,
                site_id=site_id,
                start_date=start_day,
                end_date=end_day,
                entity_name=well_name,
                source_types=None,
                limit=self.max_rows_per_source,
            )
        except OperationalContextStoreError as exc:
            logger.warning("Well timeline RAG enrichment failed: %s", exc)
            return []

        events = []
        for match in result.get("matches", []):
            source_type = match.get("source_type")
            source_id = match.get("source_id")
            if (source_type, str(source_id)) in existing_source_keys:
                continue
            event_date = match.get("event_date")
            if not event_date:
                continue
            events.append(
                self._event(
                    event_date,
                    "operational_context",
                    "Operational context",
                    self._non_empty_fields(
                        {
                            "source_type": source_type,
                            "source_id": source_id,
                            "entity_name": match.get("entity_name"),
                            "text": match.get("text"),
                        },
                        ("source_type", "source_id", "entity_name", "text"),
                    ),
                )
            )
        return events

    def _general_notes(
        self, site_id: int, well_name: str, start_day: date, end_day: date
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT date, comments
            FROM general_notes
            WHERE site_id = :site_id
                AND date >= :start_date
                AND date <= :end_date
                AND comments LIKE :well_name
            ORDER BY date
            LIMIT :limit
            """
        )
        return [
            self._event(
                row["date"],
                "general_note",
                "General note",
                self._non_empty_fields(row, ("comments",)),
            )
            for row in self._execute(
                query,
                site_id=site_id,
                start_date=start_day,
                end_date=end_day,
                well_name=f"%{well_name}%",
                optional=True,
            )
        ]

    def _work_orders(
        self, site_id: int, well_name: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT work_orders.time, work_orders.subject, work_orders.status,
                work_orders.priority, work_orders.vendor, work_orders.comments
            FROM work_orders
            WHERE work_orders.site_id = :site_id
                AND work_orders.time >= :start_time
                AND work_orders.time < :end_time
                AND (
                    work_orders.subject LIKE :well_name
                    OR work_orders.comments LIKE :well_name
                )
            ORDER BY work_orders.time
            LIMIT :limit
            """
        )
        return [
            self._event(
                row["time"],
                "work_order",
                "Work order",
                self._non_empty_fields(
                    row, ("subject", "status", "priority", "vendor", "comments")
                ),
            )
            for row in self._execute(
                query,
                site_id=site_id,
                start_time=start_time,
                end_time=end_time,
                well_name=f"%{well_name}%",
                optional=True,
            )
        ]

    def _alarms(
        self, site_id: int, well_name: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        query = text(
            """
            SELECT alarm_events.happened_on,
                alarm_events.event,
                alarm_events.number_value,
                alarm_events.flag_value,
                data_points.facility_name,
                data_points.device_name,
                data_points.data_point_name
            FROM alarm_events
            JOIN data_points ON alarm_events.data_point_id = data_points.id
            WHERE data_points.site_id = :site_id
                AND alarm_events.happened_on >= :start_ts
                AND alarm_events.happened_on < :end_ts
                AND (
                    data_points.facility_name LIKE :well_name
                    OR data_points.device_name LIKE :well_name
                    OR data_points.data_point_name LIKE :well_name
                )
            ORDER BY alarm_events.happened_on
            LIMIT :limit
            """
        )
        events = []
        for row in self._execute(
            query,
            site_id=site_id,
            start_ts=start_ts,
            end_ts=end_ts,
            well_name=f"%{well_name}%",
            optional=True,
        ):
            event_time = datetime.fromtimestamp(row["happened_on"]).isoformat(sep=" ")
            events.append(
                self._event(
                    event_time,
                    "alarm",
                    "Alarm event",
                    self._non_empty_fields(
                        row,
                        (
                            "event",
                            "number_value",
                            "flag_value",
                            "facility_name",
                            "device_name",
                            "data_point_name",
                        ),
                    ),
                )
            )
        return events

    def _execute(self, query, optional: bool = False, **params: Any) -> list[dict[str, Any]]:
        params.setdefault("limit", self.max_rows_per_source)
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, params)
                return [self._serialize_row(dict(row._mapping)) for row in result]
        except SQLAlchemyError as exc:
            if optional and _looks_like_missing_table_or_column(exc):
                return []
            raise WellTimelineClientError(f"Database query failed: {exc}") from exc

    @staticmethod
    def _event(time_value: Any, source: str, label: str, details: dict[str, Any]) -> dict[str, Any]:
        return {
            "time": _string_time(time_value),
            "source": source,
            "label": label,
            "details": details,
        }

    @staticmethod
    def _non_empty_fields(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            field: row.get(field)
            for field in fields
            if row.get(field) is not None and row.get(field) != ""
        }

    @staticmethod
    def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
        serialized = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat(sep=" ")
            elif isinstance(value, date):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise WellTimelineClientError("Dates must use YYYY-MM-DD format.") from exc


def _string_time(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _normalize_well_name(value: str) -> str:
    value = value.strip()
    upper_value = value.upper()
    if upper_value.startswith("WELL "):
        return value[5:].strip()
    return value


def _event_source_keys(events: list[dict[str, Any]]) -> set[tuple[Any, str]]:
    keys = set()
    for event in events:
        source_id = event.get("details", {}).get("source_id")
        if source_id is not None:
            keys.add((event.get("source"), str(source_id)))
    return keys


def _looks_like_missing_table_or_column(exc: SQLAlchemyError) -> bool:
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "no such table",
            "no such column",
            "unknown column",
            "doesn't exist",
        )
    )


class UnavailableWellTimelineClient:
    def __init__(self, reason: str):
        self.reason = reason

    def get_well_timeline(
        self,
        site_id: int,
        well_name: str,
        start_date: str,
        end_date: str,
        context_query: str | None = None,
    ) -> dict[str, Any]:
        raise WellTimelineClientError(self.reason)
