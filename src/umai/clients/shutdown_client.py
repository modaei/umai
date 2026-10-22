from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings
from umai.clients.well_filter_client import (
    WellFilterClientError,
    _add_filter_condition,
    _normalize_filters,
)


class ShutdownClientError(RuntimeError):
    """Raised when a well shutdown query is invalid or cannot be completed."""


DOWNTIME_CODES = {
    "PRF": "Paraffin",
    "BT": "Batch/Downhole Treating",
    "CL": "Casing Leak",
    "CR": "Compressor Problems",
    "DH": "Downhole Problems",
    "EL": "Electrical Problems",
    "ES": "Emergency Shut Down",
    "ESP": "ESP Downhole Problems",
    "FBTU": "Facility/Battery/TS Upset",
    "FPI": "Flowline/Pipeline Problems or Maintenance",
    "FSD": "Facility Shut Down",
    "NRW": "Non Rig Wellwork",
    "PR": "Pumping Rod Problems",
    "PU": "Artificial Lift Surface Failure",
    "RES": "Reservoir Management/Optimization",
    "RIG": "Rig Nearby (Sim Ops)",
    "SEI": "Surface Equipment Problems or Integrity",
    "SIB": "Shut In For Buildup",
    "TL": "Tubing Leak",
    "TPD": "Third Party Distruption",
    "TW": "Testing Another Well In Field",
    "UTP": "Uneconomical To Produce/Repair",
    "WEA": "Inclement Weather",
    "WHS": "Water Handling System Capacity",
    "WO": "Proactive/Planned Rigwork",
}


class ShutdownClient:
    def __init__(self, engine: Engine, max_rows: int = 500):
        self.engine = engine
        self.max_rows = max_rows

    @classmethod
    def from_settings(cls, settings: Settings) -> "ShutdownClient":
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

    def list_downtime_codes(self) -> dict[str, Any]:
        return {"downtime_codes": DOWNTIME_CODES}

    def get_shutdowns(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if start_day > end_day:
            raise ShutdownClientError("start_date cannot be after end_date.")

        shutdowns = self._shutdowns(site_id, start_day, end_day)
        range_start = datetime.combine(start_day, time.min)
        range_end = datetime.combine(end_day, time.min) + timedelta(days=1)

        total_hours = round(
            sum(self._shutdown_overlap_hours(row, range_start, range_end) for row in shutdowns),
            3,
        )
        return {
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "count": len(shutdowns),
            "total_hours": total_hours,
            "shutdowns": [_without_internal_ids(row) for row in shutdowns],
        }

    def get_current_shutdowns(
        self, site_id: int, as_of_date: str | None = None
    ) -> dict[str, Any]:
        as_of = self._parse_date(as_of_date) if as_of_date else date.today()
        start, end = self._day_bounds(as_of)
        query = text(
            """
            SELECT wells.name AS well_name,
                well_shutdowns.`start` AS start,
                well_shutdowns.`end` AS end,
                well_shutdowns.downtime_code,
                well_shutdowns.comments
            FROM well_shutdowns
            JOIN wells ON well_shutdowns.well_id = wells.id
            WHERE wells.site_id = :site_id
                AND well_shutdowns.`start` < :end_time
                AND (
                    well_shutdowns.`end` IS NULL
                    OR well_shutdowns.`end` >= :start_time
                )
            ORDER BY well_shutdowns.`start`, wells.name
            LIMIT :limit
            """
        )
        rows = self._execute(
            query,
            site_id=site_id,
            start_time=start,
            end_time=end,
        )
        shutdowns = [self._compact_shutdown_row(row) for row in rows]
        return {
            "site_id": site_id,
            "as_of_date": as_of.isoformat(),
            "count": len(shutdowns),
            "shutdowns": [_without_internal_ids(row) for row in shutdowns],
        }

    def summarize_shutdown_causes(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if start_day > end_day:
            raise ShutdownClientError("start_date cannot be after end_date.")
        shutdowns = self._shutdowns(site_id, start_day, end_day)

        cause_map: dict[str, dict[str, Any]] = {}
        range_start = datetime.combine(start_day, time.min)
        range_end = datetime.combine(end_day, time.min) + timedelta(days=1)
        for shutdown in shutdowns:
            cause = self._cause_bucket(cause_map, shutdown)
            cause["event_count"] += 1
            cause["total_hours"] = round(
                cause["total_hours"]
                + self._shutdown_overlap_hours(shutdown, range_start, range_end),
                3,
            )

        causes = sorted(
            cause_map.values(),
            key=lambda item: (
                item["total_hours"],
                item["event_count"],
                item["downtime_reason"],
            ),
            reverse=True,
        )
        return {
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "cause_count": len(causes),
            "main_cause": causes[0] if causes else None,
            "causes": causes,
        }

    def get_active_wells(self, site_id: int, active_date: str) -> dict[str, Any]:
        """Classify active wells for one day using only effective ONRR state."""
        day = self._parse_date(active_date)
        _, day_end = self._day_bounds(day)
        wells = self._wells_with_onrr_state(site_id, day_end)

        active_wells = []
        inactive_wells = []
        for well in wells:
            well_result = {
                "well": f"Well - {well['well_name']}",
                "onrr_code": well.get("onrr_code_name"),
                "onrr_code_description": well.get("onrr_code_description"),
                "onrr_active_well": bool(well.get("active_well")),
                "onrr_injection_well": bool(well.get("injection_well")),
            }
            if well.get("active_well"):
                active_wells.append(well_result)
            else:
                inactive_wells.append(
                    {
                        **well_result,
                        "status": "onrr_inactive",
                        "reason": "ONRR code is not marked active for this date.",
                    }
                )

        return {
            "site_id": site_id,
            "date": day.isoformat(),
            "result_type": "active",
            "active_count": len(active_wells),
            "inactive_count": len(inactive_wells),
            "active_wells": active_wells,
            "inactive_wells": inactive_wells,
            "rules": {
                "onrr_code_as_of": day_end.isoformat(sep=" "),
                "active_well_rule": (
                    "Active wells must have active_well=true on the ONRR code "
                    "as of that day."
                ),
            },
        }

    def get_producing_wells(
        self,
        site_id: int,
        producing_date: str,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Classify producing wells for one day, excluding ONRR injection wells."""
        normalized_filters = self._normalize_well_filters(filters)
        result = self._classify_wells_by_onrr_and_shutdown(
            site_id,
            producing_date,
            result_type="producing",
            exclude_injection_wells=True,
            filters=normalized_filters,
        )
        result["producing_count"] = result.pop("active_count")
        result["non_producing_count"] = result.pop("inactive_count")
        result["producing_wells"] = result.pop("active_wells")
        result["non_producing_wells"] = result.pop("inactive_wells")
        result["filters"] = normalized_filters
        return result

    def get_producing_wells_for_range(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
        range_mode: str = "any_day",
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return wells producing on at least one day in a bounded date range.

        Daily classification remains authoritative for historical ONRR codes and
        shutdown semantics. Independent days are evaluated in a small worker pool
        to avoid serial database latency without duplicating those business rules.
        """
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if end_day < start_day:
            raise ShutdownClientError("end_date must be on or after start_date.")
        if range_mode != "any_day":
            raise ShutdownClientError("Only range_mode='any_day' is supported.")
        day_count = (end_day - start_day).days + 1
        if day_count > 366:
            raise ShutdownClientError("Producing-well ranges are limited to 366 days.")
        normalized_filters = self._normalize_well_filters(filters)

        date_values = [
            (start_day + timedelta(days=offset)).isoformat()
            for offset in range(day_count)
        ]
        with ThreadPoolExecutor(max_workers=min(4, day_count)) as executor:
            daily_results = list(
                executor.map(
                    lambda value: self.get_producing_wells(
                        site_id,
                        value,
                        filters=normalized_filters,
                    ),
                    date_values,
                )
            )

        producing_by_name: dict[str, dict[str, Any]] = {}
        qualifying_dates: dict[str, list[str]] = {}
        for daily_result in daily_results:
            result_date = str(daily_result["date"])
            for well in daily_result["producing_wells"]:
                name = str(well["well"])
                producing_by_name.setdefault(name, well)
                qualifying_dates.setdefault(name, []).append(result_date)

        producing_wells = [producing_by_name[name] for name in sorted(producing_by_name)]
        return {
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "range_mode": range_mode,
            "filters": normalized_filters,
            "day_count": day_count,
            "producing_count": len(producing_wells),
            "producing_wells": producing_wells,
            "qualifying_dates_by_well": qualifying_dates,
            "rule": (
                "A well is included when it satisfies the producing-well rules "
                "on at least one day in the range."
            ),
        }

    def _classify_wells_by_onrr_and_shutdown(
        self,
        site_id: int,
        active_date: str,
        *,
        result_type: str,
        exclude_injection_wells: bool,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        day = self._parse_date(active_date)
        day_start, day_end = self._day_bounds(day)
        wells = self._wells_with_onrr_state(site_id, day_end, filters=filters)
        shutdowns_by_well = self._day_shutdown_state_by_well(site_id, day)

        active_wells = []
        inactive_wells = []
        partial_shutdown_wells = []
        for well in wells:
            shutdown_state = shutdowns_by_well.get(well["well_id"])
            well_result = {
                "well": f"Well - {well['well_name']}",
                "onrr_code": well.get("onrr_code_name"),
                "onrr_code_description": well.get("onrr_code_description"),
                "onrr_active_well": bool(well.get("active_well")),
                "onrr_injection_well": bool(well.get("injection_well")),
            }
            if shutdown_state and shutdown_state["status"] == "partial_shutdown":
                partial_shutdown_wells.append({**well_result, **shutdown_state})
            elif shutdown_state and shutdown_state["status"] == "full_day_shutdown":
                inactive_wells.append({**well_result, **shutdown_state})
            elif exclude_injection_wells and well.get("injection_well"):
                inactive_wells.append(
                    {
                        **well_result,
                        "status": "onrr_injection_well",
                        "reason": (
                            "ONRR code is marked as an injection well, so it is "
                            "not counted as producing."
                        ),
                    }
                )
            elif well.get("active_well"):
                active_wells.append(well_result)
            else:
                inactive_wells.append(
                    {
                        **well_result,
                        "status": "onrr_inactive",
                        "reason": "ONRR code is not marked active for this date.",
                    }
                )

        return {
            "site_id": site_id,
            "date": day.isoformat(),
            "result_type": result_type,
            "active_count": len(active_wells),
            "inactive_count": len(inactive_wells),
            "partial_shutdown_count": len(partial_shutdown_wells),
            "active_wells": active_wells,
            "inactive_wells": [_without_internal_ids(row) for row in inactive_wells],
            "partial_shutdown_wells": [
                _without_internal_ids(row) for row in partial_shutdown_wells
            ],
            "partial_shutdown_well_names": [
                row["well"] for row in partial_shutdown_wells
            ],
            "partial_shutdown_summary": (
                "Partial shutdown wells: "
                + ", ".join(row["well"] for row in partial_shutdown_wells)
                if partial_shutdown_wells
                else None
            ),
            "rules": {
                "onrr_code_as_of": day_end.isoformat(sep=" "),
                "producing_well_rule": (
                    "Producing wells must have active_well=true and "
                    "injection_well=false on the ONRR code as of that day."
                    if exclude_injection_wells
                    else None
                ),
                "partial_shutdown_note": (
                    "Partially shutdown wells are excluded from both producing "
                    "and non-producing counts."
                    if exclude_injection_wells
                    else "Partially shutdown wells are excluded from both active "
                    "and inactive counts."
                ),
            },
        }

    def _shutdowns(
        self, site_id: int, start_day: date, end_day: date
    ) -> list[dict[str, Any]]:
        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day, time.min) + timedelta(days=1)
        query = text(
            """
            SELECT well_shutdowns.well_id,
                wells.name AS well_name,
                well_shutdowns.`start` AS start,
                well_shutdowns.`end` AS end,
                well_shutdowns.downtime_code,
                well_shutdowns.comments
            FROM well_shutdowns
            JOIN wells ON well_shutdowns.well_id = wells.id
            WHERE wells.site_id = :site_id
                AND well_shutdowns.`start` < :end_time
                AND (
                    well_shutdowns.`end` IS NULL
                    OR well_shutdowns.`end` >= :start_time
                )
            ORDER BY well_shutdowns.`start`, wells.name
            LIMIT :limit
            """
        )
        rows = self._execute(
            query,
            site_id=site_id,
            start_time=start_time,
            end_time=end_time,
        )
        return [self._compact_shutdown_row(row) for row in rows]

    def _wells_with_onrr_state(
        self,
        site_id: int,
        as_of_time: datetime,
        filters: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["w.site_id = :site_id"]
        params: dict[str, Any] = {"site_id": site_id, "as_of_time": as_of_time}
        normalized_filters = filters or []
        for index, filter_item in enumerate(normalized_filters):
            _add_filter_condition(filter_item, index, conditions, params)
        needs_battery_join = any(item["field"] == "battery" for item in normalized_filters)
        needs_lact_join = any(item["field"] == "lact" for item in normalized_filters)
        battery_join = "LEFT JOIN batteries b ON b.id = w.battery_id" if needs_battery_join else ""
        lact_join = "LEFT JOIN lacts l ON l.id = w.lact_id" if needs_lact_join else ""
        query = text(
            f"""
            SELECT
                w.id AS well_id,
                w.name AS well_name,
                COALESCE(
                    (
                        SELECT wh_before.new_value
                        FROM well_histories wh_before
                        WHERE wh_before.well_id = w.id
                            AND wh_before.property = 'onrr_code_id'
                            AND wh_before.changed_at < :as_of_time
                        ORDER BY wh_before.changed_at DESC, wh_before.id DESC
                        LIMIT 1
                    ),
                    (
                        SELECT wh_after.old_value
                        FROM well_histories wh_after
                        WHERE wh_after.well_id = w.id
                            AND wh_after.property = 'onrr_code_id'
                            AND wh_after.changed_at >= :as_of_time
                        ORDER BY wh_after.changed_at ASC, wh_after.id ASC
                        LIMIT 1
                    ),
                    w.onrr_code_id
                ) AS effective_onrr_code_id,
                oc.name AS onrr_code_name,
                oc.active_well,
                oc.injection_well,
                oc.description AS onrr_code_description
            FROM wells w
            LEFT JOIN onrr_codes oc
                ON oc.id = COALESCE(
                    (
                        SELECT wh_before.new_value
                        FROM well_histories wh_before
                        WHERE wh_before.well_id = w.id
                            AND wh_before.property = 'onrr_code_id'
                            AND wh_before.changed_at < :as_of_time
                        ORDER BY wh_before.changed_at DESC, wh_before.id DESC
                        LIMIT 1
                    ),
                    (
                        SELECT wh_after.old_value
                        FROM well_histories wh_after
                        WHERE wh_after.well_id = w.id
                            AND wh_after.property = 'onrr_code_id'
                            AND wh_after.changed_at >= :as_of_time
                        ORDER BY wh_after.changed_at ASC, wh_after.id ASC
                        LIMIT 1
                    ),
                    w.onrr_code_id
                )
            {battery_join}
            {lact_join}
            WHERE {" AND ".join(conditions)}
            ORDER BY w.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        return rows

    @staticmethod
    def _normalize_well_filters(
        filters: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        try:
            return _normalize_filters(filters or [])
        except WellFilterClientError as exc:
            raise ShutdownClientError(str(exc)) from exc

    def _day_shutdown_state_by_well(
        self, site_id: int, day: date
    ) -> dict[int, dict[str, Any]]:
        shutdowns = self._shutdowns(site_id, day, day)
        day_start, day_end = self._day_bounds(day)

        states: dict[int, dict[str, Any]] = {}
        for shutdown in shutdowns:
            well_id = int(shutdown["well_id"])
            hours = self._shutdown_overlap_hours(shutdown, day_start, day_end)
            state = states.setdefault(
                well_id,
                {
                    "shutdown_hours": 0.0,
                    "shutdowns": [],
                },
            )
            state["shutdown_hours"] += hours
            state["shutdowns"].append(shutdown)

        classified = {}
        for well_id, state in states.items():
            total_hours = round(state["shutdown_hours"], 3)
            if total_hours >= 24:
                status = "full_day_shutdown"
                reason = "Shutdown covers the full day."
            else:
                status = "partial_shutdown"
                reason = (
                    "Shutdown covers part of the day, so the well is excluded "
                    "from both active and inactive counts."
                )
            classified[well_id] = {
                "status": status,
                "reason": reason,
                "shutdown_hours": total_hours,
                "total_shutdown_hours": total_hours,
                "shutdowns": state["shutdowns"],
            }
        return classified

    def _execute(self, query, **params: Any) -> list[dict[str, Any]]:
        params.setdefault("limit", self.max_rows)
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, params)
                return [self._serialize_row(dict(row._mapping)) for row in result]
        except SQLAlchemyError as exc:
            raise ShutdownClientError(f"Database query failed: {exc}") from exc

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
    def _compact_shutdown_row(row: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "well": f"Well - {row['well_name']}",
            "well_id": row.get("well_id"),
            "start": row.get("start"),
            "end": row.get("end"),
        }

        code = row.get("downtime_code")
        if code:
            compact["downtime_code"] = code
            compact["downtime_reason"] = DOWNTIME_CODES.get(code, code)
        if row.get("comments"):
            compact["comments"] = row["comments"]
        return compact

    @staticmethod
    def _shutdown_overlap_hours(
        row: dict[str, Any],
        range_start: datetime,
        range_end: datetime,
    ) -> float:
        start = _parse_datetime_value(row.get("start"))
        end = _parse_datetime_value(row.get("end")) or range_end
        if start is None:
            return 0.0

        overlap_start = max(start, range_start)
        overlap_end = min(end, range_end)
        if overlap_end <= overlap_start:
            return 0.0
        return round((overlap_end - overlap_start).total_seconds() / 3600, 3)

    @staticmethod
    def _cause_bucket(
        cause_map: dict[str, dict[str, Any]],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        code = row.get("downtime_code") or "UNKNOWN"
        if code not in cause_map:
            cause_map[code] = {
                "downtime_code": code,
                "downtime_reason": DOWNTIME_CODES.get(
                    code, "Unknown" if code == "UNKNOWN" else code
                ),
                "event_count": 0,
                "total_hours": 0.0,
            }
        return cause_map[code]

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ShutdownClientError("Dates must use YYYY-MM-DD format.") from exc

    @staticmethod
    def _day_bounds(day: date) -> tuple[datetime, datetime]:
        start = datetime.combine(day, time.min)
        return start, start + timedelta(days=1)


class UnavailableShutdownClient:
    def __init__(self, reason: str):
        self.reason = reason

    def list_downtime_codes(self) -> dict[str, Any]:
        return {"downtime_codes": DOWNTIME_CODES}

    def get_shutdowns(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        raise ShutdownClientError(self.reason)

    def get_current_shutdowns(
        self, site_id: int, as_of_date: str | None = None
    ) -> dict[str, Any]:
        raise ShutdownClientError(self.reason)

    def summarize_shutdown_causes(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        raise ShutdownClientError(self.reason)

    def get_active_wells(self, site_id: int, active_date: str) -> dict[str, Any]:
        raise ShutdownClientError(self.reason)

    def get_producing_wells(
        self,
        site_id: int,
        producing_date: str,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise ShutdownClientError(self.reason)


def _parse_datetime_value(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return datetime.fromisoformat(text_value)
    except ValueError:
        return None


def _without_internal_ids(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_internal_ids(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_internal_ids(item)
            for key, item in value.items()
            if key != "well_id"
        }
    return value
