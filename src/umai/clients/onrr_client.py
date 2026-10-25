from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


class OnrrClientError(RuntimeError):
    """Raised when ONRR code or well-status data cannot be queried."""


class OnrrClient:
    def __init__(self, engine: Engine, max_rows: int = 500):
        self.engine = engine
        self.max_rows = max_rows

    @classmethod
    def from_settings(cls, settings: Settings) -> "OnrrClient":
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

    def list_onrr_codes(self) -> dict[str, Any]:
        query = text(
            """
            SELECT id, name, active_well, injection_well, description
            FROM onrr_codes
            ORDER BY name
            LIMIT :limit
            """
        )
        rows = self._execute(query)
        return {
            "count": len(rows),
            "onrr_codes": [_compact_onrr_code(row) for row in rows],
        }

    def explain_onrr_code(self, code: str) -> dict[str, Any]:
        normalized = code.strip()
        if not normalized:
            raise OnrrClientError("ONRR code is required.")

        query = text(
            """
            SELECT id, name, active_well, injection_well, description
            FROM onrr_codes
            WHERE LOWER(name) = LOWER(:code)
            LIMIT 1
            """
        )
        rows = self._execute(query, code=normalized)
        return {
            "code": normalized,
            "found": bool(rows),
            "onrr_code": _compact_onrr_code(rows[0]) if rows else None,
        }

    def get_well_onrr_status_as_of_date(
        self,
        site_id: int,
        well_name: str,
        as_of_date: str,
    ) -> dict[str, Any]:
        day = self._parse_date(as_of_date)
        _, day_end = self._day_bounds(day)
        well = self._well_onrr_state(site_id, well_name, day_end)
        if well is None:
            return {
                "site_id": site_id,
                "well_name": well_name,
                "date": day.isoformat(),
                "found": False,
                "status": None,
            }
        return {
            "site_id": site_id,
            "well_name": well_name,
            "date": day.isoformat(),
            "found": True,
            "status": _compact_well_onrr_state(well),
            "rules": _onrr_rules(day_end),
        }

    def count_wells_by_onrr_status(
        self,
        site_id: int,
        as_of_date: str,
        status: str = "all",
    ) -> dict[str, Any]:
        if status not in {"all", "active", "producing", "injection", "inactive"}:
            raise OnrrClientError(
                "status must be all, active, producing, injection, or inactive."
            )
        day = self._parse_date(as_of_date)
        _, day_end = self._day_bounds(day)
        rows = self._wells_with_onrr_state(site_id, day_end)

        groups = {
            "active": [],
            "producing": [],
            "injection": [],
            "inactive": [],
        }
        for row in rows:
            compact = _compact_well_onrr_state(row)
            if compact["onrr_active_well"]:
                groups["active"].append(compact)
            else:
                groups["inactive"].append(compact)
            if compact["onrr_active_well"] and not compact["onrr_injection_well"]:
                groups["producing"].append(compact)
            if compact["onrr_injection_well"]:
                groups["injection"].append(compact)

        selected = None if status == "all" else groups[status]
        return {
            "site_id": site_id,
            "date": day.isoformat(),
            "status": status,
            "counts": {name: len(values) for name, values in groups.items()},
            "wells": selected,
            "rules": _onrr_rules(day_end),
        }

    def _well_onrr_state(
        self, site_id: int, well_name: str, as_of_time: datetime
    ) -> dict[str, Any] | None:
        query = text(
            f"""
            {WELL_ONRR_STATE_SELECT}
            WHERE w.site_id = :site_id
                AND LOWER(w.name) LIKE LOWER(:well_name)
            ORDER BY w.name
            LIMIT 1
            """
        )
        rows = self._execute(
            query,
            site_id=site_id,
            well_name=f"%{well_name.strip()}%",
            as_of_time=as_of_time,
        )
        return rows[0] if rows else None

    def _wells_with_onrr_state(
        self, site_id: int, as_of_time: datetime
    ) -> list[dict[str, Any]]:
        query = text(
            f"""
            {WELL_ONRR_STATE_SELECT}
            WHERE w.site_id = :site_id
            ORDER BY w.name
            LIMIT :limit
            """
        )
        return self._execute(query, site_id=site_id, as_of_time=as_of_time)

    def _execute(self, query, **params: Any) -> list[dict[str, Any]]:
        params.setdefault("limit", self.max_rows)
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, params)
                return [self._serialize_row(dict(row._mapping)) for row in result]
        except SQLAlchemyError as exc:
            raise OnrrClientError(f"Database query failed: {exc}") from exc

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
            raise OnrrClientError("Dates must use YYYY-MM-DD format.") from exc

    @staticmethod
    def _day_bounds(day: date) -> tuple[datetime, datetime]:
        start = datetime.combine(day, time.min)
        return start, start + timedelta(days=1)


class UnavailableOnrrClient:
    def __init__(self, reason: str):
        self.reason = reason

    def list_onrr_codes(self) -> dict[str, Any]:
        raise OnrrClientError(self.reason)

    def explain_onrr_code(self, code: str) -> dict[str, Any]:
        raise OnrrClientError(self.reason)

    def get_well_onrr_status_as_of_date(
        self, site_id: int, well_name: str, as_of_date: str
    ) -> dict[str, Any]:
        raise OnrrClientError(self.reason)

    def count_wells_by_onrr_status(
        self, site_id: int, as_of_date: str, status: str = "all"
    ) -> dict[str, Any]:
        raise OnrrClientError(self.reason)


WELL_ONRR_STATE_SELECT = """
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
        oc.name AS onrr_code,
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
"""


def _compact_onrr_code(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": row.get("name"),
        "active_well": bool(row.get("active_well")),
        "injection_well": bool(row.get("injection_well")),
        "description": row.get("description"),
    }


def _compact_well_onrr_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "well": f"Well - {row['well_name']}",
        "onrr_code": row.get("onrr_code"),
        "onrr_active_well": bool(row.get("active_well")),
        "onrr_injection_well": bool(row.get("injection_well")),
        "onrr_code_description": row.get("onrr_code_description"),
    }


def _onrr_rules(as_of_time: datetime) -> dict[str, Any]:
    return {
        "onrr_code_as_of": as_of_time.isoformat(sep=" "),
        "active_well_rule": "active_well=true on the effective ONRR code.",
        "producing_well_rule": (
            "active_well=true and injection_well=false on the effective ONRR code."
        ),
        "injection_well_rule": "injection_well=true on the effective ONRR code.",
    }
