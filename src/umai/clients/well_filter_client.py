from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


class WellFilterClientError(RuntimeError):
    """Raised when well filter data cannot be queried."""


SUPPORTED_WELL_FILTER_FIELDS = {
    "pump_type",
    "onrr_code",
    "wogcc_class",
    "wogcc_status",
    "direction",
    "prod_fm",
    "battery",
    "lact",
    "monitored",
    "disable_reading",
    "multiple_injection_form",
}


class WellFilterClient:
    """Read small, site-scoped well lists used to filter report results."""

    def __init__(self, engine: Engine, max_rows: int = 1000):
        self.engine = engine
        self.max_rows = max_rows

    @classmethod
    def from_settings(cls, settings: Settings) -> "WellFilterClient":
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

    def find_wells(
        self,
        site_id: int,
        well_name: str | None = None,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return well names matching optional well-name and attribute filters.

        Attribute comparisons are deliberately allowlisted and case-insensitive
        for text fields. This lets the assistant answer natural requests such
        as "rod wells", "TA wells", or "Battery 6 wells" without exposing an
        arbitrary SQL filter surface.
        """
        normalized_filters = _normalize_filters(filters or [])
        conditions = ["w.site_id = :site_id"]
        params: dict[str, Any] = {"site_id": site_id}

        for index, filter_item in enumerate(normalized_filters):
            _add_filter_condition(filter_item, index, conditions, params)

        if well_name:
            conditions.append(
                "(LOWER(w.name) LIKE LOWER(:well_name) OR LOWER(w.`key`) LIKE LOWER(:well_name))"
            )
            params["well_name"] = f"%{well_name.strip()}%"

        query = text(
            f"""
            SELECT w.name,
                w.`key`,
                w.pump_type,
                w.wogcc_class,
                w.wogcc_status,
                w.direction,
                w.prod_fm,
                w.monitored,
                w.disable_reading,
                w.multiple_injection_form,
                oc.name AS onrr_code,
                b.name AS battery,
                l.name AS lact
            FROM wells w
            LEFT JOIN onrr_codes oc ON oc.id = w.onrr_code_id
            LEFT JOIN batteries b ON b.id = w.battery_id
            LEFT JOIN lacts l ON l.id = w.lact_id
            WHERE {" AND ".join(conditions)}
            ORDER BY w.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        return {
            "site_id": site_id,
            "well_name": well_name,
            "filters": normalized_filters,
            "count": len(rows),
            "wells": [
                {
                    "name": row["name"],
                    "key": row.get("key"),
                    "pump_type": row.get("pump_type"),
                    "onrr_code": row.get("onrr_code"),
                    "wogcc_class": row.get("wogcc_class"),
                    "wogcc_status": row.get("wogcc_status"),
                    "direction": row.get("direction"),
                    "prod_fm": row.get("prod_fm"),
                    "battery": row.get("battery"),
                    "lact": row.get("lact"),
                }
                for row in rows
            ],
        }

    def _execute(self, query, **params: Any) -> list[dict[str, Any]]:
        params.setdefault("limit", self.max_rows)
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, params)
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as exc:
            raise WellFilterClientError(f"Database query failed: {exc}") from exc


class UnavailableWellFilterClient:
    def __init__(self, reason: str):
        self.reason = reason

    def find_wells(
        self,
        site_id: int,
        well_name: str | None = None,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise WellFilterClientError(self.reason)


def _normalize_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for filter_item in filters:
        field = str(filter_item.get("field", "")).strip().lower()
        if field not in SUPPORTED_WELL_FILTER_FIELDS:
            raise WellFilterClientError(
                f"Unsupported well filter field: {field}. "
                f"Supported fields: {', '.join(sorted(SUPPORTED_WELL_FILTER_FIELDS))}."
            )
        normalized.append(
            {
                "field": field,
                "value": _normalize_filter_value(field, filter_item.get("value")),
            }
        )
    return normalized


def _normalize_filter_value(field: str, value: Any) -> Any:
    if field in {"monitored", "disable_reading", "multiple_injection_form"}:
        return _normalize_bool(value)
    if value is None:
        return None
    text_value = " ".join(str(value).strip().split())
    if field == "pump_type":
        return _normalize_pump_type(text_value)
    return text_value


def _normalize_pump_type(value: str) -> str | None:
    value = " ".join(value.strip().lower().split())
    if not value:
        return None
    if value in {"rod", "rod pump", "rod pumps", "rod well", "rod wells"}:
        return "ROD"
    if value in {"esp", "esp well", "esp wells"}:
        return "ESP"
    if value in {"jet", "jet well", "jet wells"}:
        return "JET"
    if value in {"flowing", "flowing well", "flowing wells", "no lift", "flowing well no lift"}:
        return "flowing well no lift"
    if value in {"none", "null", "no pump type", "missing pump type"}:
        return "__NULL__"
    return value


def _normalize_bool(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    text_value = str(value).strip().lower()
    if text_value in {"1", "true", "yes", "y", "enabled", "monitored"}:
        return 1
    if text_value in {"0", "false", "no", "n", "disabled", "not monitored"}:
        return 0
    raise WellFilterClientError(f"Boolean well filter value is invalid: {value!r}.")


def _add_filter_condition(
    filter_item: dict[str, Any],
    index: int,
    conditions: list[str],
    params: dict[str, Any],
) -> None:
    field = filter_item["field"]
    value = filter_item["value"]
    param_name = f"filter_{index}"

    if field == "pump_type":
        _add_text_condition("w.pump_type", value, param_name, conditions, params)
    elif field == "onrr_code":
        _add_text_condition("oc.name", value, param_name, conditions, params)
    elif field == "battery":
        _add_battery_condition("b.name", value, param_name, conditions, params)
    elif field == "lact":
        _add_text_condition("l.name", value, param_name, conditions, params, partial=True)
    elif field in {"wogcc_class", "wogcc_status", "direction", "prod_fm"}:
        _add_text_condition(f"w.{field}", value, param_name, conditions, params, partial=True)
    elif field in {"monitored", "disable_reading", "multiple_injection_form"}:
        conditions.append(f"COALESCE(w.{field}, 0) = :{param_name}")
        params[param_name] = value


def _add_text_condition(
    column: str,
    value: Any,
    param_name: str,
    conditions: list[str],
    params: dict[str, Any],
    partial: bool = False,
) -> None:
    if value == "__NULL__":
        conditions.append(f"{column} IS NULL")
        return
    if value is None:
        conditions.append(f"{column} IS NULL")
        return
    if partial:
        conditions.append(f"LOWER({column}) LIKE LOWER(:{param_name})")
        params[param_name] = f"%{value}%"
        return
    conditions.append(f"LOWER({column}) = LOWER(:{param_name})")
    params[param_name] = value


def _add_battery_condition(
    column: str,
    value: Any,
    param_name: str,
    conditions: list[str],
    params: dict[str, Any],
) -> None:
    text_value = "" if value is None else str(value).strip()
    digits = "".join(ch for ch in text_value if ch.isdigit())
    if digits and text_value.lower().replace(" ", "") in {digits, f"battery{digits}"}:
        conditions.append(
            f"LOWER(REPLACE({column}, ' ', '')) IN (:{param_name}_compact, :{param_name}_number)"
        )
        params[f"{param_name}_compact"] = f"battery{digits}"
        params[f"{param_name}_number"] = digits
        return
    _add_text_condition(column, value, param_name, conditions, params, partial=True)
