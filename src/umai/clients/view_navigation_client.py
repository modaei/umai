from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

from umai.config.settings import Settings


REPORT_VIEWS = {
    "oil_sale", "oil_production", "injection_allocation",
    "production_allocation", "water_production", "water_transfer",
    "flared_vent", "fuel_gas", "battery", "water_injection",
}


@dataclass(frozen=True)
class OperationalView:
    """SQL metadata matching one operational index page's filters."""

    from_sql: str
    date_column: str
    search_column: str | None
    record_kind: str
    extra_where: str = ""
    default_status: str | None = None


OPERATIONAL_VIEWS: dict[str, OperationalView] = {
    "flare_readings": OperationalView("flare_readings r JOIN flares e ON e.id=r.flare_id", "r.time", "e.name", "flare_reading"),
    "flow_meter_readings": OperationalView("flow_meter_readings r JOIN flow_meters e ON e.id=r.flow_meter_id", "r.time", "e.name", "flow_meter_reading"),
    "pump_readings": OperationalView("pump_readings r JOIN pumps e ON e.id=r.pump_id", "r.time", "e.name", "pump_reading"),
    "treater_readings": OperationalView("treater_readings r JOIN treaters e ON e.id=r.treater_id", "r.time", "e.name", "treater_reading"),
    "knockout_readings": OperationalView("knock_out_readings r JOIN knock_outs e ON e.id=r.knock_out_id", "r.time", "e.name", "knockout_reading"),
    "water_plant_readings": OperationalView("water_plant_readings r JOIN water_plants e ON e.id=r.water_plant_id", "r.time", "e.name", "water_plant_reading"),
    "lact_readings": OperationalView("lact_readings r JOIN lacts e ON e.id=r.lact_id", "r.time", "e.name", "lact_reading"),
    "well_tests": OperationalView("well_tests r JOIN wells e ON e.id=r.well_id", "r.time", "e.name", "well_test"),
    "well_fluids": OperationalView("well_fluids r JOIN wells e ON e.id=r.well_id", "r.time", "e.name", "well_fluid"),
    "well_injections": OperationalView("well_injections r JOIN wells e ON e.id=r.well_id", "r.time", "e.name", "well_injection"),
    "shutdowns": OperationalView("well_shutdowns r JOIN wells e ON e.id=r.well_id", "r.`start`", "e.name", "shutdown"),
    "general_notes": OperationalView("general_notes r", "r.date", "r.comments", "general_note"),
    "work_orders": OperationalView("work_orders r", "r.time", "r.subject", "work_order", default_status="open"),
    "run_tickets": OperationalView("run_tickets r JOIN tanks e ON e.id=r.tank_id", "r.time", "e.name", "run_ticket"),
    "water_draws": OperationalView("water_draws r JOIN tanks e ON e.id=r.tank_id", "r.time", None, "water_draw"),
}


class ViewNavigationClient:
    """Prepare report/list/detail navigation without modifying operational data."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings) -> "ViewNavigationClient":
        settings.validate_database()
        return cls(create_engine(
            URL.create(
                "mysql+pymysql", username=settings.db_user,
                password=settings.db_password, host=settings.db_host,
                port=settings.db_port, database=settings.db_name,
            ),
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        ))

    def prepare(self, site_id: int, view_type: str, current_date: str,
                start_date: str | None = None, end_date: str | None = None,
                search_text: str | None = None,
                status: str | None = None) -> dict[str, Any]:
        """Return a validated navigation intent or a user-facing stop reason."""
        if view_type in REPORT_VIEWS:
            end = _parse_date(end_date or current_date, "end_date")
            start = _parse_date(start_date, "start_date") if start_date else end - timedelta(days=6)
            if start > end:
                return {"status": "invalid", "message": "start_date cannot be after end_date."}
            return {
                "status": "ready", "destination": "report",
                "view_type": view_type,
                "filters": {"startDate": start.isoformat(), "endDate": end.isoformat()},
            }

        if view_type == "tank_readings":
            return self._prepare_tank_readings(site_id, start_date, end_date, search_text)

        definition = OPERATIONAL_VIEWS.get(view_type)
        if definition is None:
            return {"status": "unsupported", "message": f"Unsupported view type: {view_type}."}
        if search_text and definition.search_column is None:
            return {"status": "invalid", "message": f"The {view_type.replace('_', ' ')} page does not support text search."}
        return self._prepare_operational(
            site_id, view_type, definition, start_date, end_date,
            search_text, status,
        )

    def _prepare_operational(self, site_id: int, view_type: str,
                             definition: OperationalView,
                             start_date: str | None, end_date: str | None,
                             search_text: str | None,
                             status: str | None) -> dict[str, Any]:
        filters, clauses, params = _filters(start_date, end_date, search_text)
        clauses.insert(0, "e.site_id=:site_id" if " JOIN " in definition.from_sql else "r.site_id=:site_id")
        params["site_id"] = site_id
        if definition.extra_where:
            clauses.append(definition.extra_where)
        if search_text:
            clauses.append(f"LOWER({definition.search_column}) LIKE LOWER(:search)")
        selected_status = status or definition.default_status
        if selected_status and selected_status != "all":
            clauses.append("r.status=:status")
            params["status"] = selected_status
            filters["selectedStatus"] = selected_status
        rows = self._rows(definition.from_sql, definition.date_column, clauses, params, definition.record_kind)
        return _result(view_type, filters, rows)

    def _prepare_tank_readings(self, site_id: int, start_date: str | None,
                               end_date: str | None,
                               search_text: str | None) -> dict[str, Any]:
        filters, clauses, params = _filters(start_date, end_date, search_text)
        clauses.insert(0, "e.site_id=:site_id")
        params["site_id"] = site_id
        if search_text:
            clauses.append("LOWER(e.name) LIKE LOWER(:search)")
        clauses.extend(_date_clauses("r.time", params))
        where = " AND ".join(clauses)
        unions = []
        for table_name, kind in (
            ("linear_tank_readings", "linear_tank_reading"),
            ("non_linear_tank_readings", "non_linear_tank_reading"),
            ("mixed_tank_readings", "mixed_tank_reading"),
        ):
            unions.append(
                f"SELECT r.id, '{kind}' AS record_kind FROM {table_name} r "
                f"JOIN tanks e ON e.id=r.tank_id WHERE {where}"
            )
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                text(" UNION ALL ".join(unions) + " LIMIT 2"), params
            ).mappings()]
        return _result("tank_readings", filters, rows)

    def _rows(self, from_sql: str, date_column: str, clauses: list[str],
              params: dict[str, Any], record_kind: str) -> list[dict[str, Any]]:
        clauses.extend(_date_clauses(date_column, params))
        query = text(
            f"SELECT r.id, '{record_kind}' AS record_kind FROM {from_sql} "
            f"WHERE {' AND '.join(clauses)} LIMIT 2"
        )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(query, params).mappings()]


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format.") from exc


def _filters(start_date: str | None, end_date: str | None,
             search_text: str | None) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    filters: dict[str, str] = {}
    params: dict[str, Any] = {}
    if start_date:
        params["start_date"] = _parse_date(start_date, "start_date").isoformat()
        filters["startDate"] = params["start_date"]
    if end_date:
        params["end_date"] = _parse_date(end_date, "end_date").isoformat()
        filters["endDate"] = params["end_date"]
    if start_date and end_date and params["start_date"] > params["end_date"]:
        raise ValueError("start_date cannot be after end_date.")
    if search_text and search_text.strip():
        filters["query"] = search_text.strip()
        params["search"] = f"%{search_text.strip()}%"
    return filters, [], params


def _date_clauses(date_column: str, params: dict[str, Any]) -> list[str]:
    clauses = []
    if "start_date" in params:
        clauses.append(f"DATE({date_column})>=DATE(:start_date)")
    if "end_date" in params:
        clauses.append(f"DATE({date_column})<=DATE(:end_date)")
    return clauses


def _result(view_type: str, filters: dict[str, str],
            rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "no_results", "view_type": view_type, "message": "No matching records were found."}
    if len(rows) == 1:
        return {
            "status": "ready", "destination": "detail",
            "view_type": view_type, "filters": filters,
            "record_id": int(rows[0]["id"]),
            "record_kind": rows[0]["record_kind"],
        }
    return {"status": "ready", "destination": "index", "view_type": view_type, "filters": filters}
