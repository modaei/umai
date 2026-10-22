from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.database_schema_client import (
    DatabaseSchemaClient,
    DatabaseSchemaClientError,
    OperationalSqlExecutionError,
)
from umai.tools.operational_sql_validator import (
    OperationalSqlValidationError,
    OperationalSqlValidator,
)


logger = logging.getLogger(__name__)


MYSQL_SQL_DIALECT_GUIDANCE = {
    "dialect": "mysql_mariadb",
    "rules": [
        "Use LOWER(column) LIKE '%text%' for case-insensitive matching; do not use ILIKE.",
        "Use DATE_FORMAT(column, '%Y-%m') for monthly grouping; do not use DATE_TRUNC.",
        "Use plain string date literals like '2026-05-01'; do not use DATE '2026-05-01'.",
        "Do not use PostgreSQL casts such as ::date.",
        "Use DATE(datetime_column) only when matching a datetime to a date column.",
    ],
}


COMMON_AGGREGATE_HINTS = [
    "ONRR well status: join wells w to onrr_codes oc on oc.id = w.onrr_code_id and filter w.site_id = :site_id. Active wells require oc.active_well = 1. Producing wells should use the dedicated producing-well tool because shutdown state also matters. Injection wells require oc.injection_well = 1.",
    "Well attributes: use case-insensitive comparisons for text attributes such as wells.pump_type, onrr_codes.name, wells.wogcc_class, wells.wogcc_status, wells.direction, and wells.prod_fm. Rod wells mean pump_type ROD; TA wells mean onrr_code TA.",
    "Well allocation totals: do not use SQL over well_tests for specific well or well-group production/injection contribution when the allocation report tool can answer it.",
    "Shutdown totals/reasons: join well_shutdowns ws to wells w, filter w.site_id = :site_id, use ws.`start`/ws.`end`, compute overlap hours for the requested date range, and group by ws.downtime_code or w.name.",
    "Well-test aggregates: join well_tests wt to wells w, filter w.site_id = :site_id, use wt.time and aggregate wt.oil, wt.water, wt.gas, or wt.runtime only for explicit well-test questions.",
    "Well-test battery aggregates: left join batteries b on b.id = w.battery_id and group by COALESCE(b.name, 'No Battery').",
    "Mixed-tank oil volume: join mixed_tank_readings m to tanks t, filter t.site_id = :site_id, compute oil height from top level minus water level, multiply by t.bbl_foot.",
    "Mixed-tank recoverable oil: compute top level minus the greater of water level and tanks.unusable_height, clamp at zero, then multiply by tanks.bbl_foot.",
    "Tank volume SQL: oil_volume, water_volume, and total_volume are computed tool-result fields, not database columns. Do not use tank_readings; use mixed_tank_readings with level fields and tanks.bbl_foot for oil.",
    "Flow meter aggregates: join flow_meter_readings r to flow_meters fm, filter fm.site_id = :site_id, use r.time, r.total, r.flow, r.odometer, and fm.type.",
]

MIXED_TANK_OIL_VOLUME_SQL = (
    "((COALESCE(m.top_level_feet, 0) + COALESCE(m.top_level_inches, 0) / 12.0) "
    "- (COALESCE(m.water_level_feet, 0) + COALESCE(m.water_level_inches, 0) / 12.0)) "
    "* t.bbl_foot"
)

MIXED_TANK_RECOVERABLE_OIL_VOLUME_SQL = (
    "GREATEST((COALESCE(m.top_level_feet, 0) + COALESCE(m.top_level_inches, 0) / 12.0) "
    "- GREATEST((COALESCE(m.water_level_feet, 0) + COALESCE(m.water_level_inches, 0) / 12.0), "
    "COALESCE(t.unusable_height, 0)), 0) * t.bbl_foot"
)


class OperationalSqlInput(BaseModel):
    """Arguments shared by the operational SQL draft and execution tools.

    The model supplies the question and SQL, while UMAI supplies the selected
    site_id as a protected bind parameter during execution.
    """

    question: str = Field(
        description="The user's operational question that the SQL draft is meant to answer."
    )
    sql: str = Field(
        description=(
            "A proposed read-only SELECT query. It must include `:site_id` for "
            "site scoping and a numeric LIMIT."
        )
    )
    notes: str | None = Field(
        default=None,
        description="Optional explanation of assumptions, joins, or expected output.",
    )


class OperationalSqlGuidanceInput(BaseModel):
    """Arguments for retrieving targeted SQL guidance before drafting a query."""

    question: str = Field(
        description="The user's operational question that may need SQL."
    )


def _json_result(value: Any) -> str:
    """Serialize tool output compactly for the model."""
    return json.dumps(value, default=str, separators=(",", ":"))


def _available_columns(
    table_names: list[str],
    column_metadata: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Return sorted available columns for the tables referenced by SQL.

    The model commonly guesses names like `duration_hours` or `alarm_date`.
    Including table columns in validation feedback lets it repair the query in
    the next round instead of repeatedly guessing and hitting the tool limit.
    """
    return {
        table: sorted(column_metadata.get(table, set()))
        for table in table_names
        if table in column_metadata
    }


SQL_RELATIONSHIP_GUIDANCE = {
    "equipment_by_battery": {
        "triggers": (
            "battery",
            "equipment",
            "tank",
            "flow meter",
            "lact",
            "flare",
            "knock",
            "treater",
            "water plant",
            "pump",
        ),
        "tables": [
            "batteries",
            "tanks",
            "flow_meters",
            "lacts",
            "flares",
            "knock_outs",
            "treaters",
            "water_plants",
            "pumps",
        ],
        "joins": [
            "Most equipment joins directly to batteries with <equipment>.battery_id = batteries.id.",
            "Pumps join to batteries through water_plants: pumps.water_plant_id = water_plants.id and water_plants.battery_id = batteries.id.",
        ],
        "pitfalls": [
            "Do not use batteries.number; batteries are identified by batteries.name or batteries.key.",
            "For simple inventory questions, prefer list_equipment instead of SQL.",
        ],
        "example_sql": (
            "SELECT t.name AS tank_name, t.key AS tank_key, t.type AS tank_type "
            "FROM tanks t JOIN batteries b ON b.id = t.battery_id "
            "WHERE t.site_id = :site_id "
            "AND LOWER(REPLACE(b.name, ' ', '')) = 'battery6' "
            "LIMIT 100"
        ),
    },
    "reading_aggregates": {
        "triggers": (
            "reading",
            "readings",
            "average",
            "avg",
            "sum",
            "total",
            "pressure",
            "flow",
            "odometer",
            "inlet",
            "oil off",
            "oil in tanks",
            "tank oil",
            "recoverable oil",
            "usable oil",
            "available oil",
            "daily oil",
        ),
        "tables": [
            "mixed_tank_readings",
            "tanks",
            "flow_meter_readings",
            "flow_meters",
            "lact_readings",
            "lacts",
            "flare_readings",
            "flares",
            "knock_out_readings",
            "knock_outs",
            "pump_readings",
            "pumps",
            "treater_readings",
            "treaters",
            "water_plant_readings",
            "water_plants",
        ],
        "joins": [
            "mixed_tank_readings.tank_id = tanks.id",
            "flow_meter_readings.flow_meter_id = flow_meters.id",
            "lact_readings.lact_id = lacts.id",
            "flare_readings.flare_id = flares.id",
            "knock_out_readings.knock_out_id = knock_outs.id",
            "pump_readings.pump_id = pumps.id",
            "treater_readings.treater_id = treaters.id",
            "water_plant_readings.water_plant_id = water_plants.id",
        ],
        "pitfalls": [
            "Use the parent equipment table for site filtering.",
            "Use reading time columns for date ranges.",
            "There is no unified tank_readings table.",
            "oil_volume, water_volume, and total_volume are computed tool-result fields, not database columns.",
            "For mixed tank oil SQL, calculate oil volume as oil height multiplied by tanks.bbl_foot.",
            "For recoverable mixed-tank oil, subtract the greater of water level and tanks.unusable_height from top level, clamp at zero, and multiply by tanks.bbl_foot.",
        ],
        "example_sql": (
            "SELECT ROUND(AVG(d.daily_oil_bbl), 2) AS average_daily_oil_bbl "
            "FROM ("
            "SELECT DATE(m.time) AS reading_date, "
            f"SUM({MIXED_TANK_OIL_VOLUME_SQL}) AS daily_oil_bbl "
            "FROM mixed_tank_readings m "
            "JOIN tanks t ON t.id = m.tank_id "
            "WHERE t.site_id = :site_id "
            "AND m.time >= '2026-06-01' "
            "AND m.time < '2026-06-21' "
            "AND t.type = 'mixed-water-oil' "
            "GROUP BY DATE(m.time)"
            ") d LIMIT 100"
        ),
    },
    "shutdowns": {
        "triggers": ("shutdown", "downtime", "down", "cause", "reason"),
        "tables": ["well_shutdowns", "wells"],
        "joins": ["well_shutdowns.well_id = wells.id"],
        "pitfalls": [
            "Use well_shutdowns.`start` and well_shutdowns.`end` for shutdown intervals.",
            "Compute downtime hours from interval overlap with the requested date range.",
            "Use well_shutdowns.downtime_code for cause/reason grouping.",
        ],
    },
    "alarms": {
        "triggers": ("alarm", "alarms", "data point", "notification"),
        "tables": ["alarm_logs", "alarm_events", "data_points"],
        "joins": [
            "alarm_logs.data_point_id = data_points.id",
            "alarm_events.data_point_id = data_points.id",
        ],
        "pitfalls": [
            "Alarm logs commonly use alarm_logs.created_at for date filtering.",
            "Do not guess alarm_date or alarm_time columns.",
            "For battery-like alarm grouping, use data_points.facility_name when it contains battery names.",
        ],
    },
    "well_status": {
        "triggers": ("active", "producing", "producer", "injection", "onrr", "ta"),
        "tables": ["wells", "onrr_codes", "well_histories"],
        "joins": [
            "wells.onrr_code_id = onrr_codes.id",
            "Historical ONRR status may require well_histories where property = 'onrr_code_id'.",
        ],
        "pitfalls": [
            "Prefer get_active_wells/get_producing_wells for active/producing counts.",
            "Producing wells also require shutdown-aware classification; prefer get_producing_wells.",
        ],
    },
    "work_orders": {
        "triggers": ("work order", "work orders", "cost", "estimate", "repair"),
        "tables": ["work_orders", "work_order_notes"],
        "joins": ["work_order_notes.work_order_id = work_orders.id"],
        "pitfalls": [
            "Use work_orders.site_id = :site_id for site filtering.",
            "Use work_order_notes for note/comment details.",
            "Prefer summarize_work_order_costs for numeric cost totals.",
        ],
    },
}


def _sql_guidance_for_question(
    question: str,
    column_metadata: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Return compact schema/recipe guidance relevant to a user question."""
    normalized = question.lower()
    matched = []
    for name, guidance in SQL_RELATIONSHIP_GUIDANCE.items():
        if any(trigger in normalized for trigger in guidance["triggers"]):
            matched.append({"name": name, **guidance})
    if not matched:
        matched = [
            {
                "name": "general",
                "tables": [],
                "joins": [],
                "pitfalls": [
                    "Use the most specific domain tool first.",
                    "If SQL is needed, use only allowlisted tables, known columns, :site_id, and LIMIT.",
                ],
            }
        ]

    tables = sorted(
        {
            table
            for guidance in matched
            for table in guidance.get("tables", [])
        }
    )
    return {
        "question": question,
        "matched_recipes": matched,
        "relevant_tables": tables,
        "available_columns": _available_columns(tables, column_metadata or {}),
        "sql_dialect": MYSQL_SQL_DIALECT_GUIDANCE,
        "aggregate_hints": [
            hint
            for hint in COMMON_AGGREGATE_HINTS
            if any(word in hint.lower() for word in normalized.split())
        ][:6],
    }


def _repair_hints_for_error(question: str, sql: str, error: str) -> list[str]:
    """Return targeted repair hints for common LLM SQL mistakes."""
    text = f"{question}\n{sql}\n{error}".lower()
    hints = []
    if "b.number" in text or "batteries.number" in text:
        hints.append("There is no batteries.number column. Match batteries by batteries.name or batteries.key.")
    if "tank_type" in text:
        hints.append("Tanks use tanks.type for tank type; do not use tanks.tank_type.")
    if "tank_readings" in text:
        hints.append("There is no unified tank_readings table. Use mixed_tank_readings, linear_tank_readings, or non_linear_tank_readings.")
    if any(field in text for field in ("oil_volume", "water_volume", "total_volume", "recoverable_oil_volume")) and "tank" in text:
        hints.append("For SQL, oil_volume, water_volume, total_volume, and recoverable_oil_volume are computed tool-result fields, not database columns.")
        hints.append(f"For mixed tank oil volume SQL, calculate: {MIXED_TANK_OIL_VOLUME_SQL}.")
        if "recoverable" in text:
            hints.append(f"For mixed tank recoverable oil SQL, calculate: {MIXED_TANK_RECOVERABLE_OIL_VOLUME_SQL}.")
    if "well_name" in text:
        hints.append("Wells use wells.name for display name; do not use wells.well_name.")
    if "production_allocation" in text:
        hints.append("Production allocation is a report output, not an allowlisted SQL table. Use allocation report tools.")
    if "alarm_date" in text or "alarm_time" in text:
        hints.append("Alarm logs commonly use alarm_logs.created_at for date filtering.")
    if "not safely scoped" in error.lower():
        hints.append("Join child tables through their site-scoped parent and filter that parent with <alias>.site_id = :site_id.")
    return hints


def build_database_schema_tools(
    client: DatabaseSchemaClient,
    site_id: int,
    max_rows: int = 100,
) -> list[StructuredTool]:
    """Build curated operational SQL tools for the selected site.

    Existing domain tools should remain the first choice. These tools are the
    flexible second path for operational questions that need a direct database
    query across the curated allowlist.
    """

    def _validate_sql(sql: str):
        """Load metadata and validate the model-supplied SQL once per tool call."""
        # Load live column metadata for the allowlisted tables. This validates
        # columns without trusting the model to know the current schema.
        column_metadata = client.load_column_metadata()
        validator = OperationalSqlValidator(column_metadata)
        validation = validator.validate(sql)
        if validation.limit > max_rows:
            raise OperationalSqlValidationError(
                f"SQL LIMIT cannot be greater than {max_rows}."
            )
        return validation, column_metadata

    def get_operational_sql_guidance(question: str) -> str:
        """Return targeted SQL recipes, joins, and pitfalls for a question."""
        logger.info("Getting operational SQL guidance site_id=%s", site_id)
        try:
            column_metadata = client.load_column_metadata()
        except DatabaseSchemaClientError:
            column_metadata = {}
        return _json_result(
            {
                "ok": True,
                "executed": False,
                "site_id": site_id,
                **_sql_guidance_for_question(question, column_metadata),
                "warning": (
                    "This guidance was not executed. Use a specific domain tool "
                    "if one fits; otherwise draft or execute one SQL query using "
                    "these joins and columns."
                ),
            }
        )

    def draft_operational_sql(
        question: str,
        sql: str,
        notes: str | None = None,
    ) -> str:
        """Return schema context and validation feedback without executing SQL."""
        logger.info("Drafting non-executed operational SQL site_id=%s", site_id)
        try:
            # Load the allowlisted schema at invocation time so edits to the
            # markdown are picked up without rebuilding the Python package.
            schema = client.load_schema()
            validation, column_metadata = _validate_sql(sql)
            # Echo the drafted SQL with explicit executed=false metadata. This is
            # useful when the model needs schema feedback before running a query.
            return _json_result(
                {
                    "ok": True,
                    "executed": False,
                    "site_id": site_id,
                    "question": question,
                    "sql": validation.sql,
                    "notes": notes,
                    "validation": {
                        "valid": True,
                        "tables": validation.tables,
                        "aliases": validation.aliases,
                        "referenced_columns": validation.referenced_columns,
                        "limit": validation.limit,
                        "warnings": validation.warnings,
                    },
                    "available_columns": _available_columns(
                        validation.tables,
                        column_metadata,
                    ),
                    "sql_dialect": MYSQL_SQL_DIALECT_GUIDANCE,
                    "aggregate_hints": COMMON_AGGREGATE_HINTS,
                    "targeted_guidance": _sql_guidance_for_question(
                        question,
                        column_metadata,
                    ),
                    "schema": schema,
                    "warning": (
                        "This SQL was not executed. Use execute_operational_sql "
                        "next with this validated SQL if it is needed to answer "
                        "the user. Do not draft another SQL variant unless this "
                        "validation result shows a specific schema problem."
                    ),
                }
            )
        except OperationalSqlValidationError as exc:
            # Validation failures are expected while the model learns a schema.
            # Return them as data so the model can correct the draft.
            logger.info("Operational SQL draft validation failed: %s", exc)
            try:
                column_metadata = client.load_column_metadata()
                validator = OperationalSqlValidator(column_metadata)
                table_names = validator.referenced_tables(sql)
                available_columns = _available_columns(table_names, column_metadata)
            except DatabaseSchemaClientError:
                column_metadata = {}
                available_columns = {}
                table_names = []
            error = str(exc)
            return _json_result(
                {
                    "ok": False,
                    "executed": False,
                    "site_id": site_id,
                    "question": question,
                    "sql": sql,
                    "validation": {
                        "valid": False,
                        "error": error,
                    },
                    "available_columns": available_columns,
                    "sql_dialect": MYSQL_SQL_DIALECT_GUIDANCE,
                    "aggregate_hints": COMMON_AGGREGATE_HINTS,
                    "repair_hints": _repair_hints_for_error(question, sql, error),
                    "targeted_guidance": _sql_guidance_for_question(
                        question,
                        column_metadata,
                    ),
                }
            )
        except DatabaseSchemaClientError as exc:
            # Return a structured tool error so the assistant can explain that
            # schema context is unavailable instead of failing the chat request.
            logger.warning("Database schema lookup failed: %s", exc)
            return _json_result({"ok": False, "executed": False, "error": str(exc)})

    def execute_operational_sql(
        question: str,
        sql: str,
        notes: str | None = None,
    ) -> str:
        """Run a validated SELECT query and return bounded rows to the model."""
        logger.info("Executing operational SQL site_id=%s", site_id)
        try:
            validation, column_metadata = _validate_sql(sql)
            rows = client.execute_readonly_sql(
                validation.sql,
                site_id=site_id,
                max_rows=max_rows,
            )
            # The successful result includes data rows because this tool is used
            # by the model to answer the user. API clients still do not receive
            # tool payloads directly; the public `/chat` response only returns
            # the final assistant answer.
            return _json_result(
                {
                    "ok": True,
                    "executed": True,
                    "site_id": site_id,
                    "question": question,
                    "sql": validation.sql,
                    "notes": notes,
                    "row_count": len(rows),
                    "max_rows": max_rows,
                    "rows": rows,
                    "available_columns": _available_columns(
                        validation.tables,
                        column_metadata,
                    ),
                    "sql_dialect": MYSQL_SQL_DIALECT_GUIDANCE,
                    "aggregate_hints": COMMON_AGGREGATE_HINTS,
                    "targeted_guidance": _sql_guidance_for_question(
                        question,
                        column_metadata,
                    ),
                    "validation": {
                        "valid": True,
                        "tables": validation.tables,
                        "aliases": validation.aliases,
                        "referenced_columns": validation.referenced_columns,
                        "limit": validation.limit,
                        "warnings": validation.warnings,
                    },
                }
            )
        except OperationalSqlValidationError as exc:
            logger.info("Operational SQL execution validation failed: %s", exc)
            try:
                column_metadata = client.load_column_metadata()
                validator = OperationalSqlValidator(column_metadata)
                table_names = validator.referenced_tables(sql)
                available_columns = _available_columns(table_names, column_metadata)
            except DatabaseSchemaClientError:
                column_metadata = {}
                available_columns = {}
                table_names = []
            error = str(exc)
            return _json_result(
                {
                    "ok": False,
                    "executed": False,
                    "site_id": site_id,
                    "question": question,
                    "sql": sql,
                    "validation": {
                        "valid": False,
                        "error": error,
                    },
                    "available_columns": available_columns,
                    "sql_dialect": MYSQL_SQL_DIALECT_GUIDANCE,
                    "aggregate_hints": COMMON_AGGREGATE_HINTS,
                    "repair_hints": _repair_hints_for_error(question, sql, error),
                    "targeted_guidance": _sql_guidance_for_question(
                        question,
                        column_metadata,
                    ),
                }
            )
        except (DatabaseSchemaClientError, OperationalSqlExecutionError):
            # Do not leak SQLAlchemy, driver, host, username, or schema details
            # into the model context. Full details are kept in application logs.
            logger.exception("Operational SQL execution failed")
            return _json_result(
                {
                    "ok": False,
                    "executed": False,
                    "site_id": site_id,
                    "question": question,
                    "sql": sql,
                    "error": "Operational SQL query failed.",
                }
            )

    return [
        StructuredTool.from_function(
            func=get_operational_sql_guidance,
            name="get_operational_sql_guidance",
            description=(
                "Return targeted MySQL/MariaDB SQL guidance, relevant tables, "
                "known joins, available columns, examples, and pitfalls for an "
                "Ometrics operational question. Use this before drafting SQL "
                "when no specific domain tool fits."
            ),
            args_schema=OperationalSqlGuidanceInput,
        ),
        StructuredTool.from_function(
            func=draft_operational_sql,
            name="draft_operational_sql",
            description=(
                "Validate a read-only SQL query and return curated Ometrics schema "
                "context without executing it. Use this when you need schema feedback "
                "before running execute_operational_sql. If the draft result is valid "
                "and the query answers the user, run execute_operational_sql next with "
                "the same SQL instead of drafting alternate valid queries."
            ),
            args_schema=OperationalSqlInput,
        ),
        StructuredTool.from_function(
            func=execute_operational_sql,
            name="execute_operational_sql",
            description=(
                "Execute a validated read-only SELECT query against curated Ometrics "
                "operational tables. Use it after more specific report, reading, "
                "shutdown, timeline, work-order, and capability tools when a direct "
                "database query is the best way to answer. Queries must use `:site_id` "
                "for selected-site scoping and a numeric LIMIT no greater than the "
                f"configured maximum of {max_rows} rows."
            ),
            args_schema=OperationalSqlInput,
        )
    ]
