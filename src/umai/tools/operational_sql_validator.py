from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping


class OperationalSqlValidationError(ValueError):
    """Raised when a drafted operational SQL query violates safety rules."""


@dataclass(frozen=True)
class OperationalSqlValidationResult:
    """Structured result returned after validating a drafted SQL statement."""

    sql: str
    tables: list[str]
    aliases: dict[str, str]
    referenced_columns: dict[str, list[str]]
    limit: int
    warnings: list[str] = field(default_factory=list)


# This allowlist mirrors knowledge/database_schema.md. Keeping it in code lets
# the validator enforce the documented contract instead of trusting the prompt.
ALLOWED_TABLES = {
    "alarm_events",
    "alarm_logs",
    "batteries",
    "chart_notes",
    "data_points",
    "data_point_data",
    "flares",
    "flare_readings",
    "flow_meters",
    "flow_meter_readings",
    "general_notes",
    "knock_outs",
    "knock_out_readings",
    "lacts",
    "lact_readings",
    "linear_tank_readings",
    "mixed_tank_readings",
    "non_linear_tank_readings",
    "non_linear_tank_volume_mappings",
    "non_linear_tank_volume_mapping_details",
    "onrr_codes",
    "pumps",
    "pump_readings",
    "rod_pump_cards",
    "rod_pump_monitoring_data",
    "rod_pump_monitoring_data_cards",
    "rod_pump_notes",
    "run_tickets",
    "silent_profiles",
    "tanks",
    "treaters",
    "treater_readings",
    "water_draws",
    "water_plants",
    "water_plant_readings",
    "wells",
    "well_documents",
    "well_fluids",
    "well_histories",
    "well_injections",
    "well_shutdowns",
    "well_tests",
    "work_orders",
    "work_order_notes",
}


# These tables can be site-filtered directly with `<alias>.site_id = :site_id`.
SITE_SCOPED_TABLES = {
    "batteries",
    "chart_notes",
    "data_points",
    "flares",
    "flow_meters",
    "general_notes",
    "knock_outs",
    "lacts",
    "pumps",
    "silent_profiles",
    "tanks",
    "treaters",
    "water_plants",
    "wells",
    "work_orders",
}


# Child tables must join through these parents to prove site scope. Multi-hop
# relations are represented one hop at a time and resolved recursively.
SITE_PARENT_BY_CHILD = {
    "alarm_events": "data_points",
    "alarm_logs": "data_points",
    "data_point_data": "data_points",
    "flare_readings": "flares",
    "flow_meter_readings": "flow_meters",
    "knock_out_readings": "knock_outs",
    "lact_readings": "lacts",
    "linear_tank_readings": "tanks",
    "mixed_tank_readings": "tanks",
    "non_linear_tank_readings": "tanks",
    "non_linear_tank_volume_mappings": "tanks",
    "non_linear_tank_volume_mapping_details": "non_linear_tank_volume_mappings",
    "pump_readings": "pumps",
    "rod_pump_cards": "rod_pump_monitoring_data_cards",
    "rod_pump_monitoring_data": "wells",
    "rod_pump_monitoring_data_cards": "rod_pump_monitoring_data",
    "rod_pump_notes": "wells",
    "run_tickets": "tanks",
    "treater_readings": "treaters",
    "water_draws": "tanks",
    "water_plant_readings": "water_plants",
    "well_documents": "wells",
    "well_fluids": "wells",
    "well_histories": "wells",
    "well_injections": "wells",
    "well_shutdowns": "wells",
    "well_tests": "wells",
    "work_order_notes": "work_orders",
}


# Global code/reference tables do not need a site filter by themselves.
GLOBAL_REFERENCE_TABLES = {"onrr_codes"}


DISALLOWED_KEYWORDS = {
    "alter",
    "create",
    "delete",
    "drop",
    "insert",
    "replace",
    "truncate",
    "update",
}


RESERVED_ALIASES = {
    "where",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "cross",
    "full",
    "on",
    "group",
    "order",
    "limit",
    "having",
    "union",
}


TABLE_PATTERN = re.compile(
    r"\b(?:from|join)\s+((?:`?[a-zA-Z_][\w]*`?\.)?`?[a-zA-Z_][\w]*`?)"
    r"(?:\s+(?:as\s+)?"
    r"(?!where\b|join\b|left\b|right\b|inner\b|outer\b|cross\b|full\b|on\b|group\b|order\b|limit\b|having\b|union\b)"
    r"(`?[a-zA-Z_][\w]*`?))?",
    re.IGNORECASE,
)
COLUMN_PATTERN = re.compile(
    r"(?<![\w`])`?([a-zA-Z_][\w]*)`?\.`?([a-zA-Z_][\w]*)`?"
)
COMMENT_PATTERN = re.compile(r"(--|#|/\*)")
SELECT_NON_COLUMN_IDENTIFIERS = {
    "as",
    "case",
    "cast",
    "coalesce",
    "concat",
    "count",
    "date",
    "date_format",
    "distinct",
    "else",
    "end",
    "false",
    "from",
    "ifnull",
    "max",
    "min",
    "null",
    "round",
    "select",
    "sum",
    "then",
    "true",
    "when",
}
DERIVED_TABLE_ALIAS = "__derived__"


class OperationalSqlValidator:
    """Validates LLM-drafted operational SQL without executing it.

    The validator is intentionally conservative. It does not try to be a full SQL
    parser; it enforces the small subset needed for safe drafts: a single SELECT,
    allowlisted tables, known columns, LIMIT, and selected-site scoping.
    """

    def __init__(self, table_columns: Mapping[str, set[str]]):
        """Receive table-column metadata for the allowlisted operational tables."""
        self.table_columns = {
            table.lower(): {column.lower() for column in columns}
            for table, columns in table_columns.items()
        }

    def validate(self, sql: str) -> OperationalSqlValidationResult:
        """Validate a non-executed SQL draft and return parsed metadata."""
        normalized_sql = self._normalize_sql(sql)
        limit = self._validate_statement_shape(normalized_sql)
        aliases = self._extract_table_aliases(normalized_sql)
        tables = sorted(
            table for table in set(aliases.values()) if table != DERIVED_TABLE_ALIAS
        )

        self._validate_allowed_tables(tables)
        self._validate_known_tables(tables)
        referenced_columns = self._validate_columns(normalized_sql, aliases)
        self._validate_site_scope(normalized_sql, aliases)

        return OperationalSqlValidationResult(
            sql=normalized_sql,
            tables=tables,
            aliases=aliases,
            referenced_columns=referenced_columns,
            limit=limit,
        )

    def referenced_tables(self, sql: str) -> list[str]:
        """Return allowlisted table names referenced by a SQL draft.

        Tools use this lighter parser when full validation fails. Returning table
        names lets the tool show available columns for those tables, which helps
        the model correct bad column guesses without burning extra tool rounds.
        """
        try:
            normalized_sql = self._normalize_sql(sql)
            aliases = self._extract_table_aliases(normalized_sql)
        except OperationalSqlValidationError:
            return []
        return sorted(table for table in set(aliases.values()) if table in ALLOWED_TABLES)

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Collapse whitespace so later regex checks see a stable statement."""
        return " ".join(sql.strip().split())

    def _validate_statement_shape(self, sql: str) -> int:
        """Reject unsafe SQL shapes and return the numeric LIMIT value."""
        if not sql:
            raise OperationalSqlValidationError("SQL draft cannot be empty.")
        if COMMENT_PATTERN.search(sql):
            raise OperationalSqlValidationError("SQL comments are not allowed.")
        if ";" in sql.rstrip(";"):
            raise OperationalSqlValidationError("Only one SQL statement is allowed.")
        sql = sql.rstrip(";").strip()
        if not re.match(r"^select\b", sql, re.IGNORECASE):
            raise OperationalSqlValidationError("Only SELECT statements are allowed.")

        tokens = {token.lower() for token in re.findall(r"\b[a-zA-Z_][\w]*\b", sql)}
        blocked = sorted(tokens & DISALLOWED_KEYWORDS)
        if blocked:
            raise OperationalSqlValidationError(
                f"Disallowed SQL keyword found: {', '.join(blocked)}."
            )
        limit_match = re.search(r"\blimit\s+(\d+)\b", sql, re.IGNORECASE)
        if not limit_match:
            raise OperationalSqlValidationError("SQL draft must include a numeric LIMIT.")
        return int(limit_match.group(1))

    @staticmethod
    def _clean_identifier(value: str) -> str:
        """Remove optional backticks/schema prefix and normalize an identifier."""
        value = value.strip("`").lower()
        return value.split(".")[-1].strip("`")

    def _extract_table_aliases(self, sql: str) -> dict[str, str]:
        """Return alias-to-table mapping for every FROM/JOIN table reference."""
        aliases: dict[str, str] = {}
        for match in TABLE_PATTERN.finditer(sql):
            table = self._clean_identifier(match.group(1))
            alias = match.group(2)
            alias_name = self._clean_identifier(alias) if alias else table
            if alias_name in RESERVED_ALIASES:
                alias_name = table
            aliases[alias_name] = table

        aliases.update(self._extract_derived_table_aliases(sql))

        if not aliases:
            raise OperationalSqlValidationError("SQL draft must reference a table.")
        return aliases

    def _extract_derived_table_aliases(self, sql: str) -> dict[str, str]:
        """Return aliases assigned to derived tables such as `FROM (...) d`.

        Base tables inside the derived query are still discovered by
        ``TABLE_PATTERN`` and validated normally. The derived alias itself is
        accepted so outer references like ``d.month`` or ``r.total`` do not look
        like unknown table aliases.
        """
        aliases: dict[str, str] = {}
        for match in re.finditer(r"\b(?:from|join)\s*\(", sql, re.IGNORECASE):
            depth = 0
            close_index = None
            for index in range(match.end() - 1, len(sql)):
                char = sql[index]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        close_index = index
                        break
            if close_index is None:
                continue

            alias_match = re.match(
                r"\s+(?:as\s+)?`?([a-zA-Z_][\w]*)`?",
                sql[close_index + 1 :],
                re.IGNORECASE,
            )
            if not alias_match:
                continue
            alias = self._clean_identifier(alias_match.group(1))
            if alias in RESERVED_ALIASES:
                continue
            aliases[alias] = DERIVED_TABLE_ALIAS
        return aliases

    @staticmethod
    def _validate_allowed_tables(tables: list[str]) -> None:
        """Reject tables outside the curated allowlist."""
        unknown = sorted(set(tables) - ALLOWED_TABLES)
        if unknown:
            raise OperationalSqlValidationError(
                f"SQL references non-allowlisted table(s): {', '.join(unknown)}."
            )

    def _validate_known_tables(self, tables: list[str]) -> None:
        """Reject allowlisted tables missing from database metadata."""
        missing = sorted(table for table in tables if table not in self.table_columns)
        if missing:
            raise OperationalSqlValidationError(
                f"Column metadata is unavailable for table(s): {', '.join(missing)}."
            )

    def _validate_columns(
        self,
        sql: str,
        aliases: dict[str, str],
    ) -> dict[str, list[str]]:
        """Validate alias-qualified columns against loaded table metadata."""
        referenced: dict[str, set[str]] = {}
        for alias, column in COLUMN_PATTERN.findall(sql):
            alias_name = alias.lower()
            column_name = column.lower()
            table = aliases.get(alias_name)
            if table is None:
                raise OperationalSqlValidationError(
                    f"Unknown table alias used in column reference: {alias}."
                )
            if table == DERIVED_TABLE_ALIAS:
                continue
            if column_name not in self.table_columns.get(table, set()):
                raise OperationalSqlValidationError(
                    f"Unknown column reference: {alias}.{column}."
                )
            referenced.setdefault(table, set()).add(column_name)

        # Unqualified columns cannot be proven safe when joins are present. Require
        # alias-qualified columns for multi-table drafts so validation is reliable.
        if len(set(aliases.values())) > 1:
            self._validate_multitable_select_columns_are_qualified(sql, aliases)

        return {table: sorted(columns) for table, columns in referenced.items()}

    def _validate_multitable_select_columns_are_qualified(
        self,
        sql: str,
        aliases: dict[str, str],
    ) -> None:
        """Reject bare column names in the SELECT list of joined queries.

        This check intentionally looks for unqualified *known column names* rather
        than splitting expressions on commas. Aggregate calls such as
        `ROUND(SUM(ws.hours), 2)` contain commas inside function arguments and are
        valid as long as the real column reference is alias-qualified.
        """
        select_part = re.split(r"\bfrom\b", sql, maxsplit=1, flags=re.IGNORECASE)[0]
        # Remove quoted string literals and already validated qualified column
        # references so their column tokens are not mistaken for bare columns.
        candidate_text = re.sub(r"'(?:''|[^'])*'", " ", select_part)
        candidate_text = COLUMN_PATTERN.sub(" ", candidate_text)

        output_aliases = {
            alias.lower()
            for alias in re.findall(
                r"\bas\s+`?([a-zA-Z_][\w]*)`?",
                candidate_text,
                flags=re.IGNORECASE,
            )
        }
        function_names = {
            function.lower()
            for function in re.findall(
                r"\b([a-zA-Z_][\w]*)\s*\(",
                candidate_text,
            )
        }
        available_columns = {
            column
            for table in set(aliases.values())
            if table != DERIVED_TABLE_ALIAS
            for column in self.table_columns.get(table, set())
        }

        unqualified_columns = []
        for identifier in re.findall(r"`?([a-zA-Z_][\w]*)`?", candidate_text):
            token = identifier.lower()
            if (
                token in SELECT_NON_COLUMN_IDENTIFIERS
                or token in aliases
                or token in output_aliases
                or token in function_names
            ):
                continue
            if token in available_columns:
                unqualified_columns.append(identifier)

        if unqualified_columns:
            raise OperationalSqlValidationError(
                "Columns in multi-table SQL drafts must be qualified with table aliases."
            )

    def _validate_site_scope(self, sql: str, aliases: dict[str, str]) -> None:
        """Ensure every operational table can be traced to a selected-site filter."""
        filtered_site_tables = self._site_filtered_tables(sql, aliases)
        tables = set(aliases.values())

        for table in tables:
            if table == DERIVED_TABLE_ALIAS:
                continue
            if table in GLOBAL_REFERENCE_TABLES:
                continue
            root = self._site_root_for_table(table, tables)
            if root not in filtered_site_tables:
                raise OperationalSqlValidationError(
                    f"Table `{table}` is not safely scoped to the selected site."
                )

    def _site_filtered_tables(self, sql: str, aliases: dict[str, str]) -> set[str]:
        """Find site-scoped tables that are explicitly filtered by `:site_id`."""
        filtered = set()
        for alias, table in aliases.items():
            if table not in SITE_SCOPED_TABLES:
                continue
            # Require a bind parameter rather than a literal number so future
            # execution can safely inject the selected site.
            pattern = rf"\b`?{re.escape(alias)}`?\.`?site_id`?\s*=\s*:site_id\b"
            if re.search(pattern, sql, re.IGNORECASE):
                filtered.add(table)
        return filtered

    def _site_root_for_table(self, table: str, available_tables: set[str]) -> str:
        """Resolve a table to the parent table that must carry the site filter."""
        if table in SITE_SCOPED_TABLES:
            return table
        parent = SITE_PARENT_BY_CHILD.get(table)
        if parent is None:
            raise OperationalSqlValidationError(
                f"No site-scope relationship is defined for table `{table}`."
            )
        if parent not in available_tables:
            raise OperationalSqlValidationError(
                f"Table `{table}` must join through `{parent}` for site scoping."
            )
        return self._site_root_for_table(parent, available_tables)
