from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


class WorkOrderClientError(RuntimeError):
    """Raised when a work order query is invalid or cannot be completed."""


CostMetric = Literal["cost", "estimate", "all"]


FINAL_COST_COLUMN = "final_cost"
ESTIMATE_COST_COLUMN = "cost_estimate"
DATE_COLUMNS = ("time", "created_at", "date")
TEXT_FILTER_COLUMNS = ("subject", "comments", "vendor", "status")


class WorkOrderClient:
    """Read work order cost totals from the source Ometrics database."""

    def __init__(self, engine: Engine):
        """Create a work order client backed by a SQLAlchemy engine."""
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings) -> "WorkOrderClient":
        """Build a MySQL-backed client from application database settings."""
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

    def summarize_costs(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
        metric: str = "all",
        search_text: str | None = None,
    ) -> dict[str, Any]:
        """Summarize final_cost and cost_estimate totals for a site/date range."""
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if start_day > end_day:
            raise WorkOrderClientError("start_date cannot be after end_date.")
        if metric not in {"cost", "estimate", "all"}:
            raise WorkOrderClientError("metric must be cost, estimate, or all.")

        try:
            columns = {
                column["name"]
                for column in inspect(self.engine).get_columns("work_orders")
            }
        except SQLAlchemyError as exc:
            raise WorkOrderClientError(
                f"Could not inspect work_orders table: {exc}"
            ) from exc
        date_column = next((column for column in DATE_COLUMNS if column in columns), None)
        if date_column is None:
            raise WorkOrderClientError(
                "The work_orders table has no supported date column."
            )

        metric_columns = self._metric_columns(metric)
        amount_columns = (FINAL_COST_COLUMN, ESTIMATE_COST_COLUMN)
        missing_metric_columns = [
            column for column in amount_columns if column not in columns
        ]
        if missing_metric_columns:
            raise WorkOrderClientError(
                "The work_orders table is missing required amount columns: "
                + ", ".join(missing_metric_columns)
            )

        text_columns = tuple(column for column in TEXT_FILTER_COLUMNS if column in columns)
        search_clause = ""
        params: dict[str, Any] = {
            "site_id": site_id,
            "start_date": start_day,
            "end_date": end_day,
        }
        if search_text and text_columns:
            search_clause = "AND (" + " OR ".join(
                f"{_identifier(column)} LIKE :search_text" for column in text_columns
            ) + ")"
            params["search_text"] = f"%{search_text}%"

        select_parts = ["COUNT(*) AS work_order_count"]
        for column in amount_columns:
            quoted_column = _identifier(column)
            select_parts.append(f"SUM({quoted_column}) AS {column}_total")
            select_parts.append(
                f"SUM(CASE WHEN {quoted_column} IS NOT NULL THEN 1 ELSE 0 END) "
                f"AS {column}_value_count"
            )

        query = text(
            f"""
            SELECT {", ".join(select_parts)}
            FROM work_orders
            WHERE site_id = :site_id
                AND DATE({_identifier(date_column)}) >= :start_date
                AND DATE({_identifier(date_column)}) <= :end_date
                {search_clause}
            """
        )
        try:
            with self.engine.connect() as connection:
                row = connection.execute(query, params).mappings().one()
        except SQLAlchemyError as exc:
            raise WorkOrderClientError(f"Work order cost summary failed: {exc}") from exc

        totals_by_field = {
            column: _number(row.get(f"{column}_total")) for column in amount_columns
        }
        value_counts_by_field = {
            column: int(row.get(f"{column}_value_count") or 0)
            for column in amount_columns
        }
        primary_field = self._primary_field(metric, value_counts_by_field)
        return {
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "metric": metric,
            "date_column": date_column,
            "work_order_count": int(row.get("work_order_count") or 0),
            "search_text": search_text,
            "primary_total_field": primary_field,
            "primary_total": totals_by_field[primary_field],
            "totals_by_field": totals_by_field,
            "value_counts_by_field": value_counts_by_field,
            "requested_total_fields": metric_columns,
            "requested_totals_by_field": {
                column: totals_by_field[column] for column in metric_columns
            },
            "requested_value_counts_by_field": {
                column: value_counts_by_field[column] for column in metric_columns
            },
        }

    @staticmethod
    def _metric_columns(metric: str) -> tuple[str, ...]:
        """Return the exact work_orders amount columns for a requested metric."""
        if metric == "cost":
            return (FINAL_COST_COLUMN,)
        elif metric == "estimate":
            return (ESTIMATE_COST_COLUMN,)
        return (FINAL_COST_COLUMN, ESTIMATE_COST_COLUMN)

    @staticmethod
    def _primary_field(
        metric: str,
        value_counts_by_field: dict[str, int],
    ) -> str:
        """Choose the field the assistant should treat as the requested total."""
        if metric == "cost":
            return FINAL_COST_COLUMN
        if metric == "estimate":
            return ESTIMATE_COST_COLUMN
        if value_counts_by_field[FINAL_COST_COLUMN] > 0:
            return FINAL_COST_COLUMN
        return ESTIMATE_COST_COLUMN

    @staticmethod
    def _parse_date(value: str) -> date:
        """Parse an ISO date string used in work order filters."""
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise WorkOrderClientError("Dates must use YYYY-MM-DD format.") from exc


class UnavailableWorkOrderClient:
    """Drop-in work order client used when database settings are unavailable."""

    def __init__(self, reason: str):
        """Store the initialization failure reason for later tool responses."""
        self.reason = reason

    def summarize_costs(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
        metric: str = "all",
        search_text: str | None = None,
    ) -> dict[str, Any]:
        """Raise the stored initialization error for any work order query."""
        raise WorkOrderClientError(self.reason)


def _number(value: Any) -> float | int | None:
    """Convert database numeric values into JSON-friendly numbers."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        value = float(value)
    number = float(value)
    if number.is_integer():
        return int(number)
    return round(number, 2)


def _identifier(value: str) -> str:
    """Quote an inspected work_orders column name for SQL text fragments."""
    return f"`{value}`"
