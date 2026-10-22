from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings
from umai.tools.operational_sql_validator import ALLOWED_TABLES


class DatabaseSchemaClientError(RuntimeError):
    """Raised when curated database schema context cannot be loaded."""


class OperationalSqlExecutionError(RuntimeError):
    """Raised when a validated read-only SQL query cannot be executed."""


class DatabaseSchemaClient:
    """Loads schema context and runs validated read-only operational SQL.

    The markdown schema and live column metadata are used before execution so
    the LLM cannot freely discover or query tables outside the documented
    allowlist. SQL execution is reserved for already-validated SELECT queries.
    """

    def __init__(self, schema_path: Path | str, engine: Engine | None = None):
        """Store the configured schema document path and optional DB engine."""
        self.schema_path = Path(schema_path)
        self.engine = engine

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        schema_path: Path | str,
    ) -> "DatabaseSchemaClient":
        """Create a schema client backed by the configured Ometrics database.

        The operational SQL credentials should belong to a read-only database
        user in production. Code-level validation is useful, but database
        permissions remain the final safety boundary.
        """
        settings.validate_database()
        url = URL.create(
            drivername="mysql+pymysql",
            username=settings.operational_sql_db_user,
            password=settings.operational_sql_db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
        )
        # PyMySQL supports socket read/write timeouts. These are not a complete
        # query governor, but they prevent the assistant from hanging forever if
        # the database or network stalls during a read-only query.
        connect_args = {
            "read_timeout": settings.operational_sql_timeout_seconds,
            "write_timeout": settings.operational_sql_timeout_seconds,
        }
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        return cls(schema_path, engine)

    def load_schema(self) -> str:
        """Return the schema markdown used by draft-only SQL tooling."""
        try:
            # Keep the schema source explicit and version-controlled.
            return self.schema_path.read_text(encoding="utf-8")
        except OSError as exc:
            # Convert filesystem failures into a domain-specific error so tools
            # can return structured JSON instead of leaking a traceback.
            raise DatabaseSchemaClientError(
                f"Could not load database schema document: {self.schema_path}"
            ) from exc

    def load_column_metadata(self) -> dict[str, set[str]]:
        """Return column names for allowlisted tables from database metadata."""
        if self.engine is None:
            raise DatabaseSchemaClientError("Database schema metadata is unavailable.")

        try:
            inspector = inspect(self.engine)
            existing_tables = set(inspector.get_table_names())
            metadata: dict[str, set[str]] = {}
            for table in sorted(ALLOWED_TABLES & existing_tables):
                # SQLAlchemy's inspector reads database metadata only; it does
                # not execute any LLM-drafted operational SQL.
                metadata[table] = {
                    str(column["name"]) for column in inspector.get_columns(table)
                }
            return metadata
        except SQLAlchemyError as exc:
            raise DatabaseSchemaClientError(
                f"Could not load database column metadata: {exc}"
            ) from exc

    def execute_readonly_sql(
        self,
        sql: str,
        *,
        site_id: int,
        max_rows: int,
    ) -> list[dict[str, Any]]:
        """Execute a pre-validated SELECT query with the selected site bind.

        This method assumes `OperationalSqlValidator` has already accepted the
        SQL. It still uses a bound `:site_id` parameter so the model never embeds
        or controls the selected site's numeric value directly.
        """
        if self.engine is None:
            raise OperationalSqlExecutionError("Operational SQL execution is unavailable.")

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql), {"site_id": site_id})
                # Fetch one extra row defensively. The validator enforces LIMIT,
                # but this keeps returned payloads bounded even if a database
                # dialect behaves unexpectedly.
                return [
                    dict(row)
                    for row in result.mappings().fetchmany(max_rows + 1)[:max_rows]
                ]
        except SQLAlchemyError as exc:
            raise OperationalSqlExecutionError("Operational SQL execution failed.") from exc


class UnavailableDatabaseSchemaClient:
    """Fallback client used when schema setup fails during tool wiring."""

    def __init__(self, reason: str):
        """Store the setup failure reason for later tool calls."""
        self.reason = reason

    def load_schema(self) -> str:
        """Raise the original setup reason when the tool is invoked."""
        raise DatabaseSchemaClientError(self.reason)

    def load_column_metadata(self) -> dict[str, set[str]]:
        """Raise the original setup reason when validation needs metadata."""
        raise DatabaseSchemaClientError(self.reason)

    def execute_readonly_sql(
        self,
        sql: str,
        *,
        site_id: int,
        max_rows: int,
    ) -> list[dict[str, Any]]:
        """Raise the original setup reason when execution is requested."""
        raise OperationalSqlExecutionError(self.reason)
