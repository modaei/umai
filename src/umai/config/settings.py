from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from umai.config.logging import LOG_LEVELS


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    umreports_api_url: str
    umreports_timeout_seconds: float
    max_report_days: int
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    db_pool_size: int
    db_max_overflow: int
    db_pool_timeout: int
    db_pool_recycle: int
    operational_sql_db_user: str
    operational_sql_db_password: str
    operational_sql_max_rows: int
    operational_sql_timeout_seconds: int
    umai_max_concurrent: int
    umai_slot_timeout: float
    umai_conversation_history_limit: int
    umai_conversation_ttl_hours: int
    umai_daily_user_limit_enabled: bool
    umai_daily_user_limit_requests: int
    timezone: str
    vector_db_host: str
    vector_db_port: int
    vector_db_user: str
    vector_db_password: str
    vector_db_name: str
    vector_db_pool_size: int
    vector_db_max_overflow: int
    vector_db_pool_timeout: int
    vector_db_pool_recycle: int
    rag_embedding_model: str
    rag_embedding_dimensions: int
    log_level: str
    monitoring_data_api_url: str = "http://metrics1.ultimatesys.com/render"
    data_point_trend_timeout_seconds: float = 20
    data_point_trend_max_data_points: int = 300

    db_archive_host: str = ""
    db_archive_port: int | None = None
    db_archive_user: str = ""
    db_archive_password: str | None = None
    db_archive_name: str = ""
    rod_pump_health_llm_model: str = ""
    rod_pump_health_llm_effort: str = "medium"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_archive_host=os.getenv("DB_ARCHIVE_HOST", "").strip(),
            db_archive_port=int(os.getenv("DB_ARCHIVE_PORT", "0") or "0") or None,
            db_archive_user=os.getenv("DB_ARCHIVE_USERNAME", os.getenv("DB_ARCHIVE_USER", "")).strip(),
            db_archive_password=os.getenv("DB_ARCHIVE_PASSWORD") or None,
            db_archive_name=os.getenv("DB_ARCHIVE_DATABASE", "").strip(),
            llm_api_key=os.getenv(
                "LLM_API_KEY", os.getenv("OPENROUTER_API_KEY", "")
            ).strip(),
            llm_model=os.getenv(
                "LLM_MODEL", os.getenv("OPENROUTER_MODEL", "gpt-5-mini")
            ).strip(),
            llm_base_url=os.getenv(
                "LLM_BASE_URL",
                os.getenv("OPENROUTER_BASE_URL", "https://api.openai.com/v1"),
            ).rstrip("/"),
            umreports_api_url=os.getenv(
                "UMREPORTS_API_URL", "http://127.0.0.1:50008/report/"
            ).strip(),
            umreports_timeout_seconds=float(
                os.getenv("UMREPORTS_TIMEOUT_SECONDS", "30")
            ),
            max_report_days=int(os.getenv("MAX_REPORT_DAYS", "366")),
            monitoring_data_api_url=os.getenv(
                "MONITORING_DATA_API_URL", "http://metrics1.ultimatesys.com/render"
            ).strip(),
            data_point_trend_timeout_seconds=float(
                os.getenv("DATA_POINT_TREND_TIMEOUT_SECONDS", "20")
            ),
            data_point_trend_max_data_points=int(
                os.getenv("DATA_POINT_TREND_MAX_DATA_POINTS", "300")
            ),
            db_host=os.getenv("DB_HOST", "127.0.0.1").strip(),
            db_port=int(os.getenv("DB_PORT", "3306")),
            db_user=os.getenv("DB_USER", "").strip(),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_name=os.getenv("DB_NAME", "").strip(),
            db_pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            db_max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
            db_pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "15")),
            db_pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
            operational_sql_db_user=os.getenv(
                "OPERATIONAL_SQL_DB_USER",
                os.getenv("DB_USER", ""),
            ).strip(),
            operational_sql_db_password=os.getenv(
                "OPERATIONAL_SQL_DB_PASSWORD",
                os.getenv("DB_PASSWORD", ""),
            ),
            operational_sql_max_rows=int(os.getenv("OPERATIONAL_SQL_MAX_ROWS", "100")),
            operational_sql_timeout_seconds=int(
                os.getenv("OPERATIONAL_SQL_TIMEOUT_SECONDS", "10")
            ),
            umai_max_concurrent=int(os.getenv("UMAI_MAX_CONCURRENT", "10")),
            umai_slot_timeout=float(os.getenv("UMAI_SLOT_TIMEOUT", "15")),
            umai_conversation_history_limit=int(
                os.getenv("UMAI_CONVERSATION_HISTORY_LIMIT", "20")
            ),
            umai_conversation_ttl_hours=int(
                os.getenv("UMAI_CONVERSATION_TTL_HOURS", "168")
            ),
            umai_daily_user_limit_enabled=_env_bool(
                "UMAI_DAILY_USER_LIMIT_ENABLED", False
            ),
            umai_daily_user_limit_requests=int(
                os.getenv("UMAI_DAILY_USER_LIMIT_REQUESTS", "150")
            ),
            timezone=os.getenv("TIMEZONE", "UTC").strip(),
            vector_db_host=os.getenv("VECTOR_DB_HOST", "127.0.0.1").strip(),
            vector_db_port=int(os.getenv("VECTOR_DB_PORT", "5432")),
            vector_db_user=os.getenv("VECTOR_DB_USER", "ometrics").strip(),
            vector_db_password=os.getenv("VECTOR_DB_PASSWORD", ""),
            vector_db_name=os.getenv("VECTOR_DB_NAME", "ometrics").strip(),
            vector_db_pool_size=int(os.getenv("VECTOR_DB_POOL_SIZE", "5")),
            vector_db_max_overflow=int(os.getenv("VECTOR_DB_MAX_OVERFLOW", "5")),
            vector_db_pool_timeout=int(os.getenv("VECTOR_DB_POOL_TIMEOUT", "15")),
            vector_db_pool_recycle=int(os.getenv("VECTOR_DB_POOL_RECYCLE", "1800")),
            rag_embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip(),
            rag_embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            # The health check has its own model; it shares the LLM key and base URL.
            rod_pump_health_llm_model=os.getenv(
                "ROD_PUMP_HEALTH_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-5-mini")
            ).strip(),
            rod_pump_health_llm_effort=os.getenv(
                "ROD_PUMP_HEALTH_LLM_EFFORT", "medium"
            ).strip().lower(),
        )

    def validate(self) -> None:
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY is not configured.")
        if self.umreports_timeout_seconds <= 0:
            raise ValueError("UMREPORTS_TIMEOUT_SECONDS must be positive.")
        if self.rod_pump_health_llm_effort not in ("low", "medium", "high", "xhigh"):
            raise ValueError("ROD_PUMP_HEALTH_LLM_EFFORT must be low, medium, high or xhigh.")
        if self.max_report_days <= 0:
            raise ValueError("MAX_REPORT_DAYS must be positive.")
        if not self.monitoring_data_api_url:
            raise ValueError("MONITORING_DATA_API_URL is not configured.")
        if self.data_point_trend_timeout_seconds <= 0:
            raise ValueError("DATA_POINT_TREND_TIMEOUT_SECONDS must be positive.")
        if self.data_point_trend_max_data_points <= 0:
            raise ValueError("DATA_POINT_TREND_MAX_DATA_POINTS must be positive.")
        if self.umai_max_concurrent <= 0:
            raise ValueError("UMAI_MAX_CONCURRENT must be positive.")
        if self.operational_sql_max_rows <= 0:
            raise ValueError("OPERATIONAL_SQL_MAX_ROWS must be positive.")
        if self.operational_sql_timeout_seconds <= 0:
            raise ValueError("OPERATIONAL_SQL_TIMEOUT_SECONDS must be positive.")
        if self.umai_slot_timeout <= 0:
            raise ValueError("UMAI_SLOT_TIMEOUT must be positive.")
        if self.umai_conversation_history_limit <= 0:
            raise ValueError("UMAI_CONVERSATION_HISTORY_LIMIT must be positive.")
        if self.umai_conversation_ttl_hours <= 0:
            raise ValueError("UMAI_CONVERSATION_TTL_HOURS must be positive.")
        if self.umai_daily_user_limit_requests <= 0:
            raise ValueError("UMAI_DAILY_USER_LIMIT_REQUESTS must be positive.")
        if not self.timezone:
            raise ValueError("TIMEZONE is not configured.")
        if not self.vector_db_host:
            raise ValueError("VECTOR_DB_HOST is not configured.")
        if self.vector_db_port <= 0:
            raise ValueError("VECTOR_DB_PORT must be positive.")
        if not self.vector_db_user:
            raise ValueError("VECTOR_DB_USER is not configured.")
        if not self.vector_db_name:
            raise ValueError("VECTOR_DB_NAME is not configured.")
        if self.vector_db_pool_size <= 0:
            raise ValueError("VECTOR_DB_POOL_SIZE must be positive.")
        if self.vector_db_max_overflow < 0:
            raise ValueError("VECTOR_DB_MAX_OVERFLOW cannot be negative.")
        if self.vector_db_pool_timeout <= 0:
            raise ValueError("VECTOR_DB_POOL_TIMEOUT must be positive.")
        if self.vector_db_pool_recycle <= 0:
            raise ValueError("VECTOR_DB_POOL_RECYCLE must be positive.")
        if not self.rag_embedding_model:
            raise ValueError("RAG_EMBEDDING_MODEL is not configured.")
        if self.rag_embedding_dimensions <= 0:
            raise ValueError("RAG_EMBEDDING_DIMENSIONS must be positive.")
        if self.log_level not in LOG_LEVELS:
            allowed = ", ".join(LOG_LEVELS)
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}.")

    def validate_database(self) -> None:
        missing = []
        if not self.db_host:
            missing.append("DB_HOST")
        if not self.db_user:
            missing.append("DB_USER")
        if not self.operational_sql_db_user:
            missing.append("OPERATIONAL_SQL_DB_USER")
        if not self.db_name:
            missing.append("DB_NAME")
        if missing:
            raise ValueError(
                "Database settings are incomplete: " + ", ".join(missing)
            )
        if self.db_port <= 0:
            raise ValueError("DB_PORT must be positive.")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
