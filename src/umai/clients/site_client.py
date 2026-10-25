from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


class SiteClientError(RuntimeError):
    """Raised when site metadata cannot be loaded."""


class SiteClient:
    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings) -> "SiteClient":
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

    def get_site_name(self, site_id: int) -> str | None:
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text("SELECT name FROM sites WHERE id = :site_id LIMIT 1"),
                    {"site_id": site_id},
                ).mappings().first()
        except SQLAlchemyError as exc:
            raise SiteClientError(f"Site lookup failed: {exc}") from exc

        if result is None:
            return None
        value: Any = result.get("name")
        return str(value) if value else None


class UnavailableSiteClient:
    def __init__(self, reason: str):
        self.reason = reason

    def get_site_name(self, site_id: int) -> str | None:
        raise SiteClientError(self.reason)
