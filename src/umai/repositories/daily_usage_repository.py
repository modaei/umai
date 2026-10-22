from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from umai.config.settings import Settings


class DailyUsageRepositoryError(RuntimeError):
    """Raised when daily AI usage cannot be read or updated."""


class DailyUsageLimitExceeded(DailyUsageRepositoryError):
    """Raised when a user has reached the configured daily request limit."""

    def __init__(self, limit: int, reset_at: datetime):
        self.limit = limit
        self.reset_at = reset_at
        super().__init__("Daily AI usage limit reached.")


@dataclass(frozen=True)
class DailyUsageResult:
    limit: int
    used: int
    remaining: int
    reset_at: datetime


class DailyUsageRepository:
    def __init__(
        self,
        engine: Engine,
        request_limit: int,
        timezone_name: str,
        enabled: bool = True,
    ):
        if request_limit <= 0:
            raise ValueError("request_limit must be positive.")
        self.engine = engine
        self.request_limit = request_limit
        self.timezone_name = timezone_name
        self.enabled = enabled

    @classmethod
    def from_settings(cls, settings: Settings) -> "DailyUsageRepository":
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
        return cls(
            engine,
            request_limit=settings.umai_daily_user_limit_requests,
            timezone_name=settings.timezone,
            enabled=settings.umai_daily_user_limit_enabled,
        )

    def consume(self, user_id: int, site_id: int) -> DailyUsageResult | None:
        if not self.enabled:
            return None

        usage_date, now, reset_at = self._current_window()
        try:
            with self.engine.begin() as connection:
                try:
                    connection.execute(
                        text(
                            """
                            INSERT INTO ai_daily_usage_limits
                                    (
                                        user_id,
                                        site_id,
                                        usage_date,
                                        request_count,
                                        created_at,
                                    updated_at
                                    )
                                VALUES
                                    (:user_id, :site_id, :usage_date, 1, :now, :now)
                            """
                        ),
                        {
                            "user_id": user_id,
                            "site_id": site_id,
                            "usage_date": usage_date,
                            "now": now,
                        },
                    )
                    return DailyUsageResult(
                        limit=self.request_limit,
                        used=1,
                        remaining=self.request_limit - 1,
                        reset_at=reset_at,
                    )
                except IntegrityError:
                    pass

                result = connection.execute(
                    text(
                        """
                        UPDATE ai_daily_usage_limits
                        SET request_count = request_count + 1,
                            updated_at = :now
                        WHERE user_id = :user_id
                            AND site_id = :site_id
                            AND usage_date = :usage_date
                            AND request_count < :request_limit
                        """
                    ),
                    {
                        "now": now,
                        "user_id": user_id,
                        "site_id": site_id,
                        "usage_date": usage_date,
                        "request_limit": self.request_limit,
                    },
                )
                used = (
                    connection.execute(
                        text(
                            """
                            SELECT request_count
                            FROM ai_daily_usage_limits
                            WHERE user_id = :user_id
                                AND site_id = :site_id
                                AND usage_date = :usage_date
                            LIMIT 1
                            """
                        ),
                        {
                            "user_id": user_id,
                            "site_id": site_id,
                            "usage_date": usage_date,
                        },
                    )
                    .mappings()
                    .first()
                )
                if used is None:
                    raise DailyUsageRepositoryError(
                        "Could not read daily AI usage after update."
                    )

                used_count = int(used["request_count"])
                if result.rowcount == 0 and used_count >= self.request_limit:
                    raise DailyUsageLimitExceeded(
                        limit=self.request_limit,
                        reset_at=reset_at,
                    )
        except DailyUsageLimitExceeded:
            raise
        except SQLAlchemyError as exc:
            raise DailyUsageRepositoryError(
                f"Could not update daily AI usage: {exc}"
            ) from exc

        return DailyUsageResult(
            limit=self.request_limit,
            used=used_count,
            remaining=self.request_limit - used_count,
            reset_at=reset_at,
        )

    def _current_window(self) -> tuple[date, datetime, datetime]:
        try:
            tz = ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise DailyUsageRepositoryError(
                f"Invalid daily usage timezone: {self.timezone_name}"
            ) from exc

        current = datetime.now(tz)
        next_day = datetime.combine(current.date(), time.min, tzinfo=tz) + timedelta(
            days=1
        )
        return (
            current.date(),
            datetime.now(timezone.utc).replace(tzinfo=None),
            next_day.astimezone(timezone.utc),
        )
