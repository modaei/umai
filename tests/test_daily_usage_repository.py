from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from umai.repositories.daily_usage_repository import (
    DailyUsageLimitExceeded,
    DailyUsageRepository,
)


def make_repository(
    request_limit: int = 2,
    enabled: bool = True,
) -> DailyUsageRepository:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE ai_daily_usage_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    site_id INTEGER NOT NULL,
                    usage_date DATE NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    created_at DATETIME,
                    updated_at DATETIME,
                    UNIQUE(user_id, site_id, usage_date)
                )
                """
            )
        )
    return DailyUsageRepository(
        engine,
        request_limit=request_limit,
        timezone_name="UTC",
        enabled=enabled,
    )


def usage_count(
    repository: DailyUsageRepository,
    user_id: int = 9,
    site_id: int | None = None,
) -> int:
    site_filter = "" if site_id is None else " AND site_id = :site_id"
    params = {"user_id": user_id, "site_id": site_id}
    with repository.engine.connect() as connection:
        value = connection.execute(
            text(
                f"""
                SELECT COALESCE(SUM(request_count), 0)
                FROM ai_daily_usage_limits
                WHERE user_id = :user_id{site_filter}
                """
            ),
            params,
        ).scalar_one()
    return int(value)


def test_consume_creates_daily_usage_row():
    repository = make_repository(request_limit=2)

    result = repository.consume(user_id=9, site_id=4)

    assert result is not None
    assert result.limit == 2
    assert result.used == 1
    assert result.remaining == 1
    assert result.reset_at.tzinfo is not None
    assert usage_count(repository) == 1


def test_consume_updates_existing_daily_usage_row():
    repository = make_repository(request_limit=2)

    first = repository.consume(user_id=9, site_id=4)
    second = repository.consume(user_id=9, site_id=4)

    assert first is not None
    assert second is not None
    assert first.used == 1
    assert second.used == 2
    assert second.remaining == 0
    assert usage_count(repository) == 2


def test_consume_rejects_when_limit_is_reached():
    repository = make_repository(request_limit=1)

    repository.consume(user_id=9, site_id=4)
    with pytest.raises(DailyUsageLimitExceeded) as exc:
        repository.consume(user_id=9, site_id=4)

    assert exc.value.limit == 1
    assert exc.value.reset_at > datetime.now(timezone.utc)
    assert usage_count(repository) == 1


def test_disabled_repository_does_not_touch_usage_table():
    repository = make_repository(request_limit=1, enabled=False)

    result = repository.consume(user_id=9, site_id=4)

    assert result is None
    assert usage_count(repository) == 0


def test_usage_limit_is_scoped_by_user_and_site():
    repository = make_repository(request_limit=1)

    first_site = repository.consume(user_id=9, site_id=4)
    second_site = repository.consume(user_id=9, site_id=5)

    assert first_site is not None
    assert second_site is not None
    assert first_site.used == 1
    assert second_site.used == 1
    assert usage_count(repository, user_id=9, site_id=4) == 1
    assert usage_count(repository, user_id=9, site_id=5) == 1
