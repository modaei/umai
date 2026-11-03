from datetime import date, datetime, timezone

from umai.services.chat_service import chat_today

# 20:00 UTC on July 15 is already July 16 in Tokyo but still July 15 in Denver.
NOW = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)


def test_supplied_current_date_wins():
    assert chat_today("2026-07-03", "Asia/Tokyo", "America/Denver", NOW) == date(2026, 7, 3)


def test_today_follows_the_site_clock():
    assert chat_today(None, "Asia/Tokyo", "America/Denver", NOW) == date(2026, 7, 16)


def test_callers_without_a_timezone_use_the_default():
    assert chat_today(None, None, "America/Denver", NOW) == date(2026, 7, 15)
