"""Stored moments are filtered by the site's days and shown on the site's clock."""

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from umai.clients.shutdown_client import ShutdownClient
from umai.rag.extractors.operational_text import OperationalTextExtractor
from umai.site_time import DEFAULT_SITE_TIMEZONE, lookup_site_zone, site_to_storage, storage_to_site

UTC = ZoneInfo("UTC")
TOKYO = ZoneInfo("Asia/Tokyo")


def make_engine():
    """A Tokyo site (id 4) whose shutdowns are stored in UTC."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE sites (id INTEGER PRIMARY KEY, timezone TEXT)"))
        connection.execute(text("INSERT INTO sites (id, timezone) VALUES (4, 'Asia/Tokyo'), (5, NULL)"))
        connection.execute(text("CREATE TABLE wells (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("INSERT INTO wells (id, site_id, name) VALUES (7, 4, '4048')"))
        connection.execute(
            text(
                "CREATE TABLE well_shutdowns (id INTEGER PRIMARY KEY, well_id INTEGER, start TEXT, "
                "end TEXT, downtime_code TEXT, comments TEXT, updated_at TEXT)"
            )
        )
        # 20:00 UTC on July 14 is 05:00 on July 15 in Tokyo; 16:00 UTC on July 15 is July 16.
        connection.execute(
            text(
                "INSERT INTO well_shutdowns (id, well_id, start, end, downtime_code, comments) VALUES "
                "(1, 7, '2026-07-14 20:00:00', '2026-07-14 23:00:00', 'PRF', 'Paraffin'), "
                "(2, 7, '2026-07-15 16:00:00', NULL, 'TL', 'Tubing leak')"
            )
        )
    return engine


def test_helpers_convert_between_site_and_storage_time():
    assert site_to_storage(datetime(2026, 7, 15), TOKYO, UTC) == datetime(2026, 7, 14, 15)
    assert storage_to_site("2026-07-14 20:00:00", TOKYO, UTC) == datetime(2026, 7, 15, 5)
    assert storage_to_site(None, TOKYO, UTC) is None


def test_site_zone_lookup_falls_back():
    engine = make_engine()
    assert lookup_site_zone(engine, 4, UTC) == TOKYO
    assert lookup_site_zone(engine, 5, UTC) == ZoneInfo(DEFAULT_SITE_TIMEZONE)
    # A schema without sites.timezone keeps stored times unchanged.
    bare = create_engine("sqlite:///:memory:")
    assert lookup_site_zone(bare, 4, UTC) == UTC


def test_shutdowns_use_site_days_and_site_times():
    client = ShutdownClient(make_engine(), storage_timezone="UTC")

    # Only the first shutdown falls on July 15 in Tokyo; the second starts on July 16 there.
    result = client.get_shutdowns(4, "2026-07-15", "2026-07-15")

    assert result["count"] == 1
    shutdown = result["shutdowns"][0]
    assert shutdown["start"] == "2026-07-15 05:00:00"
    assert shutdown["end"] == "2026-07-15 08:00:00"
    assert result["total_hours"] == 3.0


def test_rag_shutdowns_are_dated_by_the_site_day():
    extractor = OperationalTextExtractor(make_engine(), storage_timezone="UTC")

    documents = extractor.extract(4, source_types={"well_shutdown"})

    by_id = {document.source_id: document for document in documents}
    assert by_id["1"].event_date.isoformat() == "2026-07-15"
    assert "Start: 2026-07-15 05:00:00" in by_id["1"].text
    assert by_id["2"].event_date.isoformat() == "2026-07-16"


def test_timeline_history_and_chart_notes_use_the_site_clock():
    from umai.clients.well_timeline_client import WellTimelineClient

    engine = make_engine()
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE well_histories (id INTEGER PRIMARY KEY, well_id INTEGER, changed_at TEXT, "
            "property TEXT, old_value TEXT, new_value TEXT, source TEXT)"
        ))
        connection.execute(text(
            "INSERT INTO well_histories (well_id, changed_at, property, new_value) VALUES "
            "(7, '2026-07-14 20:00:00', 'onrr_code_id', '2'), (7, '2026-07-15 16:00:00', 'onrr_code_id', '3')"
        ))
        connection.execute(text(
            "CREATE TABLE chart_notes (id INTEGER PRIMARY KEY, object_id INTEGER, object_type TEXT, "
            "x_axis_value TEXT, chart_name TEXT, note TEXT)"
        ))
        connection.execute(text(
            "INSERT INTO chart_notes (id, object_id, object_type, x_axis_value, chart_name, note) VALUES "
            "(1, 7, 'RodPumpOilWell', '2026-07-14 20:00', 'Load', 'Paraffin'), "
            "(2, 7, 'RodPumpOilWell', 'not a time', 'Load', 'Odd')"
        ))
    client = WellTimelineClient(engine, storage_timezone="UTC")
    # July 15 in Tokyo, as naive site-clock bounds.
    start, end = datetime(2026, 7, 15), datetime(2026, 7, 16)

    history = client._well_history(4, 7, start, end)
    notes = client._chart_notes(4, 7, start, end)

    assert [event["time"] for event in history] == ["2026-07-15 05:00:00"]
    assert [event["time"] for event in notes] == ["2026-07-15 05:00:00"]
