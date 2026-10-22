from sqlalchemy import create_engine, text

from umai.clients.well_timeline_client import (
    UnavailableWellTimelineClient,
    WellTimelineClient,
    WellTimelineClientError,
)


class FakeOperationalContextStore:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "query": kwargs["query"],
            "count": 2,
            "matches": [
                {
                    "source_type": "chart_note",
                    "source_id": "1",
                    "event_date": "2026-05-09",
                    "entity_name": "11-1-1 Oil",
                    "text": "Chart note duplicate.",
                },
                {
                    "source_type": "mixed_tank_reading_comment",
                    "source_id": "6690",
                    "event_date": "2026-05-10",
                    "entity_name": "10-1 Oil",
                    "text": "Mixed tank reading comment. Comments: Hot oil 11-1-1 Oil FL",
                },
            ],
        }


def make_timeline_client(
    operational_context_store=None,
) -> WellTimelineClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE wells (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE well_tests (
                    id INTEGER PRIMARY KEY,
                    well_id INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    oil REAL,
                    water REAL,
                    gas REAL,
                    pip REAL,
                    m_temp REAL,
                    amps REAL,
                    tbgp REAL,
                    csgp REAL,
                    runtime REAL,
                    comments TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE well_fluids (
                    id INTEGER PRIMARY KEY,
                    well_id INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    level REAL,
                    comments TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE well_shutdowns (
                    id INTEGER PRIMARY KEY,
                    well_id INTEGER NOT NULL,
                    start TEXT NOT NULL,
                    `end` TEXT,
                    downtime_code TEXT,
                    comments TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE chart_notes (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id INTEGER NOT NULL,
                    chart_name TEXT NOT NULL,
                    x_axis_value TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE general_notes (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    comments TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE work_orders (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    subject TEXT,
                    status TEXT,
                    priority INTEGER,
                    vendor TEXT,
                    comments TEXT
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO wells (id, site_id, name) VALUES (1, 1, '11-1-1 Oil')")
        )
        connection.execute(
            text("INSERT INTO wells (id, site_id, name) VALUES (2, 1, 'HARTZOG DRAW UNIT 4048')")
        )
        connection.execute(
            text(
                """
                INSERT INTO well_tests
                    (well_id, time, oil, water, gas, runtime, comments)
                VALUES
                    (1, '2026-05-09 08:00:00', 10, 2, 50, 24, 'good test')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_fluids (well_id, time, level, comments)
                VALUES (1, '2026-05-09 09:00:00', 1234, 'fluid shot')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO chart_notes
                    (id, site_id, object_type, object_id, chart_name, x_axis_value, note)
                VALUES
                    (1, 1, 'RodPumpOilWell', 1, 'Load', '2026-05-09 10:00:00', 'Hot watering 1 load'),
                    (2, 1, 'Tank', 1, 'Level', '2026-05-09 11:00:00', 'Should not appear on well timeline')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_shutdowns
                    (well_id, start, `end`, downtime_code, comments)
                VALUES (1, '2026-05-10 00:00:00', '2026-05-10 04:30:00', 'PRF', 'paraffin')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_shutdowns
                    (well_id, start, `end`, downtime_code, comments)
                VALUES (2, '2026-06-01 00:00:00', '2026-06-01 08:00:00', 'DH', '4048 shutdown')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO general_notes (site_id, date, comments)
                VALUES (1, '2026-05-10', 'Checked 11-1-1 Oil after shutdown')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO work_orders
                    (site_id, time, subject, status, priority, vendor, comments)
                VALUES
                    (1, '2026-05-10 12:00:00', 'Repair 11-1-1 Oil', 'open', 3, 'Vendor A', 'replace part')
                """
            )
        )
    return WellTimelineClient(engine, operational_context_store=operational_context_store)


def test_get_well_timeline_returns_chronological_events():
    result = make_timeline_client().get_well_timeline(
        1, "11-1-1", "2026-05-09", "2026-05-10"
    )

    assert result["well"] == "Well - 11-1-1 Oil"
    assert [event["source"] for event in result["events"]] == [
        "well_test",
        "well_fluid",
        "chart_note",
        "general_note",
        "shutdown",
        "work_order",
    ]
    assert result["events"][0]["details"]["oil"] == 10
    assert result["events"][2]["details"]["note"] == "Hot watering 1 load"
    assert result["events"][4]["details"]["downtime_reason"] == "Paraffin"
    assert result["events"][3]["time"] == "2026-05-10"
    assert result["events"][5]["details"]["subject"] == "Repair 11-1-1 Oil"


def test_get_well_timeline_enriches_with_rag_context_and_skips_duplicate_sources():
    store = FakeOperationalContextStore()
    result = make_timeline_client(store).get_well_timeline(
        1,
        "11-1-1",
        "2026-05-09",
        "2026-05-10",
        context_query="chemical treatment hot water paraffin",
    )

    operational_context_events = [
        event for event in result["events"] if event["source"] == "operational_context"
    ]

    assert len(operational_context_events) == 1
    assert operational_context_events[0]["details"] == {
        "source_type": "mixed_tank_reading_comment",
        "source_id": "6690",
        "entity_name": "10-1 Oil",
        "text": "Mixed tank reading comment. Comments: Hot oil 11-1-1 Oil FL",
    }
    assert store.calls[0]["query"] == "chemical treatment hot water paraffin"
    assert store.calls[0]["entity_name"] == "11-1-1 Oil"


def test_numeric_well_name_resolves_hartzog_draw_unit_prefix():
    result = make_timeline_client().get_well_timeline(
        1, "4048", "2026-01-01", "2026-06-15"
    )

    assert result["well"] == "Well - HARTZOG DRAW UNIT 4048"
    assert result["events"] == [
        {
                "time": "2026-06-01 00:00:00",
            "source": "shutdown",
            "label": "Well shutdown",
            "details": {
                "downtime_code": "DH",
                "downtime_reason": "Downhole Problems",
                "comments": "4048 shutdown",
                "start": "2026-06-01 00:00:00",
                "end": "2026-06-01 08:00:00",
            },
        }
    ]


def test_well_timeline_rejects_unknown_well():
    try:
        make_timeline_client().get_well_timeline(
            1, "missing", "2026-05-09", "2026-05-10"
        )
    except WellTimelineClientError as exc:
        assert "Well not found" in str(exc)
    else:
        raise AssertionError("Expected WellTimelineClientError")


def test_unavailable_well_timeline_client_rejects_query():
    client = UnavailableWellTimelineClient("DB settings missing")
    try:
        client.get_well_timeline(1, "11-1-1", "2026-05-09", "2026-05-10")
    except WellTimelineClientError as exc:
        assert "DB settings missing" in str(exc)
    else:
        raise AssertionError("Expected WellTimelineClientError")
