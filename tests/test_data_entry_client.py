from sqlalchemy import create_engine, text

from umai.clients.data_entry_client import DataEntryClient


def make_client():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE flares (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE flow_meters (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE pumps (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE treaters (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE knock_outs (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE water_plants (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE lacts (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE tanks (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT, type TEXT)"))
        connection.execute(text("CREATE TABLE flare_readings (id INTEGER PRIMARY KEY, flare_id INTEGER, time TEXT)"))
        connection.execute(text("CREATE TABLE lact_readings (id INTEGER PRIMARY KEY, lact_id INTEGER, time TEXT)"))
        connection.execute(text("CREATE TABLE linear_tank_readings (id INTEGER PRIMARY KEY, tank_id INTEGER, time TEXT)"))
        connection.execute(text("CREATE TABLE mixed_tank_readings (id INTEGER PRIMARY KEY, tank_id INTEGER, time TEXT)"))
        connection.execute(text("CREATE TABLE non_linear_tank_readings (id INTEGER PRIMARY KEY, tank_id INTEGER, time TEXT)"))
        connection.execute(text("CREATE TABLE general_notes (id INTEGER PRIMARY KEY, site_id INTEGER, date TEXT)"))
        connection.execute(text("CREATE TABLE work_orders (id INTEGER PRIMARY KEY, site_id INTEGER, subject TEXT)"))
        connection.execute(text("INSERT INTO flares VALUES (1, 4, 'Battery 6 Flare'), (2, 4, 'Battery 60 Flare'), (3, 5, 'Battery 6 Flare')"))
        connection.execute(text("INSERT INTO lacts VALUES (8, 4, 'Battery 5 Lact')"))
        connection.execute(text("INSERT INTO tanks VALUES (5, 4, '5-2 Float Over', 'mixed-water-oil')"))
        connection.execute(text("INSERT INTO tanks VALUES (15, 4, '15-2 Float Over', 'mixed-water-oil')"))
    return DataEntryClient(engine)


def test_prepares_flare_reading_with_today_and_values():
    result = make_client().prepare(
        4,
        "flare_reading",
        "Battery 6 Flare",
        {"pressure": 10, "volume": 20},
        "2026-07-03",
    )

    assert result == {
        "status": "ready",
        "entry_type": "flare_reading",
        "entity_type": "flare",
        "entity_id": 1,
        "entity_name": "Battery 6 Flare",
        "values": {"pressure": 10, "volume": 20, "date": "2026-07-03"},
    }


def test_ambiguous_partial_name_requires_clarification():
    result = make_client().prepare(4, "flare_reading", "Battery", {}, "2026-07-03")

    assert result["status"] == "needs_clarification"
    assert [candidate["name"] for candidate in result["candidates"]] == [
        "Battery 6 Flare",
        "Battery 60 Flare",
    ]


def test_existing_reading_blocks_navigation():
    client = make_client()
    with client.engine.begin() as connection:
        connection.execute(text("INSERT INTO flare_readings VALUES (1, 1, '2026-07-03')"))

    result = client.prepare(
        4, "flare_reading", "Battery 6 Flare", {"volume": 20}, "2026-07-03"
    )

    assert result["status"] == "duplicate"


def test_unknown_fields_and_scheduled_tasks_are_rejected():
    client = make_client()

    assert client.prepare(4, "flare_reading", "Battery 6 Flare", {"bogus": 1}, "2026-07-03")["status"] == "invalid"
    assert client.prepare(4, "scheduled_task", None, {}, "2026-07-03")["status"] == "unsupported"


def test_generic_reading_asks_when_name_exists_for_multiple_entity_types():
    client = make_client()
    with client.engine.begin() as connection:
        connection.execute(text("INSERT INTO tanks VALUES (10, 4, 'Battery 6 Flare', 'linear-volume')"))

    result = client.prepare(4, "generic_reading", "Battery 6 Flare", {}, "2026-07-03")

    assert result["status"] == "needs_clarification"
    assert {candidate["type"] for candidate in result["candidates"]} == {"flare", "tank"}


def test_prepares_tank_reading_with_voice_parsed_values():
    result = make_client().prepare(
        4,
        "tank_reading",
        "tank 5–2 float over",
        {
            "top_level_feet": 5.0,
            "top_level_inches": 2.0,
            "water_level_feet": 2.0,
            "water_level_inches": 1.0,
        },
        "2026-07-07",
    )

    assert result == {
        "status": "ready",
        "entry_type": "tank_reading",
        "entity_type": "tank",
        "entity_id": 5,
        "entity_name": "5-2 Float Over",
        "values": {
            "top_level_feet": 5.0,
            "top_level_inches": 2.0,
            "water_level_feet": 2.0,
            "water_level_inches": 1.0,
            "date": "2026-07-07",
        },
    }


def test_tank_identifier_does_not_fuzzy_match_larger_identifier():
    result = make_client().prepare(
        4,
        "tank_reading",
        "tank 5–2 float over",
        {"top_level_feet": 5.0},
        "2026-07-07",
    )

    assert result["status"] == "ready"
    assert result["entity_id"] == 5
    assert result["entity_name"] == "5-2 Float Over"


def test_spoken_tank_prefix_matches_database_name_without_tank_prefix():
    result = make_client().prepare(
        4,
        "tank_reading",
        "tank 5-2 float over",
        {"top_level_feet": 5.0},
        "2026-07-07",
    )

    assert result["status"] == "ready"
    assert result["entity_id"] == 5
    assert result["entity_name"] == "5-2 Float Over"


def test_spoken_number_and_redundant_prefix_resolve_lact_entity():
    result = make_client().prepare(
        4,
        "lact_reading",
        "lact battery five lact",
        {"reading": 22.0, "comments": "hhhh"},
        "2026-07-07",
    )

    assert result == {
        "status": "ready",
        "entry_type": "lact_reading",
        "entity_type": "LACT",
        "entity_id": 8,
        "entity_name": "Battery 5 Lact",
        "values": {
            "reading": 22.0,
            "comments": "hhhh",
            "date": "2026-07-07",
        },
    }
