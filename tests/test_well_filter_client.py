import pytest
from sqlalchemy import create_engine, text

from umai.clients.well_filter_client import WellFilterClient, WellFilterClientError


def make_client() -> WellFilterClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE wells (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    pump_type TEXT,
                    wogcc_class TEXT,
                    wogcc_status TEXT,
                    direction TEXT,
                    prod_fm TEXT,
                    monitored INTEGER,
                    disable_reading INTEGER,
                    multiple_injection_form INTEGER,
                    onrr_code_id INTEGER,
                    battery_id INTEGER,
                    lact_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE onrr_codes (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE lacts (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO onrr_codes (id, name)
                VALUES (1, 'POW'), (2, 'TA'), (3, 'WIW')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO batteries (id, name)
                VALUES (1, 'Battery 1'), (6, 'Battery 6')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO lacts (id, name)
                VALUES (1, 'LACT 1'), (2, 'LACT 2')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO wells (
                    id, site_id, name, key, pump_type, wogcc_class, wogcc_status,
                    direction, prod_fm, monitored, disable_reading,
                    multiple_injection_form, onrr_code_id, battery_id, lact_id
                )
                VALUES
                    (1, 4, 'HARTZOG DRAW UNIT 1001', 'HDU1001', 'ROD', 'oil', 'active', 'vertical', 'FM1', 1, 0, 0, 1, 1, 1),
                    (2, 4, 'HARTZOG DRAW UNIT 1002', 'HDU1002', 'rod', 'oil', 'active', 'horizontal', 'FM2', 1, 0, 0, 2, 6, 2),
                    (3, 4, 'HARTZOG DRAW UNIT 2001', 'HDU2001', 'ESP', 'oil', 'active', 'vertical', 'FM1', 1, 0, 0, 2, 6, 2),
                    (4, 4, 'HARTZOG DRAW UNIT 3001', 'HDU3001', 'flowing well no lift', 'gas', 'active', 'vertical', 'FM3', 0, 0, 0, 3, 1, 1),
                    (5, 5, 'OTHER SITE UNIT 1001', 'OS1001', 'ROD', 'oil', 'active', 'vertical', 'FM1', 1, 0, 0, 1, 1, 1),
                    (6, 4, 'HARTZOG DRAW UNIT 4001', 'HDU4001', NULL, 'oil', 'inactive', 'vertical', 'FM4', 0, 1, 0, 2, 1, 1)
                """
            )
        )
    return WellFilterClient(engine)


def test_find_wells_matches_pump_type_case_insensitively():
    result = make_client().find_wells(
        4, filters=[{"field": "pump_type", "value": "rod"}]
    )

    assert result["count"] == 2
    assert [well["name"] for well in result["wells"]] == [
        "HARTZOG DRAW UNIT 1001",
        "HARTZOG DRAW UNIT 1002",
    ]


def test_find_wells_normalizes_flowing_well_alias():
    result = make_client().find_wells(
        4, filters=[{"field": "pump_type", "value": "flowing wells"}]
    )

    assert result["count"] == 1
    assert result["wells"][0]["pump_type"] == "flowing well no lift"


def test_find_wells_can_filter_missing_pump_type_when_explicit():
    result = make_client().find_wells(
        4, filters=[{"field": "pump_type", "value": "no pump type"}]
    )

    assert result["count"] == 1
    assert result["wells"][0]["name"] == "HARTZOG DRAW UNIT 4001"


def test_find_wells_filters_by_site_and_well_name_or_key():
    result = make_client().find_wells(
        4,
        well_name="HDU1002",
        filters=[{"field": "pump_type", "value": "ROD"}],
    )

    assert result["count"] == 1
    assert result["wells"][0]["name"] == "HARTZOG DRAW UNIT 1002"


def test_find_wells_filters_by_onrr_code_case_insensitively():
    result = make_client().find_wells(
        4, filters=[{"field": "onrr_code", "value": "ta"}]
    )

    assert result["count"] == 3
    assert [well["name"] for well in result["wells"]] == [
        "HARTZOG DRAW UNIT 1002",
        "HARTZOG DRAW UNIT 2001",
        "HARTZOG DRAW UNIT 4001",
    ]


def test_find_wells_filters_by_joined_battery_and_lact():
    result = make_client().find_wells(
        4,
        filters=[
            {"field": "battery", "value": "Battery 6"},
            {"field": "lact", "value": "LACT 2"},
        ],
    )

    assert result["count"] == 2
    assert [well["name"] for well in result["wells"]] == [
        "HARTZOG DRAW UNIT 1002",
        "HARTZOG DRAW UNIT 2001",
    ]


def test_find_wells_filters_by_boolean_attribute():
    result = make_client().find_wells(
        4, filters=[{"field": "disable_reading", "value": True}]
    )

    assert result["count"] == 1
    assert result["wells"][0]["name"] == "HARTZOG DRAW UNIT 4001"


def test_find_wells_rejects_unsupported_filter_field():
    with pytest.raises(WellFilterClientError, match="Unsupported well filter field"):
        make_client().find_wells(
            4, filters=[{"field": "unsafe_sql", "value": "anything"}]
        )
