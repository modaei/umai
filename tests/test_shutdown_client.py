from datetime import date, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from umai.clients.shutdown_client import (
    DOWNTIME_CODES,
    ShutdownClient,
    ShutdownClientError,
    UnavailableShutdownClient,
)


def make_shutdown_client() -> ShutdownClient:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE wells (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    onrr_code_id INTEGER NOT NULL,
                    pump_type TEXT,
                    battery_id INTEGER,
                    lact_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
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
                CREATE TABLE lacts (
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
                CREATE TABLE onrr_codes (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    active_well INTEGER NOT NULL,
                    injection_well INTEGER NOT NULL DEFAULT 0,
                    description TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE well_histories (
                    id INTEGER PRIMARY KEY,
                    well_id INTEGER NOT NULL,
                    property TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    changed_at TEXT NOT NULL
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
                INSERT INTO onrr_codes (id, name, active_well, injection_well, description)
                VALUES
                    (1, 'POW', 1, 0, 'Producing oil well'),
                    (2, 'SIW', 0, 0, 'Shut-in well'),
                    (3, 'INJ', 1, 1, 'Active injection well')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO batteries (id, site_id, name)
                VALUES
                    (6, 1, 'Battery 6'),
                    (16, 1, 'Battery 16')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO lacts (id, site_id, name)
                VALUES
                    (6, 1, 'Battery 6 LACT')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO wells (id, site_id, name, onrr_code_id, pump_type, battery_id, lact_id)
                VALUES
                    (1, 1, '11-1-1 Oil', 1, 'ROD', 6, 6),
                    (2, 1, '12-2-1 Oil', 1, 'ESP', 6, 6),
                    (3, 2, 'Other Site Well', 1, 'ROD', 6, NULL),
                    (4, 1, '13-3-1 Oil', 2, 'ROD', 6, NULL),
                    (5, 1, '14-4-1 Oil', 1, 'ROD', 16, NULL),
                    (6, 1, '15-5-1 Oil', 2, NULL, 16, NULL),
                    (7, 1, '16-6-1 Oil', 1, 'ROD', 6, NULL),
                    (8, 1, '17-7-1 Injection', 3, 'ROD', 6, NULL)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_shutdowns
                    (id, well_id, start, `end`, downtime_code, comments)
                VALUES
                    (10, 1, '2026-05-09 00:00:00', '2026-05-09 04:30:00', 'PRF', 'paraffin cleanout'),
                    (11, 2, '2026-05-01 08:00:00', NULL, 'DH', 'downhole issue'),
                    (12, 3, '2026-05-09 00:00:00', '2026-05-10 00:00:00', 'PRF', 'other site'),
                    (13, 5, '2026-06-21 00:00:00', '2026-06-22 00:00:00', 'SIB', 'full day shutdown'),
                    (14, 7, '2026-06-21 12:00:00', '2026-06-21 18:00:00', 'DH', 'partial shutdown')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_histories
                    (id, well_id, property, old_value, new_value, changed_at)
                VALUES
                    (100, 6, 'onrr_code_id', '1', '2', '2026-06-22 00:00:00')
                """
            )
        )
    return ShutdownClient(engine)


def test_get_shutdowns_returns_shutdown_intervals_for_site():
    result = make_shutdown_client().get_shutdowns(
        1, "2026-05-01", "2026-05-10"
    )

    assert result["count"] == 2
    assert result["total_hours"] == 236.5
    assert result["shutdowns"] == [
        {
            "well": "Well - 12-2-1 Oil",
            "start": "2026-05-01 08:00:00",
            "end": None,
            "downtime_code": "DH",
            "downtime_reason": "Downhole Problems",
            "comments": "downhole issue",
        },
        {
            "well": "Well - 11-1-1 Oil",
            "start": "2026-05-09 00:00:00",
            "end": "2026-05-09 04:30:00",
            "downtime_code": "PRF",
            "downtime_reason": "Paraffin",
            "comments": "paraffin cleanout",
        }
    ]


def test_get_current_shutdowns():
    result = make_shutdown_client().get_current_shutdowns(1, "2026-05-10")

    assert result["count"] == 1
    assert result["shutdowns"][0]["well"] == "Well - 12-2-1 Oil"


def test_summarize_shutdown_causes_ranks_by_total_hours():
    result = make_shutdown_client().summarize_shutdown_causes(
        1, "2026-05-01", "2026-05-10"
    )

    assert result["cause_count"] == 2
    assert result["main_cause"]["downtime_code"] == "DH"
    assert result["main_cause"]["downtime_reason"] == "Downhole Problems"
    assert result["main_cause"]["event_count"] == 1
    assert result["main_cause"]["total_hours"] == 232.0
    assert result["causes"][1]["downtime_code"] == "PRF"
    assert result["causes"][1]["total_hours"] == 4.5


def test_list_downtime_codes():
    result = make_shutdown_client().list_downtime_codes()

    assert result["downtime_codes"]["PRF"] == DOWNTIME_CODES["PRF"]


def test_get_active_wells_uses_only_onrr_state():
    result = make_shutdown_client().get_active_wells(1, "2026-06-21")

    assert result["active_count"] == 6
    assert [row["well"] for row in result["active_wells"]] == [
        "Well - 11-1-1 Oil",
        "Well - 12-2-1 Oil",
        "Well - 14-4-1 Oil",
        "Well - 15-5-1 Oil",
        "Well - 16-6-1 Oil",
        "Well - 17-7-1 Injection",
    ]
    assert result["inactive_count"] == 1
    assert [row["well"] for row in result["inactive_wells"]] == [
        "Well - 13-3-1 Oil",
    ]
    assert "partial_shutdown_count" not in result
    assert result["rules"]["active_well_rule"] == (
        "Active wells must have active_well=true on the ONRR code as of that day."
    )


def test_get_active_wells_uses_current_onrr_when_no_history_exists():
    result = make_shutdown_client().get_active_wells(1, "2026-06-23")

    assert "Well - 15-5-1 Oil" in [row["well"] for row in result["inactive_wells"]]


def test_get_producing_wells_excludes_onrr_injection_wells():
    result = make_shutdown_client().get_producing_wells(1, "2026-06-23")

    assert result["producing_count"] == 3
    assert [row["well"] for row in result["producing_wells"]] == [
        "Well - 11-1-1 Oil",
        "Well - 14-4-1 Oil",
        "Well - 16-6-1 Oil",
    ]
    injection_well = next(
        row
        for row in result["non_producing_wells"]
        if row["well"] == "Well - 17-7-1 Injection"
    )
    assert injection_well["status"] == "onrr_injection_well"
    assert injection_well["onrr_code"] == "INJ"
    assert injection_well["onrr_code_description"] == "Active injection well"
    assert injection_well["onrr_injection_well"] is True
    full_day_shutdown_well = next(
        row
        for row in result["non_producing_wells"]
        if row["well"] == "Well - 12-2-1 Oil"
    )
    assert full_day_shutdown_well["status"] == "full_day_shutdown"


def test_get_producing_wells_filters_by_battery_before_classification():
    result = make_shutdown_client().get_producing_wells(
        1,
        "2026-06-23",
        filters=[{"field": "battery", "value": "Battery 6"}],
    )

    assert result["filters"] == [{"field": "battery", "value": "Battery 6"}]
    assert result["producing_count"] == 2
    assert [row["well"] for row in result["producing_wells"]] == [
        "Well - 11-1-1 Oil",
        "Well - 16-6-1 Oil",
    ]
    assert all("14-4-1" not in row["well"] for row in result["producing_wells"])


def test_get_producing_wells_numeric_battery_filter_does_not_match_battery_16():
    result = make_shutdown_client().get_producing_wells(
        1,
        "2026-06-23",
        filters=[{"field": "battery", "value": "6"}],
    )

    assert [row["well"] for row in result["producing_wells"]] == [
        "Well - 11-1-1 Oil",
        "Well - 16-6-1 Oil",
    ]


def test_get_producing_wells_filters_by_pump_type_case_insensitively():
    result = make_shutdown_client().get_producing_wells(
        1,
        "2026-06-23",
        filters=[{"field": "pump_type", "value": "rod"}],
    )

    assert result["filters"] == [{"field": "pump_type", "value": "ROD"}]
    assert [row["well"] for row in result["producing_wells"]] == [
        "Well - 11-1-1 Oil",
        "Well - 14-4-1 Oil",
        "Well - 16-6-1 Oil",
    ]


def test_get_producing_wells_range_preserves_filters():
    result = make_shutdown_client().get_producing_wells_for_range(
        1,
        "2026-06-22",
        "2026-06-23",
        filters=[{"field": "pump_type", "value": "rod"}],
    )

    assert result["filters"] == [{"field": "pump_type", "value": "ROD"}]
    assert "Well - 11-1-1 Oil" in [row["well"] for row in result["producing_wells"]]


def test_serialize_row_handles_date_and_datetime_values():
    result = ShutdownClient._serialize_row(
        {
            "shutdown_date": date(2026, 6, 1),
            "created_at": datetime(2026, 6, 1, 7, 30),
        }
    )

    assert result == {
        "shutdown_date": "2026-06-01",
        "created_at": "2026-06-01 07:30:00",
    }


def test_shutdown_date_order_is_validated():
    try:
        make_shutdown_client().get_shutdowns(1, "2026-05-10", "2026-05-01")
    except ShutdownClientError as exc:
        assert "start_date cannot be after end_date" in str(exc)
    else:
        raise AssertionError("Expected ShutdownClientError")


def test_unavailable_shutdown_client_lists_codes_but_rejects_queries():
    client = UnavailableShutdownClient("DB settings missing")

    assert client.list_downtime_codes()["downtime_codes"]["PRF"] == "Paraffin"
    try:
        client.get_current_shutdowns(1, "2026-05-10")
    except ShutdownClientError as exc:
        assert "DB settings missing" in str(exc)
    else:
        raise AssertionError("Expected ShutdownClientError")
