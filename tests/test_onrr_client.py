from sqlalchemy import create_engine, text

from umai.clients.onrr_client import OnrrClient, OnrrClientError


def make_onrr_client() -> OnrrClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE wells (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    onrr_code_id INTEGER NOT NULL
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
                    injection_well INTEGER NOT NULL,
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
                INSERT INTO onrr_codes (id, name, active_well, injection_well, description)
                VALUES
                    (1, 'POW', 1, 0, 'Producing oil well'),
                    (2, 'SIW', 0, 0, 'Shut-in well'),
                    (3, 'WIW', 1, 1, 'Water injection well')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO wells (id, site_id, name, onrr_code_id)
                VALUES
                    (1, 4, 'HARTZOG DRAW UNIT 4048', 1),
                    (2, 4, 'HARTZOG DRAW UNIT 5050', 3),
                    (3, 4, 'HARTZOG DRAW UNIT 6060', 2),
                    (4, 5, 'OTHER SITE 4048', 1)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_histories
                    (id, well_id, property, old_value, new_value, changed_at)
                VALUES
                    (100, 1, 'onrr_code_id', '2', '1', '2026-06-15 00:00:00')
                """
            )
        )
    return OnrrClient(engine)


def test_explain_onrr_code_returns_flags_and_description():
    result = make_onrr_client().explain_onrr_code("wiw")

    assert result["found"] is True
    assert result["onrr_code"] == {
        "code": "WIW",
        "active_well": True,
        "injection_well": True,
        "description": "Water injection well",
    }


def test_get_well_onrr_status_uses_history_as_of_date():
    result = make_onrr_client().get_well_onrr_status_as_of_date(
        4, "4048", "2026-06-01"
    )

    assert result["found"] is True
    assert result["status"]["onrr_code"] == "SIW"
    assert result["status"]["onrr_active_well"] is False
    assert result["status"]["onrr_code_description"] == "Shut-in well"


def test_count_wells_by_onrr_status_groups_producing_and_injection():
    result = make_onrr_client().count_wells_by_onrr_status(
        4, "2026-06-20", "all"
    )

    assert result["counts"] == {
        "active": 2,
        "producing": 1,
        "injection": 1,
        "inactive": 1,
    }


def test_count_wells_by_onrr_status_validates_status():
    try:
        make_onrr_client().count_wells_by_onrr_status(4, "2026-06-20", "bad")
    except OnrrClientError as exc:
        assert "status must be" in str(exc)
    else:
        raise AssertionError("Expected OnrrClientError")
