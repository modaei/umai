from sqlalchemy import create_engine, text

from umai.clients.site_client import SiteClient


def test_get_site_name_returns_name_for_site_id():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE sites (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO sites (id, name) VALUES (4, 'HARTZOG DRAW')")
        )

    assert SiteClient(engine).get_site_name(4) == "HARTZOG DRAW"


def test_get_site_name_returns_none_for_unknown_site():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE sites (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
            )
        )

    assert SiteClient(engine).get_site_name(99) is None
