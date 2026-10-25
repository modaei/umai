import json

from sqlalchemy import create_engine, text

from umai.api.app import _view_intent
from umai.clients.view_navigation_client import ViewNavigationClient


def make_client():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE flares (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT)"))
        connection.execute(text("CREATE TABLE flare_readings (id INTEGER PRIMARY KEY, flare_id INTEGER, time TEXT)"))
        connection.execute(text("INSERT INTO flares VALUES (1, 4, 'test1'), (2, 5, 'test1')"))
    return ViewNavigationClient(engine)


def test_report_always_returns_navigation_and_defaults_to_last_seven_days():
    result = make_client().prepare(4, "oil_production", "2026-07-03")

    assert result == {
        "status": "ready",
        "destination": "report",
        "view_type": "oil_production",
        "filters": {"startDate": "2026-06-27", "endDate": "2026-07-03"},
    }


def test_operational_search_uses_plain_index_search_text_and_dates():
    client = make_client()
    with client.engine.begin() as connection:
        connection.execute(text("INSERT INTO flare_readings VALUES (10, 1, '2026-07-02')"))

    result = client.prepare(
        4, "flare_readings", "2026-07-03",
        start_date="2026-07-02", end_date="2026-07-02",
        search_text="test1",
    )

    assert result["status"] == "ready"
    assert result["destination"] == "detail"
    assert result["record_id"] == 10
    assert result["filters"] == {
        "startDate": "2026-07-02",
        "endDate": "2026-07-02",
        "query": "test1",
    }


def test_operational_zero_and_multiple_results_choose_expected_destination():
    client = make_client()
    assert client.prepare(4, "flare_readings", "2026-07-03", search_text="missing")["status"] == "no_results"

    with client.engine.begin() as connection:
        connection.execute(text("INSERT INTO flare_readings VALUES (10, 1, '2026-07-01'), (11, 1, '2026-07-02')"))
    result = client.prepare(4, "flare_readings", "2026-07-03", search_text="test1")
    assert result["destination"] == "index"


def test_non_show_search_page_filter_is_rejected_when_unsupported():
    result = make_client().prepare(4, "water_draws", "2026-07-03", search_text="test1")
    assert result["status"] == "invalid"


def test_api_forwards_ready_intent_only_for_show_me_prefix():
    calls = [{"tool": "prepare_data_view", "result": json.dumps({
        "status": "ready", "destination": "report",
        "view_type": "oil_sale", "filters": {},
    })}]

    assert _view_intent("  Show me oil sales", calls)["view_type"] == "oil_sale"
    assert _view_intent("What were oil sales?", calls) is None


def test_api_builds_report_navigation_from_existing_report_tool():
    calls = [{"tool": "run_report", "arguments": {
        "report_name": "oil_production",
        "start_date": "2026-06-01",
        "end_date": "2026-07-03",
    }, "result": "{}"}]

    result = _view_intent("show me oil production", calls)
    assert result["destination"] == "report"
    assert result["filters"]["startDate"] == "2026-06-01"
