import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, text

from umai.clients.rod_pump_health_client import RodPumpHealthClient, RodPumpHealthError


DENVER = ZoneInfo("America/Denver")


def review(health, **extra):
    return {
        "review_status": "complete",
        "review_source": "classifier",
        "diagnosis_status": "healthy" if health == "healthy" else "candidates",
        "health_status": health,
        "summary": f"{health} summary",
        "findings": [] if health == "healthy" else [{
            "side": "downhole",
            "kind": "shape",
            "title": "Incomplete fillage with fluid pound",
            "reasoning": "Sharp unloading on the downstroke.",
            "actions": "Review the pump-off setpoint.",
            "confidence": "high",
            "card_ids": ["live:1"],
        }],
        "limitations": [],
        "surface_load": {"status": "normal"},
        "parameter_trends": {
            "status": "complete",
            "parameters": {"Pump Fillage": [{"timestamp": 1, "value": 80}]},
            "trends": [{"parameter": "Pump Fillage", "title": "Variable fillage"}],
            "interpretation": "Pump-off cycling.",
            "token_usage": {"total_tokens": 10},
        },
        "conclusion": {"status": "complete", "title": "Pump-off cycling", "body": "Likely cycling.", "token_usage": {}},
        **extra,
    }


@pytest.fixture
def client():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE wells(id INTEGER, site_id INTEGER, name TEXT, `key` TEXT, pump_type TEXT)"
        ))
        connection.execute(text(
            "CREATE TABLE report_configurations(id INTEGER, site_id INTEGER, function_name TEXT, related_entities TEXT)"
        ))
        connection.execute(text(
            "CREATE TABLE rod_pump_health_analyses(well_id INTEGER, status TEXT, latest_card_at TEXT, "
            "analyzed_at TEXT, refresh_started_at TEXT, last_refresh_failed_at TEXT, "
            "last_refresh_error TEXT, findings TEXT)"
        ))
        connection.execute(text("INSERT INTO wells VALUES(:id,:site,:name,:key,:pump)"), [
            {"id": 295, "site": 4, "name": "HARTZOG DRAW UNIT 3051", "key": "HDU_3051", "pump": "ROD"},
            {"id": 296, "site": 4, "name": "HARTZOG DRAW UNIT 5823", "key": "HDU_5823", "pump": "rod"},
            {"id": 297, "site": 4, "name": "HARTZOG DRAW UNIT 1200", "key": "HDU_1200", "pump": "ROD"},
            {"id": 298, "site": 4, "name": "HARTZOG DRAW UNIT 7000", "key": "HDU_7000", "pump": "ROD"},
            {"id": 299, "site": 4, "name": "ESP 3052", "key": "ESP_3052", "pump": "ESP"},
            {"id": 400, "site": 5, "name": "OTHER 3051", "key": "OT_3051", "pump": "ROD"},
        ])
        connection.execute(text("INSERT INTO report_configurations VALUES(1,4,'rod_pump_monitoring',:entities)"), {
            "entities": json.dumps({"well_ids": [295, 296, 297, 400]}),
        })
        connection.execute(text(
            "INSERT INTO rod_pump_health_analyses VALUES(:well,:status,:card,:analyzed,:started,:failed,:error,:findings)"
        ), [
            {"well": 295, "status": "complete", "card": "2026-10-29 19:18:08", "analyzed": "2026-10-30 01:32:03",
             "started": None, "failed": None, "error": None, "findings": json.dumps(review("watch"))},
            {"well": 296, "status": "complete", "card": "2026-10-29 18:00:00", "analyzed": "2026-10-30 01:00:00",
             "started": None, "failed": "2026-10-30 02:00:00", "error": "Analysis failed.",
             "findings": json.dumps(review("healthy"))},
            {"well": 400, "status": "complete", "card": None, "analyzed": None,
             "started": None, "failed": None, "error": None, "findings": json.dumps(review("problem"))},
        ])
    return RodPumpHealthClient(engine, timezone="America/Denver")


def test_well_health_returns_saved_review_without_raw_trends(client):
    result = client.well_health(4, "3051")

    assert result["well"] == {"id": 295, "name": "HARTZOG DRAW UNIT 3051", "key": "HDU_3051"}
    assert result["available"] is True
    assert result["health_status"] == "watch"
    assert result["review_source"] == "classifier"
    assert result["findings"][0]["title"] == "Incomplete fillage with fluid pound"
    assert "card_ids" not in result["findings"][0]
    assert result["analyzed_at"] == "2026-10-30T01:32:03-06:00"
    assert result["card_captured_at"] == "2026-10-29T19:18:08-06:00"
    assert result["parameter_trends"]["interpretation"] == "Pump-off cycling."
    assert "parameters" not in result["parameter_trends"]
    assert "token_usage" not in result["parameter_trends"]
    assert result["conclusion"] == {"status": "complete", "title": "Pump-off cycling", "body": "Likely cycling."}
    assert result["refresh"] == {"in_progress": False}


def test_well_health_reports_last_refresh_failure(client):
    result = client.well_health(4, "HDU_5823")

    assert result["refresh"]["last_failed_at"] == "2026-10-30T02:00:00-06:00"
    assert result["refresh"]["last_error"] == "Analysis failed."


def test_well_health_reports_refresh_in_progress(client):
    started = datetime.now(DENVER).replace(tzinfo=None) - timedelta(minutes=2)
    with client.engine.begin() as connection:
        connection.execute(
            text("UPDATE rod_pump_health_analyses SET refresh_started_at=:started WHERE well_id=295"),
            {"started": started.isoformat(sep=" ", timespec="seconds")},
        )

    assert client.well_health(4, "3051")["refresh"]["in_progress"] is True


def test_well_health_without_saved_review(client):
    result = client.well_health(4, "1200")

    assert result["available"] is False
    assert "No saved health analysis" in result["reason"]


@pytest.mark.parametrize("name, message", [
    ("051", "not found"),
    ("3052", "not found"),
    ("7000", "not configured for rod-pump health monitoring"),
    ("", "required"),
])
def test_resolve_well_rejects_partial_other_and_unmonitored_wells(client, name, message):
    with pytest.raises(RodPumpHealthError, match=message):
        client.resolve_well(4, name)


def test_site_health_orders_by_health_and_stays_in_site(client):
    result = client.site_health(4)

    assert [row["well"]["id"] for row in result["wells"]] == [295, 296, 297]
    assert result["counts"] == {"watch": 1, "healthy": 1, "not_analyzed": 1}
    assert result["wells"][0]["finding_titles"] == ["Incomplete fillage with fluid pound"]
    assert result["wells"][2] == {"well": {"id": 297, "name": "HARTZOG DRAW UNIT 1200", "key": "HDU_1200"}, "available": False}
