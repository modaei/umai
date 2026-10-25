import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

import pytest

from umai.clients.rod_pump_analysis_client import (
    RodPumpAnalysisClient,
    RodPumpAnalysisError,
)


class Response:
    def __init__(self, points):
        self.points = points

    def raise_for_status(self):
        return None

    def json(self):
        return [{"datapoints": self.points}]


def make_client():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        for statement in (
            "CREATE TABLE sites (id INTEGER PRIMARY KEY, `key` TEXT)",
            "CREATE TABLE wells (id INTEGER PRIMARY KEY, site_id INTEGER, name TEXT, `key` TEXT, pump_type TEXT)",
            "CREATE TABLE rod_pump_monitoring_data (id INTEGER PRIMARY KEY, well_id INTEGER, pull_timestamp INTEGER)",
            "CREATE TABLE rod_pump_monitoring_data_cards (rod_pump_monitoring_data_id INTEGER, rod_pump_card_id INTEGER)",
            "CREATE TABLE rod_pump_cards (id INTEGER PRIMARY KEY, card_type TEXT, down_hole INTEGER, data_points TEXT)",
            "CREATE TABLE chart_notes (object_type TEXT, object_id INTEGER, x_axis_value TEXT, chart_name TEXT, note TEXT)",
            "CREATE TABLE data_points (id INTEGER PRIMARY KEY, site_id INTEGER, facility_name TEXT, data_point_name TEXT)",
            "CREATE TABLE data_point_data (data_point_id INTEGER, data TEXT)",
        ):
            connection.execute(text(statement))
        connection.execute(text("INSERT INTO sites VALUES (4, 'HARTZOG')"))
        connection.execute(text("INSERT INTO wells VALUES (10,4,'HARTZOG DRAW UNIT 5823','HDU_5823','ROD')"))
        first = int(datetime(2026, 7, 5, 12, tzinfo=timezone.utc).timestamp())
        second = int(datetime(2026, 7, 5, 13, tzinfo=timezone.utc).timestamp())
        connection.execute(text("INSERT INTO rod_pump_monitoring_data VALUES (1,10,:ts)"), {"ts": first})
        connection.execute(text("INSERT INTO rod_pump_monitoring_data VALUES (2,10,:ts)"), {"ts": second})
        cards = [
            (1, "current", 0, [{"position": 0, "load": 100}, {"position": 100, "load": 300}]),
            (2, "startup", 0, [{"position": 0, "load": 200}, {"position": 100, "load": 500}]),
            (3, "current", 1, [{"position": 0, "load": 20}, {"position": 100, "load": 80}]),
            (4, "current", 0, [{"position": 0, "load": 250}, {"position": 100, "load": 600}]),
            (5, "current", 1, [{"position": 0, "load": 30}, {"position": 100, "load": 90}]),
        ]
        for card_id, card_type, down_hole, points in cards:
            connection.execute(
                text("INSERT INTO rod_pump_cards VALUES (:id,:type,:down,:points)"),
                {"id": card_id, "type": card_type, "down": down_hole, "points": json.dumps(points)},
            )
        for datum_id, card_id in ((1,1),(1,2),(1,3),(2,4),(2,5)):
            connection.execute(text("INSERT INTO rod_pump_monitoring_data_cards VALUES (:d,:c)"), {"d": datum_id, "c": card_id})
        connection.execute(text("INSERT INTO chart_notes VALUES ('RodPumpOilWell',10,'2026-06-20 10:00:00','Load','Paraffin buildup')"))
        connection.execute(text("INSERT INTO chart_notes VALUES ('RodPumpOilWell',10,'2026-06-21 10:00:00','Load','Hot Watering 1 Load')"))

    values = {
        "min_load_last_stroke": [100, 120, 140],
        "peak_load_last_stroke": [300, 350, 400],
        "pump_fillage": [80, 82, 81],
        "yesterday_cycles": [1000, 1001, 1002],
        "yesterday_stroke_min": [7, 7, 7],
        "max_load": [500, 500, 500],
        "min_load": [100, 100, 100],
    }

    def get(url, params, timeout):
        targets = [value for key, value in params if key == "target"]
        response = Response([])
        response.json = lambda: [
            {
                "target": target,
                "datapoints": [
                    [value, 1751716800 + index * 3600]
                    for index, value in enumerate(values[target.rsplit(".", 1)[-1]])
                ],
            }
            for target in targets
        ]
        return response

    return RodPumpAnalysisClient(engine, "http://graphite/render", http_get=get)


def test_averages_surface_and_downhole_separately_for_every_pull():
    result = make_client().analyze(
        4,
        "5823",
        "2026-07-05T00:00:00+00:00",
        "2026-07-06T00:00:00+00:00",
    )

    cards = result["averaged_dynographs"]
    assert [(card["category"], card["contributing_card_count"]) for card in cards] == [
        ("surface", 2),
        ("downhole", 1),
        ("surface", 1),
        ("downhole", 1),
    ]
    assert cards[0]["points"] == [
        {"position": 0.0, "load": 150.0},
        {"position": 1.0, "load": 400.0},
    ]
    assert cards[1]["points"][1]["load"] == 80


def test_chart_notes_are_categorized_and_support_paraffin_assessment():
    result = make_client().analyze(
        4, "5823", "2026-07-05T00:00:00+00:00", "2026-07-06T00:00:00+00:00"
    )

    assert [note["event"] for note in result["chart_note_events"]] == [
        "paraffin",
        "thermal_treatment",
    ]
    paraffin = next(item for item in result["diagnoses"] if item["name"] == "Possible paraffin/wax buildup")
    assert paraffin["confidence"] == 0.7
    assert "Scale" in paraffin["alternative_causes"]


def test_prediction_is_unavailable_without_validated_artifact():
    result = make_client().analyze(
        4, "5823", "2026-07-05T00:00:00+00:00", "2026-07-06T00:00:00+00:00"
    )
    assert result["paraffin_prediction"]["available"] is False


def test_analysis_rejects_an_incomplete_identifier():
    with pytest.raises(RodPumpAnalysisError, match="not found"):
        make_client().analyze(
            4, "823", "2026-07-05T00:00:00+00:00", "2026-07-06T00:00:00+00:00"
        )


def test_batch_ranking_respects_configured_wells_and_reports_ignored_ids():
    client = make_client()

    def analyze(site_id, well_name, start_time, end_time):
        return {
            "well": {"id": 10, "name": well_name, "key": "HDU_5823"},
            "health": {"score": 65, "category": "watchlist"},
            "diagnoses": [{
                "name": "Rod/tubing friction",
                "severity": "medium",
                "confidence": .66,
            }],
            "warnings": [],
        }

    client.analyze = analyze
    result = client.rank_wells(
        4,
        "2026-07-06T00:00:00+00:00",
        well_ids=[10, 99],
    )

    assert result["well_scope"] == "configured"
    assert result["ignored_well_ids"] == [99]
    assert [row["well"]["name"] for row in result["wells"]] == ["HARTZOG DRAW UNIT 5823"]


def test_batch_ranking_preserves_an_explicit_empty_scope():
    result = make_client().rank_wells(
        4,
        "2026-07-06T00:00:00+00:00",
        well_ids=[],
    )
    assert result["well_scope"] == "configured"
    assert result["wells"] == []


@pytest.mark.parametrize(
    "identifier",
    ["5823", "Well 5823", "HDU_5823", "hartzog draw unit 5823"],
)
def test_analysis_resolves_supported_exact_identifiers(identifier):
    result = make_client().analyze(
        4,
        identifier,
        "2026-07-05T00:00:00+00:00",
        "2026-07-06T00:00:00+00:00",
    )
    assert result["well"]["name"] == "HARTZOG DRAW UNIT 5823"


def test_analysis_rejects_an_ambiguous_terminal_identifier():
    client = make_client()
    with client.engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO wells VALUES (11,4,'OTHER UNIT 5823','OTHER_5823','rod')"
        ))
    with pytest.raises(RodPumpAnalysisError, match="ambiguous"):
        client.analyze(
            4,
            "5823",
            "2026-07-05T00:00:00+00:00",
            "2026-07-06T00:00:00+00:00",
        )
