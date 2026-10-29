import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import text

from umai.rod_pump_health.surface_load import card_statuses, surface_load_check

from test_rod_pump_health_reader import analyzer, archived_reference, surface_points  # noqa: F401

DENVER = ZoneInfo("America/Denver")
HOUR = 3600
START = int(datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp())


def cards(*loads, step=6 * HOUR):
    """Cards every six hours with (max_load, min_load)."""
    return [
        {"id": index + 1, "time": START + index * step, "max_load": high, "min_load": low}
        for index, (high, low) in enumerate(loads)
    ]


NORMAL = (28000, 15900)  # HDU_2510 before its tubing leak: 12,100 lb range


def test_each_card_is_compared_with_the_median_of_the_wells_good_cards():
    rows = card_statuses(
        cards(NORMAL, NORMAL, NORMAL, (27500, 17000), (27000, 18000), (26000, 18000), (22800, 19700))
    )
    assert [row["status"] for row in rows] == [
        "no_baseline", "no_baseline", "no_baseline", "normal", "load_dropping", "load_dropping", "collapse",
    ]
    assert rows[3]["normal"] == {"max_load": 28000, "min_load": 15900, "load_range": 12100}
    assert rows[4]["range_ratio"] == pytest.approx(9000 / 12100)


def test_low_range_is_a_collapse_without_a_baseline_and_can_be_turned_off():
    assert card_statuses(cards((5000, 2500)))[0]["status"] == "collapse"
    assert card_statuses(cards((5000, 2500)), {"low_load_range_lb": 0})[0]["status"] == "no_baseline"


def test_collapsed_cards_never_become_the_new_normal():
    rows = card_statuses(cards(*[NORMAL] * 3, *[(22800, 19700)] * 20))
    assert {row["status"] for row in rows[3:]} == {"collapse"}
    assert rows[-1]["normal"]["load_range"] == 12100


def test_baseline_uses_only_the_last_cards_within_the_window():
    rows = card_statuses(cards(*[(30000, 10000)] * 3, *[NORMAL] * 4), {"baseline_cards": 3})
    assert rows[-1]["normal"]["load_range"] == 12100
    old = cards(*[(30000, 10000)] * 3, NORMAL, step=5 * 86400)
    assert card_statuses(old, {"baseline_days": 14})[-1]["status"] == "no_baseline"


def test_collapse_details_for_the_current_card():
    history = cards(*[NORMAL] * 4, (22800, 19700), (22800, 19700))
    current = {"time": history[-1]["time"] + 6 * HOUR, "max_load": 22800, "min_load": 19700}
    check = surface_load_check(history, current, DENVER)
    assert check["status"] == "collapse"
    assert check["load_range"] == 3100
    assert check["normal"] == {"max_load": 28000, "min_load": 15900, "load_range": 12100}
    assert check["range_ratio"] == pytest.approx(0.256, abs=1e-3)
    assert check["cards"] == 3
    assert check["since"] == datetime.fromtimestamp(history[4]["time"], DENVER).isoformat()
    assert check["last_normal_card_at"] == datetime.fromtimestamp(history[3]["time"], DENVER).isoformat()
    assert check["normal_card_id"] == 4
    # HDU_2510's tubing leak sat at about 45% of its normal card.
    assert check["card_level_pct"] == 44
    assert check["recovering"] is False
    assert check["after_shutdown"] is None
    assert "tubing leak" in check["hint"]
    assert len(check["trend"]) == 7 and check["trend"][-1] == [current["time"], 22800, 19700]


def test_normal_cards_carry_no_collapse_details():
    history = cards(*[NORMAL] * 5)
    check = surface_load_check(history, {"time": START + 30 * HOUR, "max_load": 28100, "min_load": 15800}, DENVER)
    assert check["status"] == "normal"
    assert "since" not in check and "trend" not in check


def test_a_gap_of_more_than_five_days_starts_a_new_collapse():
    history = cards(*[NORMAL] * 3, (22800, 19700))
    current = {"time": history[-1]["time"] + 6 * 86400, "max_load": 22800, "min_load": 19700}
    check = surface_load_check(history, current, DENVER)
    assert check["cards"] == 1
    assert check["last_normal_card_at"] == datetime.fromtimestamp(history[2]["time"], DENVER).isoformat()


def test_recovering_and_after_shutdown_change_the_hint():
    history = cards(*[NORMAL] * 3, (21000, 19000), (21500, 18500))
    current = {"time": history[-1]["time"] + 6 * HOUR, "max_load": 22500, "min_load": 18000}
    check = surface_load_check(history, current, DENVER)
    assert check["recovering"] is True and "coming back" in check["hint"]
    since = datetime.fromtimestamp(history[3]["time"], DENVER)
    for hours, expected in ((11, True), (13, False)):
        shutdown = {"end": since - timedelta(hours=hours), "downtime_code": "PM"}
        check = surface_load_check(history, current, DENVER, shutdowns=[shutdown])
        assert (check["after_shutdown"] is not None) is expected
    check = surface_load_check(
        history, current, DENVER,
        shutdowns=[{"end": since - timedelta(hours=2), "downtime_code": "PM"}],
    )
    assert check["after_shutdown"]["downtime_code"] == "PM"
    assert "refilling" in check["hint"]


def scaled_points(low, high):
    """The fixture card shape with its loads moved to the low..high range."""
    return [
        {"position": point["position"], "load": low + point["load"] / 1000 * (high - low)}
        for point in surface_points()
    ]


def collapse_card_14(service):
    with service.archive_engine.begin() as connection:
        connection.execute(
            text("UPDATE rod_pump_card_archive SET data_points=:points WHERE id=14"),
            {"points": json.dumps(scaled_points(400, 600))},
        )


def test_history_reads_metrics_falls_back_to_points_and_skips_later_cards(analyzer):  # noqa: F811
    service, _ = analyzer
    packet = service.prepare(23, archived_reference(12))
    history = service._surface_history(23, packet.card)
    assert [entry["id"] for entry in history] == list(range(1, 12))
    with service.archive_engine.begin() as connection:
        connection.execute(text("UPDATE rod_pump_card_archive_metrics SET surface_max_load=900"))
    history = service._surface_history(23, packet.card)
    assert {entry["max_load"] for entry in history[:10]} == {900}
    assert history[10]["max_load"] == 1000  # card 11 has no metrics; read from its points


def test_normal_surface_load_is_reported_without_changing_the_review(analyzer):  # noqa: F811
    service, calls = analyzer
    result = service.analyze(23, archived_reference())
    assert result["surface_load"]["status"] == "normal"
    assert result["surface_load"]["range_ratio"] == 1
    assert result["routing"]["reason"] == "classifier unavailable"
    assert calls["review"] == 1


def test_surface_collapse_goes_to_the_llm_with_its_history(analyzer):  # noqa: F811
    from test_rod_pump_health_classifier import FakeClassifier, save_review

    service, calls = analyzer
    service.classifier = FakeClassifier()
    service.routing["min_downhole_load_range_lb"] = 0
    save_review(service, service.analyze(23, archived_reference(13)))
    collapse_card_14(service)
    sent = []
    review = service.reviewer
    service.reviewer = lambda content: sent.append(json.loads(content[0]["text"])) or review(content)
    result = service.analyze(23, archived_reference(14))
    assert result["routing"] == {"llm_called": True, "reason": "surface load collapse (range 20% of normal)"}
    history = sent[0]["surface"]["load_history"]
    assert history["status"] == "collapse" and history["cards"] == 1
    assert "trend" not in history and "hint" not in history
    surface = result["surface_load"]
    assert surface["normal_card"]["reference"] == "archive:13"
    assert surface["normal_card"]["points"][0] == [0.0, 0]
    assert len(surface["trend"]) == 14
    assert "normal_card_id" not in surface


def test_surface_status_is_part_of_the_signature(analyzer):  # noqa: F811
    service, _ = analyzer
    service.point_limits = {"surface": 500, "downhole": 500}
    first = service.analyze(23, archived_reference())["input_signature"]
    with service.archive_engine.begin() as connection:
        connection.execute(text("UPDATE rod_pump_card_archive_metrics SET surface_max_load=4000"))
        connection.execute(text("DELETE FROM rod_pump_card_archive WHERE id IN (11,12,13)"))
    # Same selected card, but its range is now a quarter of the well's normal.
    assert service.analyze(23, archived_reference())["input_signature"] != first


def test_collapse_after_a_recorded_shutdown_is_marked(analyzer):  # noqa: F811
    service, _ = analyzer
    collapse_card_14(service)
    with service.engine.begin() as connection:
        connection.execute(text("INSERT INTO well_shutdowns VALUES(23,'2026-10-13 20:00:00','2026-10-14 03:00:00','PM')"))
    surface = service.analyze(23, archived_reference())["surface_load"]
    assert surface["after_shutdown"] == {"end": "2026-10-14T03:00:00+00:00", "downtime_code": "PM"}
    assert "refilling" in surface["hint"]


def test_a_failing_surface_check_does_not_stop_the_review(analyzer, caplog):  # noqa: F811
    service, calls = analyzer
    with service.archive_engine.begin() as connection:
        connection.execute(text("DROP TABLE rod_pump_card_archive_metrics"))
    result = service.analyze(23, archived_reference())
    assert result["surface_load"] is None
    assert result["review_status"] == "complete"
    assert "Surface load check failed" in caplog.text
