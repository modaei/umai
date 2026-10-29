import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import create_engine, text

from umai.rod_pump_health.reader import RecentCardReference, RodPumpHealthAnalyzer, physical_points


def surface_points():
    edges = (
        [(i * 500, 0) for i in range(21)]
        + [(10000, i * 50) for i in range(1, 21)]
        + [(10000 - i * 500, 1000) for i in range(1, 21)]
        + [(0, 1000 - i * 50) for i in range(1, 21)]
    )
    return [{"position": x, "load": y} for x, y in edges]


@pytest.fixture
def analyzer():
    operational, archive = create_engine("sqlite://"), create_engine("sqlite://")
    with operational.begin() as connection:
        connection.execute(text("CREATE TABLE wells(id INTEGER, site_id INTEGER, pump_type TEXT)"))
        connection.execute(text("INSERT INTO wells VALUES(23,4,'rod'),(24,5,'rod')"))
        connection.execute(
            text(
                "CREATE TABLE report_configurations(site_id INTEGER, function_name TEXT, "
                "related_entities TEXT)"
            )
        )
        connection.execute(
            text("INSERT INTO report_configurations VALUES(4,'rod_pump_monitoring',:entities)"),
            {"entities": json.dumps({"well_ids": [23]})},
        )
        connection.execute(
            text(
                "CREATE TABLE rod_pump_health_analyses(well_id INTEGER, input_signature TEXT, "
                "findings TEXT)"
            )
        )
        connection.execute(
            text("CREATE TABLE well_shutdowns(well_id INTEGER, start TEXT, end TEXT, downtime_code TEXT)")
        )
    with archive.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE rod_pump_card_archive_metrics(rod_pump_card_archive_id INTEGER, "
                "surface_max_load REAL, surface_min_load REAL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE rod_pump_card_archive(id INTEGER, well_id INTEGER, "
                "card_timestamp INTEGER, "
                "pull_timestamp INTEGER, source_card_id INTEGER, card_type TEXT, data_points TEXT, "
                "info TEXT, archive_reasons TEXT, retention_state TEXT)"
            )
        )
        for index in range(14):
            timestamp = int(
                (datetime(2026, 10, 1, 9, tzinfo=timezone.utc) + timedelta(days=index)).timestamp()
            )
            connection.execute(
                text(
                    "INSERT INTO rod_pump_card_archive VALUES(:id,23,:time,:time,:id,'current',"
                    ":points,:info,:reasons,'retained')"
                ),
                {
                    "id": index + 1,
                    "time": timestamp,
                    "points": json.dumps(surface_points()),
                    "info": json.dumps({"number_points": 81, "stroke_period": 1060}),
                    "reasons": json.dumps(["daily_baseline"]),
                },
            )
            if index < 10:
                # Cards archived since the last nightly run have no metrics yet.
                connection.execute(
                    text("INSERT INTO rod_pump_card_archive_metrics VALUES(:id,1000,0)"),
                    {"id": index + 1},
                )
    calls = {"report": [], "review": 0}

    def post(url, **kwargs):
        calls["report"].append(kwargs["json"])
        cards = kwargs["json"]["params"]["surface_cards"]

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                # The report API's existing JSON-string envelope is supported.
                return json.dumps(
                    {
                        "cards": [
                            {
                                "ok": True,
                                "card_id": card["id"],
                                "positions": [p["position"] / 100 for p in card["data_points"]],
                                "loads": [p["load"] for p in card["data_points"]],
                            }
                            for card in reversed(cards)
                        ]
                    }
                )

        return Response()

    def reviewer(content):
        calls["review"] += 1
        assert [part["type"] for part in content] == ["text"]
        return {"health_status": "healthy", "summary": "Normal card.", "findings": [], "limitations": []}

    # The fixture cards span only 1,000 lb, so the absolute surface collapse limit is off.
    service = RodPumpHealthAnalyzer(
        operational, archive, "UTC", "http://reports", reviewer, post,
        surface_load={"low_load_range_lb": 0},
    )
    yield service, calls
    operational.dispose()
    archive.dispose()


def archived_reference(card_id=14):
    """Reference one fixture archive card the way UMetriqs' selector would."""
    capture = int(
        (datetime(2026, 10, 1, 9, tzinfo=timezone.utc) + timedelta(days=card_id - 1)).timestamp()
    )
    return RecentCardReference(
        source="archive",
        id=card_id,
        pull_timestamp=capture,
        card_timestamp=capture,
        pumping_window_start=capture - 600,
        pumping_window_end=capture + 300,
    )


def test_point_limits_reduce_only_sent_coordinates_and_change_signature(analyzer):
    service, _ = analyzer
    full = service.analyze(23, archived_reference())
    sent = []
    review = service.reviewer
    service.reviewer = lambda content: sent.append(json.loads(content[0]["text"])) or review(content)
    service.point_limits = {"surface": 10, "downhole": 12}
    reduced = service.analyze(23, archived_reference())
    assert len(sent[0]["surface"]["points"]) == 10
    assert len(sent[0]["downhole"]["points"]) == 12
    assert reduced["input_cards"] == full["input_cards"]
    assert reduced["input_signature"] != full["input_signature"]


def test_selected_card_is_the_only_card_reviewed_with_both_sides(analyzer):
    service, calls = analyzer
    result = service.analyze(23, archived_reference())
    assert result["site_id"] == 4
    assert result["well_id"] == 23
    assert len(result["input_cards"]) == 1
    card = result["input_cards"][0]
    assert card["reference"] == "archive:14"
    assert card["roles"] == ["recent"]
    assert card["surface"][20]["position"] == 100
    assert card["downhole"][20]["position"] == 100
    assert result["status"] == "complete"
    assert result["health_status"] == "healthy"
    assert result["window_start_at"] == result["latest_card_at"] == "2026-10-14T09:00:00+00:00"
    assert len(calls["report"]) == 1
    assert [c["id"] for c in calls["report"][0]["params"]["surface_cards"]] == ["archive:14"]
    assert calls["review"] == 1


def test_missing_selected_card_is_rejected_without_reading_reports(analyzer):
    service, calls = analyzer
    with pytest.raises(ValueError, match="No eligible pumping card"):
        service.analyze(23)
    assert calls["report"] == []
    assert calls["review"] == 0


def test_unchanged_saved_result_skips_llm_but_changed_source_reanalyzes(analyzer):
    service, calls = analyzer
    first = service.analyze(23, archived_reference())
    with service.engine.begin() as connection:
        connection.execute(
            text("INSERT INTO rod_pump_health_analyses VALUES(23,:signature,:findings)"),
            {
                "signature": first["input_signature"],
                "findings": json.dumps({"review_status": "complete"}),
            },
        )
    assert service.analyze(23, archived_reference())["unchanged"] is True
    assert calls["review"] == 1
    points = surface_points()
    points[5]["load"] = 25
    with service.archive_engine.begin() as connection:
        connection.execute(
            text("UPDATE rod_pump_card_archive SET data_points=:points WHERE id=14"),
            {"points": json.dumps(points)},
        )
    assert service.analyze(23, archived_reference())["unchanged"] is False
    assert calls["review"] == 2


def test_downhole_failure_preserves_surface_analysis(analyzer):
    service, _ = analyzer

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    service.http_post = unavailable
    result = service.analyze(23, archived_reference())
    assert result["status"] == "partial"
    assert result["review_status"] == "complete"
    card = result["input_cards"][0]
    assert card["surface"] and not card["downhole"]
    assert card["downhole_error"]


def test_unconfigured_or_unknown_wells_are_rejected_before_archive_or_report_access(analyzer):
    service, calls = analyzer
    for well_id in [24, 999]:
        with pytest.raises(ValueError):
            service.analyze(well_id, archived_reference())
    assert calls["report"] == []
    assert calls["review"] == 0


def test_unit_conversion_and_bad_curves_are_not_partially_repaired():
    assert physical_points([{"position": 8501, "load": 12000}], 100) == [
        {"position": 85.01, "load": 12000}
    ]
    assert physical_points([{"position": 10, "load": float("inf")}]) == []
    assert physical_points([{"position": 10, "load": 20}, {"position": "bad", "load": 30}]) == []
    assert physical_points(surface_points(), declared_empty=True) == []


def test_selected_archive_card_resolves_even_without_baseline_reason(analyzer, monkeypatch):
    from umai.rod_pump_health.reader import RecentCardReference
    import umai.rod_pump_health.reader as reader

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 16, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(reader, "datetime", Clock)
    service, calls = analyzer
    capture = int(datetime(2026, 10, 15, 9, tzinfo=timezone.utc).timestamp())
    with service.archive_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO rod_pump_card_archive SELECT 99,well_id,:capture,:capture,99,"
                "card_type,data_points,info,:reasons,'pending' FROM rod_pump_card_archive WHERE id=1"
            ),
            {"capture": capture, "reasons": json.dumps(["event"])},
        )
    reference = RecentCardReference(
        source="archive",
        id=99,
        pull_timestamp=capture,
        card_timestamp=capture,
        pumping_window_start=capture - 600,
        pumping_window_end=capture + 300,
    )
    result = service.analyze(23, reference)
    assert len(result["input_cards"]) == 1
    assert result["input_cards"][0]["reference"] == "archive:99"
    assert result["input_cards"][0]["roles"] == ["recent"]
    assert calls["review"] == 1
    assert result["status"] == "complete"


def test_recent_reference_cannot_cross_wells_or_lie_about_pumping_window(analyzer, monkeypatch):
    from umai.rod_pump_health.reader import RecentCardReference
    import umai.rod_pump_health.reader as reader

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 16, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(reader, "datetime", Clock)
    service, _ = analyzer
    capture = int(datetime(2026, 10, 14, 9, tzinfo=timezone.utc).timestamp())
    reference = RecentCardReference(
        source="archive",
        id=14,
        pull_timestamp=capture,
        card_timestamp=capture,
        pumping_window_start=capture - 600,
        pumping_window_end=capture + 300,
    )
    with pytest.raises(ValueError):
        service._recent_card(24, reference)
    invalid = reference.model_copy(update={"pumping_window_end": capture + 200})
    with pytest.raises(ValueError):
        service._recent_card(23, invalid)
    packet = service.prepare(23, reference)
    assert packet.card.reference == "archive:14"
    assert packet.card.roles == ["recent"]


def test_live_reference_reads_its_linked_pull_and_survives_exact_archive_fallback(
    analyzer, monkeypatch
):
    from umai.rod_pump_health.reader import RecentCardReference
    import umai.rod_pump_health.reader as reader

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 16, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(reader, "datetime", Clock)
    service, _ = analyzer
    capture = int(datetime(2026, 10, 15, 9, tzinfo=timezone.utc).timestamp())
    with service.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE rod_pump_cards(id INTEGER, timestamp INTEGER, "
                "card_type TEXT, info TEXT, data_points TEXT)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE rod_pump_monitoring_data("
                "id INTEGER, well_id INTEGER, pull_timestamp INTEGER)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE rod_pump_monitoring_data_cards("
                "rod_pump_monitoring_data_id INTEGER, rod_pump_card_id INTEGER)"
            )
        )
        connection.execute(
            text("INSERT INTO rod_pump_cards VALUES(900,:time,'current',:info,:points)"),
            {
                "time": capture,
                "info": json.dumps({"number_points": 81, "stroke_period": 1060}),
                "points": json.dumps(surface_points()),
            },
        )
        connection.execute(
            text("INSERT INTO rod_pump_monitoring_data VALUES(50,23,:time)"), {"time": capture}
        )
        connection.execute(text("INSERT INTO rod_pump_monitoring_data_cards VALUES(50,900)"))
    reference = RecentCardReference(
        source="live",
        id=900,
        monitoring_data_id=50,
        pull_timestamp=capture,
        card_timestamp=capture,
        pumping_window_start=capture - 600,
        pumping_window_end=capture + 300,
    )
    result = service.analyze(23, reference)
    assert result["input_cards"][-1]["reference"] == "live:900"
    assert result["input_cards"][-1]["archive_id"] is None
    assert result["status"] == "complete"
    with service.archive_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO rod_pump_card_archive SELECT "
                "99,well_id,:time,:time,900,card_type,data_points,info,archive_reasons,'pending' "
                "FROM rod_pump_card_archive WHERE id=1"
            ),
            {"time": capture},
        )
    with service.engine.begin() as connection:
        connection.execute(text("DELETE FROM rod_pump_cards WHERE id=900"))
    recovered = service.analyze(23, reference)
    assert recovered["input_cards"][-1]["reference"] == "archive:99"
    assert recovered["status"] == "complete"
