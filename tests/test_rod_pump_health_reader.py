import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import create_engine, text

from umai.rod_pump_health.reader import RodPumpHealthAnalyzer, physical_points


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
    with archive.begin() as connection:
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
        assert any(part["type"] == "image_url" for part in content)
        return {"findings": [], "limitations": []}

    service = RodPumpHealthAnalyzer(operational, archive, "UTC", "http://reports", reviewer, post)
    yield service, calls
    operational.dispose()
    archive.dispose()


def test_well_id_resolves_site_and_retrieves_both_card_sides(analyzer):
    service, calls = analyzer
    result = service.analyze(23)
    assert result["site_id"] == 4
    assert result["well_id"] == 23
    assert len(result["input_cards"]) == 14
    assert result["input_cards"][0]["surface"][20]["position"] == 100
    assert result["input_cards"][0]["downhole"][20]["position"] == 100
    assert result["status"] == "complete"  # Latest baseline supplies the local fallback.
    assert calls["report"][0]["site_id"] == 4
    assert calls["report"][0]["params"]["surface_cards"][0]["id"] == "archive:1"
    assert calls["review"] == 0


def test_local_result_reuses_cache_and_never_calls_llm(analyzer):
    service, calls = analyzer
    first = service.analyze(23)
    with service.engine.begin() as connection:
        connection.execute(
            text("INSERT INTO rod_pump_health_analyses VALUES(23,:signature,:findings)"),
            {
                "signature": first["input_signature"],
                "findings": json.dumps({"review_status": "complete"}),
            },
        )
    assert service.analyze(23)["unchanged"] is True
    assert calls["review"] == 0
    points = surface_points()
    points[5]["load"] = 25
    with service.archive_engine.begin() as connection:
        connection.execute(
            text("UPDATE rod_pump_card_archive SET data_points=:points WHERE id=1"),
            {"points": json.dumps(points)},
        )
    assert service.analyze(23)["unchanged"] is False
    assert calls["review"] == 0


def test_downhole_failure_preserves_surface_analysis(analyzer):
    service, _ = analyzer

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    service.http_post = unavailable
    result = service.analyze(23)
    assert result["status"] == "partial"
    assert result["review_status"] == "complete"
    assert all(card["surface"] and not card["downhole"] for card in result["input_cards"])
    assert all(card["downhole_error"] for card in result["input_cards"])


def test_unconfigured_or_unknown_wells_are_rejected_before_archive_or_report_access(analyzer):
    service, calls = analyzer
    for well_id in [24, 999]:
        with pytest.raises(ValueError):
            service.analyze(well_id)
    assert calls["report"] == []
    assert calls["review"] == 0


def test_daily_median_deduplication_and_event_exclusion(analyzer):
    service, _ = analyzer
    first = int(datetime(2026, 10, 1, 9, tzinfo=timezone.utc).timestamp())
    with service.archive_engine.begin() as connection:
        for card_id, offset, reason in [
            (101, 0, "daily_baseline"),
            (102, 100, "daily_baseline"),
            (103, 200, "daily_baseline"),
            (104, 300, "daily_baseline"),
            (105, 400, "event"),
        ]:
            connection.execute(
                text(
                    "INSERT INTO rod_pump_card_archive SELECT :id,well_id,card_timestamp+:offset,"
                    "pull_timestamp,source_card_id,card_type,data_points,info,:reason,retention_state "
                    "FROM rod_pump_card_archive WHERE id=1"
                ),
                {"id": card_id, "offset": offset, "reason": json.dumps([reason])},
            )
    cards, _, _ = service._daily_cards(23)
    assert len(cards) == 14
    assert cards[0]["id"] == 102  # Lower median after deduplicating repeated capture time.
    assert cards[0]["card_timestamp"] == first + 100


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
    assert len(result["input_cards"]) == 15
    assert result["input_cards"][-1]["reference"] == "archive:99"
    assert result["input_cards"][-1]["roles"] == ["recent"]
    assert result["matched_cards"]["surface"] == "archive:99"
    assert calls["review"] == 0
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
    packet, _, _ = service.prepare(23, reference)
    assert len(packet.cards) == 14
    assert packet.cards[-1].roles == ["baseline", "recent"]


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


def test_previous_matcher_version_is_not_reused(analyzer):
    import hashlib

    service, calls = analyzer
    packet, _, _ = service.prepare(23)
    old_signature = hashlib.sha256(json.dumps(
        [packet.model_dump(mode="json"), "local-dynograph-matching-v1", "dynograph-templates-v2"],
        sort_keys=True,
    ).encode()).hexdigest()
    with service.engine.begin() as connection:
        connection.execute(text("INSERT INTO rod_pump_health_analyses VALUES(23,:signature,:findings)"), {
            "signature": old_signature,
            "findings": json.dumps({"review_status": "complete", "matching_method": "local-dynograph-matching-v1"}),
        })
    result = service.analyze(23)
    assert result["unchanged"] is False
    assert result["matching_method"] == "local-dynograph-matching-v11"
    assert calls["review"] == 0
