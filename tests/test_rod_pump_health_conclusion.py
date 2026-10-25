import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, text

from umai.rod_pump_health.conclusion import ConclusionAnalyzer, historical_findings


@pytest.fixture
def analyzer():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE rod_pump_health_analyses(well_id INTEGER, findings TEXT, input_cards TEXT)"
        ))
    health = SimpleNamespace(
        engine=engine, timezone=ZoneInfo("UTC"), _well=lambda _: {"id": 23, "site_id": 4},
    )
    calls = []

    def reviewer(content):
        calls.append(content)
        return {
            "title": "Intermittent underfilling",
            "body": (
                "Fillage and card findings suggest limited inflow.\nGas interference remains possible."
            ),
            "token_usage": {"input_tokens": 200, "output_tokens": 40},
        }

    yield ConclusionAnalyzer(health, reviewer), calls
    engine.dispose()


def cards():
    return {
        "review_status": "complete",
        "input_cards": [{"reference": "live:1", "roles": ["recent"]}],
        "findings": [
            {"scope": "historical", "side": "downhole", "title": "Fluid pound",
             "reasoning": "Historical unloading transition.", "confidence": "high"},
            {"scope": "last_eligible_card", "side": "surface", "title": "Latest-only friction"},
            {"scope": "historical", "side": "surface", "title": "Uncertain", "confidence": "LOW"},
        ],
    }


def parameters():
    return {"status": "complete", "trends": [{"parameter": "Pump Fillage", "title": "Decline"}],
            "interpretation": "Variable fillage suggests intermittent underfilling."}


def test_conclusion_is_a_separate_text_review_of_retained_historical_diagnoses(analyzer):
    service, calls = analyzer
    result = service.analyze(23, cards(), parameters())
    assert result["title"] == "Intermittent underfilling"
    assert "\n" not in result["body"]
    assert result["status"] == "complete"
    assert result["token_usage"]["input_tokens"] == 200
    assert [part["type"] for part in calls[0]] == ["text"]
    packet = json.loads(calls[0][0]["text"])
    assert packet["surface"] == []
    assert [row["title"] for row in packet["downhole"]] == ["Fluid pound"]
    assert packet["parameter_trends"]["interpretation"] == parameters()["interpretation"]
    assert "input_cards" not in packet


def test_cached_card_findings_are_resolved_and_unchanged_conclusion_is_reused(analyzer):
    service, calls = analyzer
    first = service.analyze(23, cards(), parameters())
    saved = {**cards(), "conclusion": first}
    with service.engine.begin() as connection:
        connection.execute(text("INSERT INTO rod_pump_health_analyses VALUES(23,:review,:cards)"), {
            "review": json.dumps(saved), "cards": json.dumps(cards()["input_cards"]),
        })
    cached = service.analyze(23, {"unchanged": True}, parameters())
    assert cached["unchanged"] is True
    assert len(calls) == 1
    changed = parameters()
    changed["interpretation"] = "Stable fillage with increased loads suggests friction."
    service.analyze(23, {"unchanged": True}, changed)
    assert len(calls) == 2


def test_missing_reviews_do_not_produce_a_healthy_conclusion(analyzer):
    service, calls = analyzer
    assert service.analyze(23, {"review_status": "failed"}, {"status": "failed"})["status"] \
        == "insufficient_data"
    assert calls == []
    partial = service.analyze(23, cards(), {"status": "failed"})
    assert partial["status"] == "partial"
    assert json.loads(calls[0][0]["text"])["parameter_trends"]["status"] == "failed"


def test_successful_empty_reviews_are_still_synthesized(analyzer):
    service, calls = analyzer
    service.analyze(23, {"review_status": "complete", "findings": []},
                    {"status": "complete", "trends": [], "interpretation": "No unusual trends."})
    assert len(calls) == 1


def test_legacy_historical_scope_and_healthy_case_are_handled():
    review = {"findings": [
        {"kind": "Healthy", "side": "surface", "title": "Healthy"},
        {"side": "surface", "title": "Historical loading", "card_ids": ["live:1"]},
        {"side": "downhole", "title": "Past loading", "card_ids": [12]},
    ]}
    result = historical_findings(review, [{"reference": "live:1", "roles": ["recent"]}])
    assert [row["title"] for row in result["surface"]] == ["Historical loading"]
    assert [row["title"] for row in result["downhole"]] == ["Past loading"]
