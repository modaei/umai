import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from umai.rod_pump_health.analysis import (
    ASSESSMENT_SCHEMA,
    SYSTEM_PROMPT,
    AnalysisRequest,
    _response_token_usage,
    analyze,
    coordinates,
    features,
    reduce_points,
    health_reviewer,
    Point,
)


def rectangle(height=1000):
    """A closed 100 in x height lb card, traced in SAM1 reverse-time order."""
    edges = (
        [(i * 5, 0) for i in range(21)]
        + [(100, i * height / 20) for i in range(1, 21)]
        + [(100 - i * 5, height) for i in range(1, 21)]
        + [(0, height - i * height / 20) for i in range(1, 21)]
    )
    return [Point(position=x, load=y) for x, y in reversed(edges)]


def packet(downhole=True):
    return AnalysisRequest(
        site_id=4,
        well_id=2,
        card={
            "reference": "live:900",
            "timestamp": datetime(2026, 10, 20, 9, tzinfo=timezone.utc),
            "header": {"number_points": 81, "stroke_period": 1060, "max_load": 1000, "min_load": 0},
            "surface": rectangle(),
            "downhole": rectangle(600) if downhole else [],
            "downhole_summary": {"fluid_load_lb": 600} if downhole else {},
            "period_s": 10.6,
            "point_order": "sam1_reverse_time",
            "downhole_error": None if downhole else "Downhole calculation service unavailable.",
        },
    )


def problem_review(**overrides):
    finding = {
        "side": "downhole",
        "title": "Fluid pound",
        "reasoning": "Load stays high   into the downstroke.",
        "actions": "Review the SAM1 pump-off setpoint.",
        "confidence": "medium",
    }
    return {
        "health_status": "problem",
        "summary": "Incomplete pump fillage.",
        "findings": [finding],
        "limitations": [],
        "token_usage": {"input_tokens": 2500, "output_tokens": 900, "total_tokens": 3400},
        **overrides,
    }


def test_review_sends_only_text_coordinates_without_images_or_templates():
    sent = []

    def reviewer(content):
        sent.append(content)
        return problem_review()

    result = analyze(packet(), reviewer)
    assert len(sent) == 1
    assert [part["type"] for part in sent[0]] == ["text"]
    body = json.loads(sent[0][0]["text"])
    assert "templates" not in body
    assert body["card_reference"] == "live:900"
    assert body["controller_header"]["stroke_period_s"] == 10.6
    assert body["controller_header"]["strokes_per_minute"] == 5.66
    assert "stroke_period" not in body["controller_header"]
    assert len(body["surface"]["points"]) == 81
    assert len(body["downhole"]["points"]) == 81
    assert body["downhole"]["calculation"] == {"fluid_load_lb": 600}
    assert body["surface"]["measurements"]["area_in_lb"] == 100000
    assert result["review_status"] == "complete"
    assert result["status"] == "complete"
    assert result["health_status"] == "problem"
    assert result["summary"] == "Incomplete pump fillage."
    assert result["diagnosis_status"] == "candidates"
    assert result["token_usage"]["total_tokens"] == 3400
    assert result["findings"] == [
        {
            "side": "downhole",
            "kind": "shape",
            "scope": "last_eligible_card",
            "title": "Fluid pound",
            "reasoning": "Load stays high into the downstroke.",
            "actions": "Review the SAM1 pump-off setpoint.",
            "confidence": "medium",
            "card_ids": ["live:900"],
        }
    ]


def test_coordinates_are_rounded_and_put_in_time_order():
    points = [Point(position=1.26, load=10.4), Point(position=2.0, load=20.6)]
    assert coordinates(points, "sam1_reverse_time") == [[2.0, 21], [1.3, 10]]
    assert coordinates(points, "unknown") == [[1.3, 10], [2.0, 21]]


def test_reduce_points_keeps_limit_endpoints_and_corners():
    pairs = coordinates(rectangle(), "sam1_reverse_time")
    reduced = reduce_points(pairs, 12)
    assert len(reduced) == 12
    assert reduced[0] == pairs[0] and reduced[-1] == pairs[-1]
    assert all(corner in reduced for corner in ([100.0, 0], [100.0, 1000], [0.0, 1000]))
    assert reduce_points(pairs, None) == pairs
    assert reduce_points(pairs[:5], 12) == pairs[:5]


def test_point_limits_apply_per_side_while_measurements_use_full_curves():
    sent = []

    def reviewer(content):
        sent.append(json.loads(content[0]["text"]))
        return problem_review()

    result = analyze(packet(), reviewer, {"surface": 20, "downhole": 40})
    assert len(sent[0]["surface"]["points"]) == 20
    assert len(sent[0]["downhole"]["points"]) == 40
    assert "down-sampled" in sent[0]["point_format"]
    assert result["measurements"] == analyze(packet(), lambda _: problem_review())["measurements"]


def test_healthy_review_has_no_findings():
    result = analyze(packet(), lambda _: problem_review(health_status="healthy", findings=[]))
    assert result["health_status"] == "healthy"
    assert result["diagnosis_status"] == "healthy"
    assert result["findings"] == []


def test_missing_downhole_card_is_reviewed_from_surface_as_partial():
    sent = []

    def reviewer(content):
        sent.append(json.loads(content[0]["text"]))
        return problem_review(health_status="watch")

    result = analyze(packet(downhole=False), reviewer)
    assert sent[0]["downhole"]["unavailable"] == "Downhole calculation service unavailable."
    assert "points" not in sent[0]["downhole"]
    assert result["review_status"] == "complete"
    assert result["status"] == "partial"
    assert any(item.startswith("downhole:") for item in result["limitations"])


def test_at_most_two_findings_are_kept():
    findings = [problem_review()["findings"][0] | {"title": f"Finding {i}"} for i in range(3)]
    result = analyze(packet(), lambda _: problem_review(findings=findings))
    assert [finding["title"] for finding in result["findings"]] == ["Finding 0", "Finding 1"]


def test_invalid_structured_response_fails_without_a_healthy_verdict():
    result = analyze(packet(), lambda _: {"health_status": "fine", "summary": "ok"})
    assert result["review_status"] == "failed"
    assert result["diagnosis_status"] == "unavailable"
    assert "Invalid structured LLM response" in result["review_error"]


def test_provider_failure_is_logged_and_returned_without_credentials(caplog):
    def reviewer(_):
        raise RuntimeError("401 invalid API key sk-secret123 Bearer private-token")

    result = analyze(packet(), reviewer)
    assert result["review_status"] == "failed"
    assert "LLM request" in result["review_error"]
    assert "401" in result["review_error"]
    assert "sk-secret123" not in result["review_error"]
    assert "private-token" not in caplog.text
    assert "well_id=2" in caplog.text


def test_unusable_card_does_not_call_llm(caplog):
    data = packet().model_dump()
    data["card"]["surface"] = []
    data["card"]["downhole"] = []
    result = analyze(AnalysisRequest(**data), lambda _: pytest.fail("LLM must not be called"))
    assert result["review_status"] == "unavailable"
    assert "LLM was not called" in result["review_error"]
    assert "well_id=2" in caplog.text


def test_timezone_is_required():
    data = packet().model_dump()
    data["card"]["timestamp"] = datetime(2026, 10, 1)
    with pytest.raises(ValueError):
        AnalysisRequest(**data)


def test_branch_features_require_sam1_provenance():
    assert features(rectangle())["orientation"] == "unknown"
    verified = features(rectangle(), "sam1_reverse_time")
    assert verified["orientation"] == "verified_sam1_reverse_time"


def test_prompt_describes_the_sam1_scenario_and_schema_is_strict():
    assert "Lufkin SAM1" in SYSTEM_PROMPT
    assert "no VFD" in SYSTEM_PROMPT
    assert "template" not in SYSTEM_PROMPT.lower()
    assert ASSESSMENT_SCHEMA["additionalProperties"] is False
    item = ASSESSMENT_SCHEMA["properties"]["findings"]["items"]
    assert set(item["required"]) == set(item["properties"])


def test_health_reviewer_uses_dedicated_model_effort_and_json_schema(monkeypatch):
    import langchain_openai

    created = {}

    class FakeModel:
        def __init__(self, **kwargs):
            created["init"] = kwargs

        def bind(self, **kwargs):
            created["bind"] = kwargs
            return self

        def invoke(self, messages):
            created["messages"] = messages
            return SimpleNamespace(
                content=[{"type": "text", "text": json.dumps({"health_status": "healthy"})}],
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeModel)
    settings = SimpleNamespace(
        rod_pump_health_llm_model="anthropic/claude-opus-5.5",
        rod_pump_health_llm_effort="medium",
        llm_api_key="key",
        llm_base_url="https://openrouter.ai/api/v1",
    )
    result = health_reviewer(settings)([{"type": "text", "text": "{}"}])
    assert created["init"]["model"] == "anthropic/claude-opus-5.5"
    assert created["init"]["reasoning_effort"] == "medium"
    assert created["init"]["base_url"] == "https://openrouter.ai/api/v1"
    assert created["bind"]["response_format"]["type"] == "json_schema"
    assert created["bind"]["response_format"]["json_schema"]["schema"] is ASSESSMENT_SCHEMA
    assert created["messages"][0].content == SYSTEM_PROMPT
    assert result == {"health_status": "healthy", "token_usage": {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }}


def test_internal_api_delegates_only_well_id_to_server_analyzer(monkeypatch):
    import importlib
    from umai.rod_pump_health.reader import WellAnalysisRequest

    module = importlib.import_module("umai.api.app")
    calls = []

    class Analyzer:
        def _well(self, well_id):
            return {"id": well_id, "site_id": 4}

        def analyze(self, well_id, recent_card=None):
            calls.append(well_id)
            return {"review_status": "complete", "well_id": well_id}

    monkeypatch.setattr(module, "analyzer_from_settings", lambda settings: Analyzer())
    app = module.create_app()
    route = next(r for r in app.routes if r.path == "/rod-pump-health-analysis")
    result = route.endpoint(WellAnalysisRequest(well_id=23))
    assert result["review_status"] == "complete"
    assert result["parameter_trends"] == {"status": "disabled"}
    assert calls == [23]


def test_response_token_usage_normalizes_langchain_and_openai_metadata():
    langchain_response = SimpleNamespace(
        usage_metadata={"input_tokens": 900, "output_tokens": 120, "total_tokens": 1020}
    )
    assert _response_token_usage(langchain_response) == {
        "input_tokens": 900,
        "output_tokens": 120,
        "total_tokens": 1020,
    }
    openai_response = SimpleNamespace(
        response_metadata={
            "token_usage": {"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550}
        }
    )
    assert _response_token_usage(openai_response) == {
        "input_tokens": 500,
        "output_tokens": 50,
        "total_tokens": 550,
    }
