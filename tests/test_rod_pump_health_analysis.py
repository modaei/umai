from datetime import datetime, timedelta, timezone
import json

import pytest
from umai.rod_pump_health.analysis import AnalysisRequest, analyze, features, Point, trends
from umai.rod_pump_health.templates import TEMPLATES, VERSION as TEMPLATE_VERSION


def test_templates_remove_full_pump_and_like_from_names():
    names = [template["title"] for template in TEMPLATES]

    assert len(names) == 11
    assert all("like" not in name.lower() for name in names)
    assert all(template["id"] != "dh_full" for template in TEMPLATES)
    assert TEMPLATE_VERSION == "dynograph-templates-v2"


def rectangle(load=1000):
    # Closed counterclockwise loop with sufficient samples on each edge.
    return [
        Point(position=x, load=y)
        for x, y in (
            [(i * 5, 0) for i in range(21)]
            + [(100, i * load / 20) for i in range(1, 21)]
            + [(100 - i * 5, load) for i in range(1, 21)]
            + [(0, load - i * load / 20) for i in range(1, 21)]
        )
    ]


def packet():
    return AnalysisRequest(
        site_id=1,
        well_id=2,
        cards=[
            dict(
                archive_id=i + 1,
                timestamp=datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(days=i),
                surface=[p.model_dump() for p in rectangle(1000 - i * 20)],
                downhole=[],
                point_order="sam1_reverse_time",
            )
            for i in range(14)
        ],
    )


def branch_match_evidence():
    return {
        "match_strength": "moderate",
        "matching_features": [
            "Upper branch curvature and transition agree.",
            "Lower branch direction and separation agree.",
        ],
        "mismatching_features": [],
    }


def test_physical_thinning_survives_normalization():
    a, b = features(rectangle()), features(rectangle(500))
    assert a["area_in_lb"] == 100000
    assert b["area_in_lb"] == 50000
    assert a["normalized_area"] == b["normalized_area"] == 1
    rows = [
        {"timestamp": c.timestamp.isoformat(), "features": features(c.surface)} for c in packet().cards
    ]
    result = trends(rows)
    assert result["metrics"]["area_in_lb"]["slope_per_day"] == -2000
    assert result["metrics"]["area_in_lb"]["percent_change"] < 0


def test_sparse_history_and_invalid_curves_do_not_become_healthy():
    assert features([])["status"] == "invalid"
    rows = [
        {"timestamp": c.timestamp.isoformat(), "features": features(c.surface)}
        for c in packet().cards[:2]
    ]
    assert trends(rows)["status"] == "insufficient_history"


def test_both_images_and_measured_json_are_sent_and_findings_are_preserved():
    received = []

    def reviewer(content):
        received.extend(content)
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "trend",
                    "title": "Declining card area",
                    "suspected_problem": "Possible unloading",
                    "template_ids": ["sf_unloading"],
                    "card_ids": [1, 7, 14],
                    "reasoning": "Area decreases through the history.",
                    "confidence": "high",
                    "actions": "Review pump-off settings.\nCheck controller fillage data.",
                    "supporting_features": ["Reduced branch separation"],
                    "contradicting_features": [],
                    "alternatives": ["Operating speed changes"],
                    "match_strength": "moderate",
                    "evidence_keys": ["surface.trend.area_in_lb.absolute_change"],
                }
            ],
            "token_usage": {
                "input_tokens": 1200,
                "output_tokens": 180,
                "total_tokens": 1380,
            },
            "limitations": [],
        }

    result = analyze(packet(), reviewer)
    assert result["review_status"] == "complete"
    assert result["status"] == "partial"  # Missing pump curves are explicit.
    assert result["findings"][0]["template_ids"] == ["sf_unloading"]
    assert result["findings"][0]["confidence"] == "high"
    assert result["findings"][0]["actions"] == (
        "Review pump-off settings. Check controller fillage data."
    )
    assert result["token_usage"]["input_tokens"] == 1200
    assert result["token_usage"]["output_tokens"] == 180
    assert received[0]["type"] == "text"
    assert any(c["type"] == "image_url" for c in received)
    assert "period_s" in received[0]["text"]
    templates = json.loads(received[0]["text"])["templates"]
    assert all(set(template) == {"id", "side", "title", "points"} for template in templates)
    assert "source" not in received[0]["text"]


def test_unresolved_shape_references_are_still_returned_to_the_user():
    def reviewer(_):
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "shape",
                    "title": "Load fell 99 percent",
                    "suspected_problem": "Fault",
                    "template_ids": ["imaginary"],
                    "card_ids": [999],
                    "reasoning": "Fault",
                    "supporting_features": [],
                    "contradicting_features": [],
                    "alternatives": [],
                    "match_strength": "strong",
                    "evidence_keys": ["surface.card.999.peak_load_lb"],
                }
            ],
            "limitations": [],
        }

    result = analyze(packet(), reviewer)
    assert result["review_status"] == "complete"
    assert result["findings"][0]["title"] == "Load fell 99 percent"
    assert result["diagnosis_status"] == "candidates"


def test_timezone_required_and_duplicate_ids_rejected():
    data = packet().model_dump()
    data["cards"][0]["timestamp"] = datetime(2026, 10, 1)
    with pytest.raises(ValueError):
        AnalysisRequest(**data)
    data = packet().model_dump()
    data["cards"][1]["archive_id"] = 1
    with pytest.raises(ValueError):
        AnalysisRequest(**data)


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
    from types import SimpleNamespace
    monkeypatch.setattr(module, "parameter_analyzer_from_settings", lambda settings: SimpleNamespace(
        analyze=lambda well_id: {"status": "complete", "trends": []},
    ))
    monkeypatch.setattr(module, "conclusion_analyzer_from_settings", lambda settings: SimpleNamespace(
        analyze=lambda *args: {"status": "complete", "title": "Healthy", "body": "No problem found."},
    ))
    app = module.create_app()
    route = next(r for r in app.routes if r.path == "/rod-pump-health-analysis")
    result = route.endpoint(WellAnalysisRequest(well_id=23))
    assert result["review_status"] == "complete"
    assert calls == [23]
    with pytest.raises(ValueError):
        WellAnalysisRequest(well_id=23, cards=[])


def test_branch_features_require_provenance_and_numeric_prose_is_allowed():
    assert features(rectangle())["orientation"] == "unknown"
    # Reverse-time provenance allows direction-specific geometry.
    verified = features(rectangle(), "sam1_reverse_time")
    assert verified["orientation"] == "verified_sam1_reverse_time"
    assert verified["upstroke_transition_width_percent"] == 0

    data = packet().model_dump()
    data["cards"][-1]["roles"] = ["baseline", "recent"]
    payload = AnalysisRequest(**data)

    def reviewer(_):
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "shape",
                    "title": "Possible pump unloading",
                    "suspected_problem": "Possible unloading",
                    "template_ids": ["sf_unloading"],
                    "card_ids": [14],
                    "reasoning": "Reduced separation.",
                    "supporting_features": [],
                    "contradicting_features": [],
                    "alternatives": [],
                    **branch_match_evidence(),
                    "evidence_keys": ["surface.card.archive:14.area_in_lb"],
                }
            ],
            "limitations": [],
        }

    result = analyze(payload, reviewer)
    assert result["review_status"] == "complete"
    assert len(result["findings"]) == 1


def test_weak_slope_only_shape_is_still_shown_to_the_user():
    data = packet().model_dump()
    data["cards"][-1]["roles"] = ["baseline", "recent"]
    payload = AnalysisRequest(**data)

    def reviewer(_):
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "shape",
                    "title": "Undertravel slope",
                    "suspected_problem": "Possible undertravel",
                    "template_ids": ["sf_undertravel"],
                    "card_ids": [14],
                    "reasoning": "The card has a positive overall slope.",
                    "match_strength": "weak",
                    "matching_features": ["Positive overall slope."],
                }
            ]
        }

    result = analyze(payload, reviewer)
    assert result["review_status"] == "complete"
    assert result["findings"][0]["title"] == "Undertravel slope"
    assert result["diagnosis_status"] == "candidates"


def test_review_returns_at_most_two_diagnoses_with_single_paragraph_reasoning():
    candidates = [
        {
            "side": "surface",
            "kind": "trend",
            "scope": "historical",
            "title": f"Declining card metric {index}",
            "card_ids": [1, 7, 14],
            "reasoning": "Measured area is declining.\nThe change spans the archive window.",
        }
        for index in range(3)
    ]
    result = analyze(packet(), lambda _: {"findings": candidates})
    assert len(result["findings"]) == 2
    assert result["findings"][0]["scope"] == "historical"
    assert "\n" not in result["findings"][0]["reasoning"]


def test_review_preserves_overlong_reasoning():
    finding = {
        "side": "surface",
        "kind": "trend",
        "title": "Declining card area",
        "card_ids": [1, 7, 14],
        "reasoning": " ".join(["evidence"] * 101),
    }
    result = analyze(packet(), lambda _: {"findings": [finding]})
    assert len(result["findings"][0]["reasoning"].split()) == 101
    assert result["diagnosis_status"] == "candidates"


def test_review_sends_exactly_three_images_and_recent_card_does_not_weight_trends():
    from datetime import timedelta

    data = packet().model_dump()
    recent = data["cards"][-1].copy()
    recent.update(
        archive_id=None,
        reference="live:1",
        roles=["recent"],
        timestamp=recent["timestamp"] + timedelta(days=1),
        pumping_window={"start": 100, "end": 1000},
    )
    recent["surface"] = [p.model_dump() for p in rectangle(300)]
    data["cards"].append(recent)
    payload = AnalysisRequest(**data)
    received = []

    def reviewer(content):
        received.extend(content)
        return {"findings": [], "limitations": []}

    result = analyze(payload, reviewer)
    assert sum(item["type"] == "image_url" for item in received) == 3
    baseline = analyze(packet(), lambda _: {"findings": [], "limitations": []})
    assert result["measurements"]["surface"]["trends"] == baseline["measurements"]["surface"]["trends"]
    assert "live:1" in received[0]["text"]
    assert "pumping_evidence" in received[0]["text"]


def test_shape_findings_are_labeled_by_recent_or_historical_evidence():
    data = packet().model_dump()
    data["cards"][-1]["roles"] = ["baseline", "recent"]
    payload = AnalysisRequest(**data)

    def reviewer(_):
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "shape",
                    "title": "Possible pump unloading",
                    "suspected_problem": "Possible unloading",
                    "template_ids": ["sf_unloading"],
                    "card_ids": ["archive:14"],
                    "reasoning": "The recent loop is narrow.",
                    "supporting_features": [],
                    "contradicting_features": [],
                    "alternatives": [],
                    **branch_match_evidence(),
                    "evidence_keys": ["surface.card.archive:14.area_in_lb"],
                }
            ],
            "limitations": [],
        }

    assert analyze(payload, reviewer)["review_status"] == "complete"
    assert analyze(packet(), reviewer)["review_status"] == "complete"


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


def test_unusable_cards_report_coverage_and_do_not_call_llm(caplog):
    data = packet().model_dump()
    for card in data["cards"]:
        card["surface"] = []
        card["downhole"] = []
    result = analyze(AnalysisRequest(**data), lambda _: pytest.fail("LLM must not be called"))
    assert result["review_status"] == "unavailable"
    assert "LLM was not called" in result["review_error"]
    assert "surface: valid=0" in result["review_error"]
    assert "well_id=2" in caplog.text


def test_optional_response_fields_may_be_omitted():
    result = analyze(packet(), lambda _: {"findings": []})
    assert result["review_status"] == "complete"
    assert result["findings"] == []
    assert result["diagnosis_status"] == "healthy"


def test_current_shape_is_returned_even_without_a_recent_card_anchor():
    data = packet().model_dump()
    data["cards"][-1]["roles"] = ["baseline", "recent"]
    payload = AnalysisRequest(**data)
    received = []

    def reviewer(content):
        received.extend(content)
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "shape",
                    "title": "Possible pump unloading",
                    "suspected_problem": "Possible unloading",
                    "template_ids": ["sf_unloading"],
                    "card_ids": ["archive:14", "archive:1"],
                    "reasoning": "The recent card is narrower than the earlier comparison card.",
                    "supporting_features": [],
                    "contradicting_features": [],
                    "alternatives": [],
                    **branch_match_evidence(),
                    "evidence_keys": [
                        "surface.card.archive:14.area_in_lb",
                        "surface.card.archive:1.area_in_lb",
                    ],
                }
            ],
            "limitations": [],
        }

    result = analyze(payload, reviewer)
    assert result["review_status"] == "complete"
    assert result["findings"][0]["card_ids"] == ["archive:14", "archive:1"]
    assert "required_shape_card_ids" in received[0]["text"]
    # Historical-only shape claims are shown without code-side evidence rejection.
    without_recent = analyze(packet(), reviewer)
    assert without_recent["review_status"] == "complete"
    assert without_recent["findings"][0]["title"] == "Possible pump unloading"
    assert without_recent["diagnosis_status"] == "candidates"


def test_prompt_is_concise_and_allows_domain_reasoning():
    from umai.rod_pump_health.analysis import SYSTEM_PROMPT

    assert "Lufkin SAM1" in SYSTEM_PROMPT
    assert "without a VFD" in SYSTEM_PROMPT
    assert "recent surface/downhole cards" in SYSTEM_PROMPT
    assert "archived dynographs" in SYSTEM_PROMPT
    assert "historical trends" in SYSTEM_PROMPT
    assert "rod-pump knowledge" in SYSTEM_PROMPT
    assert "separate concise reasoning and actions paragraphs" in SYSTEM_PROMPT
    assert "setpoint changes in actions" in SYSTEM_PROMPT
    assert "numeric values only when data supports them" in SYSTEM_PROMPT
    assert "scope=last_eligible_card" in SYSTEM_PROMPT
    assert "scope=historical" in SYSTEM_PROMPT
    assert "PDF" not in SYSTEM_PROMPT
    assert "severity" not in SYSTEM_PROMPT
    assert "Confidence is low/medium/high" in SYSTEM_PROMPT
    assert "Size alone" in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) < 1600
    data = packet().model_dump()
    data["cards"][-1]["roles"] = ["baseline", "recent"]
    payload = AnalysisRequest(**data)

    def reviewer(_):
        return {
            "findings": [
                {
                    "side": "surface",
                    "kind": "shape",
                    "title": "Loading change",
                    "suspected_problem": "Possible drive or loading issue",
                    "template_ids": [],
                    "card_ids": ["archive:14"],
                    "reasoning": (
                        "Engineering knowledge suggests a loading change from the observed card."
                    ),
                    "supporting_features": ["Reduced load separation"],
                    "contradicting_features": [],
                    "alternatives": ["Measurement calibration"],
                    "match_strength": "weak",
                    "evidence_keys": ["surface.card.archive:14.area_in_lb"],
                }
            ],
            "limitations": [],
        }

    result = analyze(payload, reviewer)
    assert result["review_status"] == "complete"
    assert result["findings"][0]["title"] == "Loading change"
    assert result["diagnosis_status"] == "candidates"

    def unsupported(_):
        result = reviewer(_)
        result["findings"][0]["supporting_features"] = []
        return result

    assert analyze(payload, unsupported)["diagnosis_status"] == "candidates"


def test_minimal_findings_and_non_object_items_are_still_displayable():
    result = analyze(
        packet(),
        lambda _: {
            "findings": [
                "not a finding",
                {
                    "side": "surface",
                    "title": "Loading observation",
                    "reasoning": "Area changed by 10 percent.",
                },
            ]
        },
    )
    assert result["review_status"] == "complete"
    assert result["findings"][0]["title"] == "not a finding"
    assert result["findings"][1]["title"] == "Loading observation"
    assert result["diagnosis_status"] == "candidates"
    assert result["status"] == "partial"


def test_flat_load_and_open_curves_are_reviewable_with_quality_warnings():
    points = [Point(position=x, load=1000) for x in [0, 25, 50, 75, 100]]
    measured = features(points)
    assert measured["status"] == "questionable"
    assert measured["load_range_lb"] == 0
    data = packet().model_dump()
    for card in data["cards"]:
        card["surface"] = [p.model_dump() for p in points]
    calls = []
    result = analyze(AnalysisRequest(**data), lambda content: calls.append(content) or {"findings": []})
    assert calls
    assert result["review_status"] == "complete"
    assert any("questionable" in note for note in result["limitations"])


def test_short_history_produces_provisional_trends_without_interpolation():
    rows = [
        {"timestamp": c.timestamp.isoformat(), "features": features(c.surface)}
        for c in packet().cards[:3]
    ]
    result = trends(rows)
    assert result["status"] == "available"
    assert result["preliminary"] is True
    assert result["metrics"]["area_in_lb"]["absolute_change"] < 0


def test_response_token_usage_normalizes_langchain_and_openai_metadata():
    from types import SimpleNamespace

    from umai.rod_pump_health.analysis import _response_token_usage

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


def test_response_token_usage_normalizes_langchain_and_openai_metadata():
    from types import SimpleNamespace

    from umai.rod_pump_health.analysis import _response_token_usage

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
