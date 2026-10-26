"""Numerical matcher regressions independent of LLMs and external services."""
import math
from datetime import datetime, timezone

import pytest

from umai.rod_pump_health.analysis import AnalysisRequest
from umai.rod_pump_health.matching import MIN_SCORE, analyze_local, match_score, normalized_outline
from umai.rod_pump_health.templates import TEMPLATES


def packet(points, downhole=None, load_references=None):
    return AnalysisRequest(site_id=1, well_id=1,
                           calculation_snapshot={"load_references": load_references or {}}, cards=[{
        "archive_id": 1, "timestamp": datetime.now(timezone.utc), "roles": ["recent"],
        "surface": [{"position": x, "load": y} for x, y in points],
        "downhole": [{"position": x, "load": y} for x, y in (downhole or [])],
    }])


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda row: row["id"])
def test_exact_templates_match_despite_units_start_and_traversal(template):
    points = template["points"]
    reference = {"source": "expected_full_load", "load_range_lb": 1}
    assert match_score(points, points, load_reference=reference) == 100
    # Change starting sample and acquisition direction without rotating the physical axes.
    unique = points[:-1] if points[0] == points[-1] else points
    shifted = unique[2:] + unique[:2]
    actual = [(x * 120 + 50, y * 3000 + 8000) for x, y in reversed(shifted)]
    reference["load_range_lb"] = 3000
    assert match_score(actual, points, load_reference=reference) >= 97


def test_different_outline_is_rejected():
    rectangle = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
    triangle = [[0, 0], [0.5, 1], [1, 0], [0, 0]]
    assert match_score(rectangle, triangle) < MIN_SCORE


def test_at_most_two_matches_across_both_sides():
    surface = next(row for row in TEMPLATES if row["side"] == "surface")
    downhole = next(row for row in TEMPLATES if row["side"] == "downhole")
    result = analyze_local(packet(surface["points"], downhole["points"]))
    assert 1 <= len(result["matches"]) <= 2
    assert all(match["score"] >= MIN_SCORE for match in result["matches"])
    assert result["matches"][0]["score"] >= result["matches"][-1]["score"]
    assert result["token_usage"] is None


def test_unrelated_crossed_shape_has_no_template_match():
    result = analyze_local(packet([[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]))
    assert result["matches"] == []
    assert result["diagnosis_status"] == "no_match"


def test_all_low_scores_return_no_match(monkeypatch):
    monkeypatch.setattr("umai.rod_pump_health.matching.match_score", lambda *args, **kwargs: MIN_SCORE - 0.1)
    result = analyze_local(packet(TEMPLATES[0]["points"]))
    assert result["matches"] == []
    assert result["diagnosis_status"] == "no_match"


def test_invalid_curve_has_no_match_and_is_unavailable():
    result = analyze_local(packet([[0, 1], [1, 1], [2, 1], [3, 1]]))
    assert result["matches"] == []
    assert result["unavailable_sides"] == ["surface", "downhole"]
    assert match_score([[0, math.nan]] * 4, TEMPLATES[0]["points"]) == 0


def test_broad_card_cannot_match_gas_lock_even_with_a_reference():
    broad = [[0, 0], [0.15, 1], [1, 1], [1, 0], [0, 0]]
    gas_lock = next(row for row in TEMPLATES if row["id"] == "dh_gas_lock")
    assert match_score(broad, gas_lock["points"]) == 0
    reference = {"source": "expected_full_load", "load_range_lb": 1}
    assert match_score(broad, gas_lock["points"], load_reference=reference) < MIN_SCORE
    result = analyze_local(packet(broad, broad))
    assert "dh_gas_lock" not in [row["template_id"] for row in result["matches"]]


@pytest.mark.parametrize("template_id", ["dh_gas_lock", "sf_unloading"])
def test_narrow_shape_only_matches_with_independent_load_reference(template_id):
    template = next(row for row in TEMPLATES if row["id"] == template_id)
    points = [(x * 100, y * 8000 + 2000) for x, y in template["points"]]
    assert match_score(points, template["points"]) == 0
    reference = {"source": "validated_normal_card", "load_range_lb": 8000}
    assert match_score(points, template["points"], load_reference=reference) == 100
    stretched = [(x, y * 8) for x, y in points]
    assert match_score(stretched, template["points"], load_reference=reference) < MIN_SCORE


@pytest.mark.parametrize("value", [None, {}, {"source": "current_card", "load_range_lb": 8000},
                                  {"source": "expected_full_load", "load_range_lb": 0},
                                  {"source": "expected_full_load", "load_range_lb": math.nan},
                                  {"source": "expected_full_load", "load_range_lb": math.inf},
                                  {"source": "expected_full_load", "load_range_lb": True}])
def test_invalid_or_self_referencing_load_scale_cannot_enable_low_load_match(value):
    template = next(row for row in TEMPLATES if row["id"] == "dh_gas_lock")
    assert match_score(template["points"], template["points"], load_reference=value) == 0


def test_supported_load_reference_is_used_by_analysis():
    template = next(row for row in TEMPLATES if row["id"] == "dh_gas_lock")
    reference = {"source": "validated_normal_card", "load_range_lb": 1}
    result = analyze_local(packet([], template["points"], {"downhole": reference}))
    assert any(row["template_id"] == "dh_gas_lock" for row in result["matches"])


def test_template_load_separation_is_not_stretched():
    template = next(row for row in TEMPLATES if row["id"] == "dh_gas_lock")
    outline = normalized_outline(template["points"], template=True)
    loads = [y for _, y in outline]
    assert max(loads) - min(loads) == pytest.approx(0.12)


def mock_scores(monkeypatch, scores):
    """Control the full well ranking independently of geometrical similarity tests."""
    def assessment(actual, template, load_reference=None):
        return {"score": scores.get(template["id"], 0), "structure_passed": True, "structural_checks": []}
    monkeypatch.setattr("umai.rod_pump_health.matching.evaluate_outline", assessment)


def test_highest_healthy_match_replaces_all_other_well_matches(monkeypatch):
    mock_scores(monkeypatch, {"dh_full": 97, "sf_undertravel": 96, "dh_gas": 95})
    points = TEMPLATES[0]["points"]
    result = analyze_local(packet(points, points))
    assert result["diagnosis_status"] == "healthy"
    assert [row["template_id"] for row in result["matches"]] == ["dh_full"]


def test_nonfirst_healthy_is_removed_before_selecting_two_problem_matches(monkeypatch):
    mock_scores(monkeypatch, {"sf_undertravel": 98, "dh_full": 97, "dh_gas": 96, "dh_tubing": 95})
    points = TEMPLATES[0]["points"]
    result = analyze_local(packet(points, points))
    assert result["diagnosis_status"] == "matches"
    assert [row["template_id"] for row in result["matches"]] == ["sf_undertravel", "dh_gas"]


def test_tied_problem_score_prevents_healthy_verdict(monkeypatch):
    mock_scores(monkeypatch, {"dh_full": 95, "dh_gas": 95})
    points = TEMPLATES[0]["points"]
    result = analyze_local(packet([], points))
    assert result["diagnosis_status"] == "matches"
    assert [row["template_id"] for row in result["matches"]] == ["dh_gas"]


def test_low_healthy_score_does_not_override_no_match(monkeypatch):
    mock_scores(monkeypatch, {"dh_full": MIN_SCORE - 0.1})
    result = analyze_local(packet([], TEMPLATES[0]["points"]))
    assert result["diagnosis_status"] == "no_match"
    assert result["matches"] == []


def test_real_healthy_outline_is_the_winning_match():
    healthy = next(row for row in TEMPLATES if row["id"] == "dh_full")
    points = [(x * 90, y * 6000 - 800) for x, y in healthy["points"]]
    result = analyze_local(packet([], points))
    assert result["diagnosis_status"] == "healthy"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["template_id"] == "dh_full"
    assert result["matches"][0]["score"] == 100
