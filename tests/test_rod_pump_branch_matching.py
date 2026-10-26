"""Direction, local alignment, defining features, ambiguity, and zero-LLM regressions."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from umai.rod_pump_health.analysis import AnalysisRequest
from umai.rod_pump_health.branch_matching import (
    analyze_branches, compare_branches, constrained_error, prepare_branches,
)
from umai.rod_pump_health.matching import VERSION, analyze_local
from umai.rod_pump_health.templates import TEMPLATES


def template(template_id):
    return next(row for row in TEMPLATES if row["id"] == template_id)


def packet(template_id="dh_full", *, order="sam1_reverse_time", points=None, reference=False):
    row = template(template_id)
    forward = points or row["points"]
    raw = list(reversed(forward)) if order == "sam1_reverse_time" else forward
    card = {
        "archive_id": 1, "timestamp": datetime.now(timezone.utc), "roles": ["recent"],
        "point_order": order, "surface": [], "downhole": [],
    }
    card[row["side"]] = [{"position": 50 + x * 100, "load": 2000 + y * 8000} for x, y in raw]
    snapshot = {"load_references": {row["side"]: {
        "source": "expected_full_load", "load_range_lb": 6000 if template_id == "sf_friction" else 8000,
    }}} if reference else {}
    return AnalysisRequest(site_id=1, well_id=1, cards=[card], calculation_snapshot=snapshot)


@pytest.mark.parametrize("template_id", [row["id"] for row in TEMPLATES])
def test_known_direction_exact_templates_pass_defining_features(template_id):
    result = analyze_branches(packet(template_id, reference=True))
    assert result["diagnosis_status"] in ("healthy", "matches")
    assert result["matches"][0]["template_id"] == template_id
    assert result["matches"][0]["score"] == 100
    assert all(check["passed"] for check in result["matches"][0]["feature_checks"])


def test_unknown_direction_is_ambiguous_instead_of_silently_reversed():
    result = analyze_branches(packet(order="unknown"))
    assert result["diagnosis_status"] == "ambiguous"
    assert any("direction is unverified" in reason for reason in result["reasons"])
    assert all(row["features_passed"] for row in result["matches"])
    assert not any("Defining feature checks" in reason for reason in result["reasons"])


def test_reversing_acquisition_does_not_swap_upstroke_and_downstroke_to_force_a_match():
    forward = template("dh_full")["points"]
    correct, _ = prepare_branches(list(reversed(forward)), "sam1_reverse_time")
    wrong, _ = prepare_branches(forward, "sam1_reverse_time")
    reference, _ = prepare_branches(forward, template=True)
    assert compare_branches(correct, reference)["score"] == 100
    result = compare_branches(wrong, reference)
    assert result["score"] < 75
    assert not result["features_passed"]


def test_open_card_is_ambiguous_and_cannot_receive_a_healthy_verdict():
    points = [[0, 0], [0.06, 1], [0.94, 1], [1, 0], [0.2, 0]]
    result = analyze_branches(packet(points=points))
    assert result["diagnosis_status"] == "ambiguous"
    assert any("not sufficiently closed" in reason for reason in result["reasons"])


def test_feature_checks_reject_healthy_shape_as_a_standing_valve_leak():
    actual, _ = prepare_branches(template("dh_full")["points"], template=True)
    reference, _ = prepare_branches(template("dh_standing_valve")["points"], template=True)
    result = compare_branches(actual, reference)
    assert not result["features_passed"]
    assert any(not check["passed"] and check["feature"] == "Upstroke slope"
               for check in result["feature_checks"])


def test_small_transition_shift_uses_bounded_dtw_without_hiding_feature_differences():
    points = [[0, 0], [0, 1], [1, 1], [0.65, 1], [0.63, 0], [0, 0]]
    result = analyze_branches(packet("dh_fluid_pound", points=points))
    match = next(row for row in result["matches"] if row["template_id"] == "dh_fluid_pound")
    assert match["family_parameters"]["release_top"] > 0.62
    actual, _ = prepare_branches(list(reversed(points)), "sam1_reverse_time")
    reference, _ = prepare_branches(template("dh_fluid_pound")["points"], template=True)
    assert compare_branches(actual, reference)["alignment"] == "constrained_dtw"
    assert match["features_passed"]
    assert match["score"] >= 75


def test_dtw_cannot_align_a_large_displacement_shift():
    left = [(i / 47, 1 if i < 12 else 0) for i in range(48)]
    right = [(i / 47, 1 if i < 36 else 0) for i in range(48)]
    assert constrained_error(left, right) > 0.1


def test_close_competing_scores_are_ambiguous(monkeypatch):
    import umai.rod_pump_health.branch_matching as module
    original = module.compare_template_branches
    def close_scores(actual, template, load_reference=None):
        result = original(actual, template, load_reference)
        return {**result, "score": 95, "features_passed": True, "structure_passed": True,
                "structural_checks": [{**check, "passed": True} for check in result["structural_checks"]]}
    monkeypatch.setattr(module, "compare_template_branches", close_scores)
    result = analyze_branches(packet())
    assert result["diagnosis_status"] == "ambiguous"
    assert any("within 5 score points" in reason for reason in result["reasons"])
    assert len(result["matches"]) <= 2


def test_low_load_templates_require_independent_reference_in_both_approaches():
    result = analyze_local(packet("dh_gas_lock"))
    for method in (result, analyze_branches(packet("dh_gas_lock"))):
        assert not any(row["template_id"] == "dh_gas_lock" for row in method["matches"])


def test_outline_analysis_does_not_run_or_return_retired_branch_comparison():
    result = analyze_local(packet())
    assert result["diagnosis_status"] == "healthy"
    assert "branch_comparison" not in result
    assert result["matches"][0]["card_reference"] == "archive:1"
    assert result["token_usage"] is None


@pytest.mark.parametrize("mode", ["fresh", "cached", "failed"])
def test_local_comparison_never_invokes_optional_llms_even_if_legacy_flag_is_enabled(monkeypatch, mode):
    import importlib
    from umai.rod_pump_health.reader import WellAnalysisRequest
    module = importlib.import_module("umai.api.app")
    def analyze(*args):
        if mode == "failed":
            raise RuntimeError("Card unavailable")
        return {"well_id": 1, "review_status": "complete", "analyzer_version": VERSION,
                "unchanged": mode == "cached"}
    monkeypatch.setattr(module, "EXTENDED_HEALTH_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(module, "analyzer_from_settings", lambda _: SimpleNamespace(
        _well=lambda _: {"id": 1, "site_id": 1}, analyze=analyze,
    ))
    with patch.object(module, "parameter_analyzer_from_settings") as parameters:
        with patch.object(module, "conclusion_analyzer_from_settings") as conclusion:
            app = module.create_app()
            route = next(row for row in app.routes if row.path == "/rod-pump-health-analysis")
            result = route.endpoint(WellAnalysisRequest(well_id=1))
    assert result["parameter_trends"]["status"] == "disabled"
    assert result["conclusion"]["status"] == "disabled"
    parameters.assert_not_called()
    conclusion.assert_not_called()


def test_handleless_card_is_rejected_even_as_an_ambiguous_candidate():
    """Approximate screenshot geometry must not bypass the mandatory handle gate."""
    points = [(0, .1), (.03, .45), (.10, .82), (.18, 1), (.5, .98), (.88, .99),
              (.90, .985), (.88, .97), (.94, .945), (.98, .94), (1, .90), (.96, .75),
              (.95, .54), (.98, .4), (.991, .13), (.97, .01), (.90, .02),
              (.5, .03), (.08, 0), (.04, .1), (0, .1)]
    result = analyze_branches(packet(points=points))
    assert not any(row["template_id"] == "dh_fluid_pound" for row in result["matches"])
    rejection = next(row for row in result["structural_rejections"]
                     if row["title"] == template("dh_fluid_pound")["title"])
    assert any("handle" in check.lower() for check in rejection["failed_checks"])


def test_rejected_high_score_cannot_make_healthy_match_ambiguous(monkeypatch):
    """Structurally invalid rivals cannot undermine an eligible healthy winner."""
    import umai.rod_pump_health.branch_matching as module
    original = module.compare_template_branches

    def high_rejected_score(actual, template, load_reference=None):
        result = original(actual, template, load_reference)
        if template["id"] == "dh_fluid_pound":
            return {**result, "score": 99, "structure_passed": False, "features_passed": False}
        return result

    monkeypatch.setattr(module, "compare_template_branches", high_rejected_score)
    result = analyze_branches(packet())
    assert result["diagnosis_status"] == "healthy"
    assert [row["template_id"] for row in result["matches"]] == ["dh_full"]
