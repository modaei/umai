"""Field-card and counterexample regressions for template-specific topology, without LLMs."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from umai.rod_pump_health.analysis import AnalysisRequest
from umai.rod_pump_health.branch_matching import prepare_branches, compare_template_branches
from umai.rod_pump_health.matching import analyze_local, evaluate_outline
from umai.rod_pump_health.shape_structure import outline_structure, structural_checks
from umai.rod_pump_health.templates import TEMPLATES


def template(template_id):
    return next(row for row in TEMPLATES if row["id"] == template_id)


def packet(points, *, reverse_time=False):
    raw = points if reverse_time else list(reversed(points))
    return AnalysisRequest(site_id=1, well_id=1, cards=[{
        "archive_id": 1, "timestamp": datetime.now(timezone.utc), "roles": ["recent"],
        "point_order": "sam1_reverse_time", "surface": [],
        "downhole": [{"position": x * 100, "load": y * 8000 + 2000} for x, y in raw],
    }])


def test_saved_4314_card_matches_fluid_pound_in_both_methods():
    fixture = json.loads((Path(__file__).parent / 'fixtures/rod_pump_4314_downhole.json').read_text())
    result = analyze_local(packet(fixture["points"], reverse_time=True))
    for method in (result,):
        assert method["diagnosis_status"] == "matches"
        assert [row["template_id"] for row in method["matches"]] == ["dh_fluid_pound"]
        match = method["matches"][0]
        assert match["score"] >= 90
        assert all(row["passed"] for row in match["structural_checks"])
        handle = next(row for row in match["structural_checks"] if "handle present" in row["feature"])
        assert 0.23 <= handle["actual"] <= 0.27
        assert match["family_parameters"]["release_top"] >= 0.73
        assert match["matched_template_points"] != template("dh_fluid_pound")["points"]
    rejected = next(row for row in result["structural_rejections"] if row["title"] == "Tubing-movement pattern")
    assert "No fluid-pound handle" in rejected["failed_checks"]


@pytest.mark.parametrize("length", [0.12, 0.25, 0.40, 0.55])
@pytest.mark.parametrize("pickup", [0, 0.10, 0.22])
def test_fluid_pound_family_recognizes_variable_handle_and_pickup(length, pickup):
    release = 1 - length
    points = [[0, 0], [pickup, 1], [1, 1], [release, 1], [release - 0.03, 0], [0, 0]]
    result = analyze_local(packet(points))
    for method in (result,):
        assert any(row["template_id"] == "dh_fluid_pound" for row in method["matches"])
        assert method["diagnosis_status"] == "matches"


@pytest.mark.parametrize("template_id", ["dh_full", "dh_gas", "dh_tubing", "dh_standing_valve", "dh_tagging"])
def test_other_structures_cannot_be_accepted_as_fluid_pound(template_id):
    points = template(template_id)["points"]
    outline = evaluate_outline(points, template("dh_fluid_pound"))
    branches, _ = prepare_branches(list(reversed(points)), "sam1_reverse_time")
    directed = compare_template_branches(branches, template("dh_fluid_pound"))
    assert not outline["structure_passed"]
    assert not directed["features_passed"]
    assert any(not check["passed"] and "handle" in check["feature"].lower()
               for check in outline["structural_checks"])


def test_extension_at_lower_load_is_not_an_upper_right_handle():
    points = [[0, 0], [0, 1], [0.6, 1], [0.62, 0], [1, 0], [0, 0]]
    result = evaluate_outline(points, template("dh_fluid_pound"))
    assert not result["structure_passed"]
    assert outline_structure(points)["handle_length"] < 0.08


def test_handle_does_not_override_a_gradual_gas_like_release():
    points = [[0, 0], [0, 1], [1, 1], [0.80, 1], [0.70, 0.8], [0.60, 0.6], [0.50, 0.3], [0.4, 0], [0, 0]]
    result = evaluate_outline(points, template("dh_fluid_pound"))
    assert not result["structure_passed"]
    assert any(check["feature"] == "Abrupt load release" and not check["passed"]
               for check in result["structural_checks"])


@pytest.mark.parametrize("template_id", [row["id"] for row in TEMPLATES])
def test_each_reference_passes_its_own_defining_structures(template_id):
    row = template(template_id)
    reference = {"source": "validated_normal_card", "load_range_lb": 0.7 if template_id == "sf_friction" else 1}
    result = evaluate_outline(row["points"], row, reference)
    assert result["structure_passed"], result["structural_checks"]
    assert result["score"] == 100


@pytest.mark.parametrize("template_id", ["dh_gas_lock", "sf_unloading", "sf_friction"])
def test_load_dependent_shapes_cannot_match_without_independent_reference(template_id):
    row = template(template_id)
    result = evaluate_outline(row["points"], row)
    assert not result["structure_passed"]
    check = next(row for row in result["structural_checks"] if "independent reference" in row["feature"])
    assert check["actual"] is None
    assert not check["passed"]


def test_friction_requires_actual_increase_over_reference():
    row = template("sf_friction")
    result = evaluate_outline(row["points"], row, {"source": "validated_normal_card", "load_range_lb": 1})
    assert not result["structure_passed"]
    increased = evaluate_outline(row["points"], row, {"source": "validated_normal_card", "load_range_lb": 0.7})
    assert increased["structure_passed"]


def test_outline_handle_is_independent_of_traversal_but_not_axis_reflection():
    points = template("dh_fluid_pound")["points"]
    assert outline_structure(points)["handle_length"] == outline_structure(list(reversed(points)))["handle_length"]
    reflected = [(x, 1 - y) for x, y in points]
    assert not all(row["passed"] for row in structural_checks("dh_fluid_pound", outline_structure(reflected)))


@pytest.mark.parametrize("target,other,failed_label", [
    ("dh_full", "dh_fluid_pound", "No substantial upper-right handle"),
    ("dh_full", "dh_tagging", "No end-of-stroke excursion"),
    ("dh_gas", "dh_full", "Progressive load release"),
    ("dh_traveling_valve", "dh_full", "Sustained upper-branch load decline"),
    ("dh_standing_valve", "dh_full", "Gradual load pickup"),
    ("dh_tubing", "dh_standing_valve", "Leaning release edge"),
    ("sf_undertravel", "sf_overtravel", "Overall loop slope"),
    ("sf_overtravel", "sf_undertravel", "Overall loop slope"),
])
def test_defining_checks_reject_confusable_patterns(target, other, failed_label):
    result = evaluate_outline(template(other)["points"], template(target))
    assert not result["structure_passed"]
    assert any(check["feature"] == failed_label and not check["passed"] for check in result["structural_checks"])


def test_midstroke_excursion_is_not_an_end_of_stroke_tag():
    points = [[0, 0], [0, 1], [1, 1], [1, 0], [0.6, 0], [0.5, -0.25], [0.4, 0], [0, 0]]
    result = evaluate_outline(points, template("dh_tagging"))
    assert not result["structure_passed"]
    assert any(check["feature"] == "Excursion at stroke end" and not check["passed"]
               for check in result["structural_checks"])


def test_saved_3051_fluid_pound_plateaus_exclude_pickup_and_release_ramps():
    """The real 3051 handle passes without counting transition ramps as plateau variation."""
    fixture = json.loads((Path(__file__).parent / 'fixtures/rod_pump_3051_downhole.json').read_text())
    result = analyze_local(packet(fixture["points"], reverse_time=True))
    assert result["diagnosis_status"] == "matches"
    match = next(row for row in result["matches"] if row["template_id"] == "dh_fluid_pound")
    assert match["score"] == 93.2
    assert all(row["passed"] for row in match["structural_checks"])
    assert "branch_comparison" not in result


@pytest.mark.parametrize('well,score', [('4553', 81.0), ('4593', 85.3)])
def test_saved_healthy_cards_tolerate_rounded_pickup_and_gradual_load_rise(well, score):
    """Real healthy field cards must pass without changing their geometric similarity score."""
    fixture = json.loads((Path(__file__).parent / f'fixtures/rod_pump_{well}_downhole.json').read_text())
    result = analyze_local(packet(fixture['points'], reverse_time=True))
    assert result['diagnosis_status'] == 'healthy'
    assert [row['template_id'] for row in result['matches']] == ['dh_full']
    assert result['matches'][0]['score'] == score


@pytest.mark.parametrize('points', [
    [[0, 0], [.06, 1], [.86, 1], [.9, 1.16], [.94, 1], [1, 0], [0, 0]],
    [[0, 0], [.06, 1], [.94, 1], [1, 0], [.12, 0], [.06, -.16], [0, 0]],
    [[0, 0], [.06, 1], [.86, 1], [.9, 1.16], [.94, 1], [1, 0],
     [.12, 0], [.06, -.16], [0, 0]],
])
def test_localized_spikes_cannot_be_healthy_even_with_multiple_distant_excursions(points):
    """A localized spike at either end or on both branches must still veto a healthy verdict."""
    result = evaluate_outline(points, template('dh_full'))
    assert not result['structure_passed']
    assert any(row['feature'] == 'No end-of-stroke excursion' and not row['passed']
               for row in result['structural_checks'])


def test_large_broad_load_rise_is_not_excused_as_a_healthy_gradual_rise():
    points = [[0, 0], [.06, 1], [.6, 1], [.92, 1.4], [1, 0], [0, 0]]
    result = evaluate_outline(points, template('dh_full'))
    assert not result['structure_passed']
    assert any(row['feature'] == 'Bounded gradual end-load change' and not row['passed']
               for row in result['structural_checks'])


@pytest.mark.parametrize('template_id', [row['id'] for row in TEMPLATES
                                        if row['side'] == 'downhole' and row['id'] != 'dh_full'])
def test_downhole_fault_references_are_not_reclassified_as_healthy(template_id):
    """The healthy-first selection rule must never suppress canonical fault references."""
    result = evaluate_outline(template(template_id)['points'], template('dh_full'),
                              {'source': 'expected_full_load', 'load_range_lb': 1})
    assert not result['structure_passed']


def test_late_pickup_is_not_healthy_even_when_transition_itself_is_sharp():
    points = [[0, 0], [.2, 0], [.31, 1], [.94, 1], [1, 0], [0, 0]]
    result = evaluate_outline(points, template('dh_full'))
    assert not result['structure_passed']
    assert any(row['feature'] == 'Early upper plateau' and not row['passed']
               for row in result['structural_checks'])



def test_saved_4155_matches_tubing_movement_and_cannot_be_healthy():
    """Real paired positive edge lean must veto healthy before the healthy-first selection rule."""
    fixture = json.loads((Path(__file__).parent / 'fixtures/rod_pump_4155_downhole.json').read_text())
    result = analyze_local(packet(fixture['points'], reverse_time=True))
    assert result['diagnosis_status'] == 'matches'
    assert result['matches'][0]['template_id'] == 'dh_tubing'
    assert result['matches'][0]['score'] == 84.4
    assert not any(row['template_id'] == 'dh_full' for row in result['matches'])
    healthy = evaluate_outline(fixture['points'], template('dh_full'))
    assert any(row['feature'] == 'No parallel leaning edges' and not row['passed']
               for row in healthy['structural_checks'])


@pytest.mark.parametrize('lean', [0.065, 0.075, 0.10, 0.20])
@pytest.mark.parametrize('reverse', [False, True])
def test_moderate_parallel_edge_lean_is_tubing_movement_in_either_traversal(lean, reverse):
    points = [[0, 0], [lean * 2, 1], [1, 1], [1 - lean * 2, 0], [0, 0]]
    if reverse:
        points.reverse()
    healthy = evaluate_outline(points, template('dh_full'))
    tubing = evaluate_outline(points, template('dh_tubing'))
    assert not healthy['structure_passed']
    assert tubing['structure_passed']


def test_opposing_edge_lean_is_not_tubing_movement():
    points = [[0, 0], [.15, 1], [.85, 1], [1, 0], [0, 0]]
    result = evaluate_outline(points, template('dh_tubing'))
    assert not result['structure_passed']
    assert any(row['feature'] == 'Leaning release edge' and not row['passed']
               for row in result['structural_checks'])


def test_saved_5118_matches_tagging_using_separate_body_and_spike_scores():
    fixture = json.loads((Path(__file__).parent / 'fixtures/rod_pump_5118_downhole.json').read_text())
    result = analyze_local(packet(fixture['points'], reverse_time=True))
    assert result['diagnosis_status'] == 'matches'
    tagging = next(row for row in result['matches'] if row['template_id'] == 'dh_tagging')
    assert tagging['score'] == 84.1
    assert tagging['body_score'] == 81.0
    assert tagging['spike_score'] == 89.9
    assert all(row['passed'] for row in tagging['structural_checks'])
    assert not any(row['template_id'] == 'dh_full' for row in result['matches'])


@pytest.mark.parametrize('flip_x,flip_y', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('reverse', [False, True])
def test_tagging_family_preserves_end_location_and_load_polarity(flip_x, flip_y, reverse):
    points = [[1-x if flip_x else x, 1-y if flip_y else y] for x, y in template('dh_tagging')['points']]
    if reverse:
        points.reverse()
    result = evaluate_outline(points, template('dh_tagging'))
    assert result['structure_passed']
    assert result['score'] == 100
    assert result['body_score'] == 100
    assert result['spike_score'] == 100


@pytest.mark.parametrize('well', ['4553', '4593', '4155', '3051', '4314'])
def test_region_scoring_does_not_invent_tagging_on_other_field_cards(well):
    fixture = json.loads((Path(__file__).parent / f'fixtures/rod_pump_{well}_downhole.json').read_text())
    result = evaluate_outline(fixture['points'], template('dh_tagging'))
    assert not result['structure_passed']



@pytest.mark.parametrize('well,template_id,score', [
    ('6288', 'dh_tubing', 78.6), ('5797', 'dh_tubing', 78.4), ('6287', 'dh_fluid_pound', 90.0),
])
def test_asymmetric_tubing_edges_are_distinguished_from_fluid_pound_handle(well, template_id, score):
    fixture = json.loads((Path(__file__).parent / f'fixtures/rod_pump_{well}_downhole.json').read_text())
    result = analyze_local(packet(fixture['points'], reverse_time=True))
    assert result['diagnosis_status'] == 'matches'
    assert [row['template_id'] for row in result['matches']] == [template_id]
    assert result['matches'][0]['score'] == score
    other = 'dh_fluid_pound' if template_id == 'dh_tubing' else 'dh_tubing'
    assert not evaluate_outline(fixture['points'], template(other))['structure_passed']


def test_fluid_pound_load_family_preserves_bounded_body_and_handle_levels():
    fixture = json.loads((Path(__file__).parent / 'fixtures/rod_pump_6287_downhole.json').read_text())
    result = evaluate_outline(fixture['points'], template('dh_fluid_pound'))
    parameters = result['family_parameters']
    assert 0 <= parameters['body_lower'] <= .15
    assert .80 <= parameters['body_upper'] <= 1
    assert parameters['body_upper'] <= parameters['handle_load'] <= 1
    assert parameters['handle_load'] - parameters['body_upper'] <= .15
    assert result['handle_score'] >= 75


@pytest.mark.parametrize('pickup,release', [(.08, .045), (.115, .043), (.065, .10)])
def test_asymmetric_signed_edges_pass_tubing_and_veto_healthy(pickup, release):
    points = [[0, 0], [2*pickup, 1], [1, 1], [1-2*release, 0], [0, 0]]
    assert evaluate_outline(points, template('dh_tubing'))['structure_passed']
    healthy = evaluate_outline(points, template('dh_full'))
    assert not healthy['structure_passed']
    assert any(row['feature'] == 'No parallel leaning edges' and not row['passed']
               for row in healthy['structural_checks'])


def test_one_sided_pickup_ramp_does_not_qualify_as_tubing():
    points = [[0, 0], [.22, 1], [.96, 1], [1, 0], [0, 0]]
    assert not evaluate_outline(points, template('dh_tubing'))['structure_passed']


def test_large_body_load_adjustment_is_not_fitted_by_fluid_pound_family():
    from umai.rod_pump_health.shape_structure import template_variants
    points = [[0, 0], [.05, .55], [.65, .55], [1, 1], [.65, .55], [.63, 0], [0, 0]]
    variants = template_variants(template('dh_fluid_pound'), outline_structure(points))
    assert not any('body_upper' in parameters for _, parameters in variants)
