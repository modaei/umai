import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sqlalchemy import text

from umai.config.settings import DEFAULT_CLASSIFIER_MODEL
from umai.rod_pump_health.classifier import FORMAT, card_features, load_classifier

from test_rod_pump_health_reader import analyzer, archived_reference  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CARDS = json.loads((ROOT / "tests/fixtures/rod_pump_classifier_cards.json").read_text())


def test_features_match_the_training_script():
    # Expected vectors were produced by the original card_matcher.resample_card.
    for card in CARDS:
        positions = [point["position"] for point in card["points"]]
        loads = [point["load"] for point in card["points"]]
        np.testing.assert_allclose(card_features(positions, loads), card["features"], rtol=1e-9)


def test_committed_model_reproduces_original_predictions():
    classifier = load_classifier(str(ROOT / DEFAULT_CLASSIFIER_MODEL))
    assert classifier is not None
    for card in CARDS:
        result = classifier.classify(card["points"])
        assert result["health"] == card["health"]
        assert result["label"] == card["label"]
        assert result["label_confidence"] == pytest.approx(card["label_confidence"], abs=1e-3)
        assert len(result["model_version"]) == 12


def test_untrustworthy_model_files_disable_the_classifier(tmp_path):
    assert load_classifier("") is None
    assert load_classifier(str(tmp_path / "missing.joblib")) is None
    other_version = tmp_path / "other.joblib"
    joblib.dump({"format": FORMAT, "sklearn_version": "0.0.1", "models": {}}, other_version)
    assert load_classifier(str(other_version)) is None
    unknown = tmp_path / "unknown.joblib"
    joblib.dump({"models": {}}, unknown)
    assert load_classifier(str(unknown)) is None


class FakeClassifier:
    def __init__(self, label="healthy", confidence=0.99, version="v1"):
        self.result = {
            "model_version": version,
            "health": "Healthy" if label == "healthy" else "Problem",
            "health_confidence": confidence,
            "label": label,
            "label_confidence": confidence,
        }

    def classify(self, points):
        return dict(self.result)


@pytest.fixture
def routed(analyzer):  # noqa: F811
    """Analyzer with a classifier, separate routine reviewer, and no collapsed-card cut-off
    (the fixture card's downhole loads span only 1,000 lb)."""
    service, calls = analyzer
    calls["routine"] = 0
    service.classifier = FakeClassifier()
    review = service.reviewer

    def routine(content):
        calls["routine"] += 1
        return review(content)

    service.routine_reviewer = routine
    service.routing["min_downhole_load_range_lb"] = 0
    return service, calls


def save_review(service, result):
    """Persist a result the way UMetriqs stores it, under a different signature."""
    findings = {
        key: result.get(key)
        for key in ("review_status", "health_status", "summary", "findings", "llm_review")
    }
    with service.engine.begin() as connection:
        connection.execute(text("DELETE FROM rod_pump_health_analyses WHERE well_id=23"))
        connection.execute(
            text("INSERT INTO rod_pump_health_analyses VALUES(23,'older',:findings)"),
            {"findings": json.dumps(findings)},
        )


def test_first_review_of_a_healthy_card_uses_the_routine_effort(routed):
    service, calls = routed
    result = service.analyze(23, archived_reference())
    assert (calls["review"], calls["routine"]) == (1, 1)
    assert result["review_source"] == "llm"
    assert result["routing"] == {"llm_called": True, "reason": "no previous LLM review"}
    assert result["classifier"]["label"] == "healthy"
    assert result["llm_review"] == {
        "card_reference": "archive:14",
        "card_captured_at": "2026-10-14T09:00:00+00:00",
        "effort": "routine",
        "classifier_label": "healthy",
        "classifier_model_version": "v1",
    }


def test_same_pattern_within_refresh_days_reuses_the_llm_review(routed):
    service, calls = routed
    first = service.analyze(23, archived_reference(12))
    first["summary"] = "Normal full card."
    save_review(service, first)
    result = service.analyze(23, archived_reference(14))
    assert calls["review"] == 1
    assert result["review_source"] == "classifier"
    assert result["review_status"] == "complete"
    assert result["summary"] == "Normal full card."
    assert result["llm_review"]["card_reference"] == "archive:12"
    assert result["input_cards"][0]["reference"] == "archive:14"
    assert result["measurements"]["downhole"]["status"] == "valid"
    assert result["token_usage"] is None
    assert result["routing"]["llm_called"] is False


@pytest.mark.parametrize(
    ("change", "reason", "routine"),
    [
        (lambda s: setattr(s, "classifier", FakeClassifier("fluid_pound_mild")),
         "pattern changed from healthy to fluid_pound_mild", 0),
        (lambda s: setattr(s, "classifier", FakeClassifier("gas_interference_severe")),
         "gas_interference_severe needs an LLM review", 0),
        (lambda s: setattr(s, "classifier", FakeClassifier(confidence=0.8)),
         "classifier confidence 0.80 is below the threshold", 0),
        (lambda s: setattr(s, "classifier", FakeClassifier(version="v2")),
         "classifier model changed since the last LLM review", 1),
        (lambda s: s.routing.update(min_downhole_load_range_lb=3000), "collapsed downhole card", 0),
        (lambda s: s.routing.update(refresh_days=1), "previous LLM review is 2 days old", 1),
    ],
)
def test_changed_or_uncertain_cards_go_back_to_the_llm(routed, change, reason, routine):
    service, calls = routed
    save_review(service, service.analyze(23, archived_reference(12)))
    calls["review"] = calls["routine"] = 0
    change(service)
    result = service.analyze(23, archived_reference(14))
    assert result["routing"] == {"llm_called": True, "reason": reason}
    assert result["review_source"] == "llm"
    assert (calls["review"], calls["routine"]) == (1, routine)


def test_as_of_run_does_not_reuse_a_review_of_a_later_card(routed):
    service, calls = routed
    save_review(service, service.analyze(23, archived_reference(14)))
    result = service.analyze(23, archived_reference(12))
    assert result["routing"]["reason"] == "previous LLM review is of a later card"
    assert calls["review"] == 2


def test_failed_or_legacy_reviews_are_not_reused(routed):
    service, calls = routed
    first = service.analyze(23, archived_reference(12))
    save_review(service, {**first, "llm_review": None})
    assert service.analyze(23, archived_reference(14))["routing"]["reason"] == "no previous LLM review"
    save_review(service, {**first, "review_status": "failed"})
    assert service.analyze(23, archived_reference(14))["routing"]["llm_called"] is True
    assert calls["review"] == 3


def test_classifier_version_is_part_of_the_signature(routed):
    service, _ = routed
    first = service.analyze(23, archived_reference())["input_signature"]
    service.classifier = FakeClassifier(version="v2")
    assert service.analyze(23, archived_reference())["input_signature"] != first


def test_without_classifier_every_card_goes_to_the_llm_at_the_default_effort(routed):
    service, calls = routed
    service.classifier = None
    save_review(service, service.analyze(23, archived_reference(12)))
    result = service.analyze(23, archived_reference(14))
    assert result["routing"] == {"llm_called": True, "reason": "classifier unavailable"}
    assert result["classifier"] is None
    assert calls["routine"] == 0


def test_training_command_writes_a_loadable_model(tmp_path, capsys):
    from umai.rod_pump_health.train_classifier import main

    rng = np.random.default_rng(0)
    rows = []
    for card in CARDS[:2]:
        for copy in range(8):
            points = [
                {"position": p["position"], "load": p["load"] + float(rng.normal(0, 20))}
                for p in card["points"]
            ]
            rows.append({
                "well": f"W{copy}", "source_card_id": f"{card['label']}{copy}",
                "card_date": "2026-10-01", "data_points": json.dumps(points),
                "health": card["health"], "label": card["label"],
            })
    import pandas as pd

    training = tmp_path / "training.csv"
    pd.DataFrame(rows).to_csv(training, index=False)
    output = tmp_path / "model.joblib"
    main(["--train-file", str(training), "--output", str(output), "--trees", "10", "--folds", "2"])
    assert "Cross-validation holds out whole wells" in capsys.readouterr().out
    classifier = load_classifier(str(output))
    assert classifier.classify(CARDS[1]["points"])["label"] == CARDS[1]["label"]
