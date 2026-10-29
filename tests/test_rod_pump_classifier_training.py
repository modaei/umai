import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from umai.config.settings import DEFAULT_CLASSIFIER_MODEL
from umai.rod_pump_health import classifier_cli, train_classifier
from umai.rod_pump_health.classifier import file_version
from umai.rod_pump_health.classifier_registry import ClassifierRegistry, RegistryError

ROOT = Path(__file__).resolve().parents[1]
CARDS = json.loads((ROOT / "tests/fixtures/rod_pump_classifier_cards.json").read_text())
NOW = datetime(2026, 11, 8, 2, tzinfo=timezone.utc)
OPTIONS = {"trees": 20, "folds": 3}


def label_engine(rows):
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE rod_pump_card_labels(id INTEGER PRIMARY KEY, well_key TEXT, source_card_id TEXT, "
            "card_timestamp INTEGER, downhole_points TEXT, health TEXT, label TEXT, label_confidence TEXT, "
            "label_source TEXT, source_detail TEXT, classifier_label TEXT, classifier_confidence REAL, "
            "classifier_model_version TEXT, excluded INTEGER DEFAULT 0, created_at TEXT)"
        ))
        for row in rows:
            connection.execute(text(
                "INSERT INTO rod_pump_card_labels(well_key, source_card_id, card_timestamp, downhole_points, "
                "health, label, label_confidence, label_source, source_detail, classifier_label, "
                "classifier_confidence, classifier_model_version, excluded, created_at) VALUES(:well_key, "
                ":source_card_id, :card_timestamp, :downhole_points, :health, :label, :label_confidence, "
                ":label_source, :source_detail, :classifier_label, :classifier_confidence, "
                ":classifier_model_version, :excluded, :created_at)"
            ), row)
    return engine


def label_row(card, well, day, source="rules", **overrides):
    rng = np.random.default_rng(hash((well, day, card["label"])) % 2**32)
    points = [{"position": p["position"], "load": p["load"] + float(rng.normal(0, 30))} for p in card["points"]]
    return {
        "well_key": well, "source_card_id": f"{card['label']}-{day}",
        "card_timestamp": int((NOW - timedelta(days=60 - day)).timestamp()),
        "downhole_points": json.dumps(points), "health": card["health"], "label": card["label"],
        "label_confidence": "high" if source == "llm_review" else None, "label_source": source,
        "source_detail": None, "classifier_label": None, "classifier_confidence": None,
        "classifier_model_version": None, "excluded": 0,
        "created_at": (NOW - timedelta(days=1)).isoformat(), **overrides,
    }


def training_rows():
    """Ten wells with every fixture pattern on several days; a quarter are LLM labels."""
    return [
        label_row(card, f"W{well}", day, "llm_review" if day % 4 == 0 else "rules")
        for well in range(10) for card in CARDS for day in range(4)
    ]


@pytest.fixture
def registry(tmp_path):
    return ClassifierRegistry(str(tmp_path / "registry"), str(ROOT / DEFAULT_CLASSIFIER_MODEL), 0.9)


@pytest.fixture
def trained(registry):
    report = train_classifier.train(label_engine(training_rows()), registry, 0.9, OPTIONS, now=NOW, wells_reviewed=12)
    return registry, report


def test_prepare_keeps_one_trusted_label_per_card_and_day():
    healthy, pound = CARDS[0], CARDS[1]
    rows = [
        label_row(healthy, "W1", 1, "rules"),
        # Same card from a more trusted source wins.
        label_row(pound, "W1", 1, "human_review", source_card_id="healthy-1"),
        # Same well, label and day: thinned away.
        label_row(healthy, "W1", 1, "rules", source_card_id="healthy-1b"),
        label_row(healthy, "W2", 1, "llm_review", label_confidence="medium"),
        label_row(healthy, "W3", 1, "rules", label="tv_leak_suspect", health=None),
        label_row(healthy, "W4", 1, "llm_review", label="other"),
        label_row(healthy, "W5", 1, "rules", excluded=1),
    ]
    frame, features = train_classifier.prepare(train_classifier.read_labels(label_engine(rows)))
    assert sorted(zip(frame["well_key"], frame["label"], frame["label_source"])) == [
        ("W1", "fluid_pound_severe", "human_review"),
        ("W1", "healthy", "rules"),
        ("W3", "traveling_valve_leak", "rules"),
    ]
    assert features.shape == (3, 205)
    assert frame.set_index("label").loc["traveling_valve_leak", "health"] == "Problem"


def test_training_saves_a_candidate_and_leaves_the_live_model(trained):
    registry, report = trained
    assert registry.candidates() == ["20261108-020000"]
    assert registry.pointer() is None
    assert registry.state()["source"] == "legacy"
    assert report["live"]["version"] == file_version(ROOT / DEFAULT_CLASSIFIER_MODEL)
    assert {"healthy", "fluid_pound_severe", "gas_interference_severe", "low_fluid_load"} <= set(report["labels"])
    healthy = report["labels"]["healthy"]
    assert healthy["cards_real"] == 40 and healthy["qualifies"] and healthy["threshold"] >= 0.9
    # The live model trusts two fluid pound grades this data has no cards of, so it is kept.
    recommendation = report["recommendation"]
    assert recommendation["action"] == "keep_current"
    assert recommendation["trusted_lost"] == ["fluid_pound_mild", "fluid_pound_moderate"]
    assert recommendation["reasons"][0] == (
        "fluid_pound_mild is trusted now but has no training cards, so the candidate cannot decide it."
    )
    assert recommendation["new_trusted"] == ["gas_interference_severe", "low_fluid_load"]
    assert recommendation["commands"] == {"status": "umai-card-classifier status", "untrust": []}
    assert report["re_review_wells"] == 12
    assert all(len(example["points"]) <= 200 for example in report["examples"])
    assert registry.report("20261108-020000") == json.loads(json.dumps(report, default=str))


def test_promote_trust_untrust_and_rollback(trained):
    registry, report = trained
    version = report["version"]
    qualifying = [label for label, entry in report["labels"].items() if entry["qualifies"]]
    new = next(label for label in qualifying if label not in registry.state()["trusted"])
    with pytest.raises(RegistryError, match="does not qualify"):
        registry.promote(version, "mo", trust=["rod_part"])

    state, notes = registry.promote(version, "mo", trust=[new])
    assert state["version"] == version and state["updated_by"] == "mo"
    assert new in state["trusted"]
    assert state["trusted"][new] == report["labels"][new]["threshold"]
    # The legacy model was adopted under its own hash, so rollback returns to it unchanged.
    assert state["previous"] == [file_version(ROOT / DEFAULT_CLASSIFIER_MODEL)]
    live = registry.active()
    assert live.version == version and live.trusted == state["trusted"]

    state = registry.set_trust(new, False, "mo")
    assert new not in state["trusted"]
    assert new not in registry.active().trusted
    with pytest.raises(RegistryError, match="is not trusted"):
        registry.set_trust(new, False, "mo")
    state = registry.set_trust(new, True, "mo")
    assert state["trusted"][new] == report["labels"][new]["threshold"]

    state = registry.rollback("mo")
    assert state["version"] == file_version(ROOT / DEFAULT_CLASSIFIER_MODEL)
    assert sorted(state["trusted"]) == ["fluid_pound_mild", "fluid_pound_moderate", "fluid_pound_severe", "healthy"]
    assert [entry["action"] for entry in state["history"]] == [
        "adopt legacy model", f"promote {version}", f"untrust {new}", f"trust {new}",
        f"rollback to {state['version']}",
    ]
    with pytest.raises(RegistryError, match="no earlier model"):
        registry.rollback("mo")


def test_untrusting_a_legacy_label_keeps_the_legacy_model_version(registry):
    legacy = file_version(ROOT / DEFAULT_CLASSIFIER_MODEL)
    state = registry.set_trust("fluid_pound_mild", False, "mo")
    assert state["version"] == legacy and "fluid_pound_mild" not in state["trusted"]
    assert registry.active().version == legacy
    with pytest.raises(RegistryError, match="does not qualify"):
        registry.set_trust("gas_interference_severe", True, "mo")
    state = registry.set_trust("gas_interference_severe", True, "mo", threshold=0.97, force=True)
    assert state["trusted"]["gas_interference_severe"] == 0.97


def test_should_train_needs_new_data(trained):
    registry, report = trained
    rows = training_rows()
    later = NOW + timedelta(days=2)
    needed, reason = train_classifier.should_train(train_classifier.read_labels(label_engine(rows)), registry,
                                                   train_classifier.DEFAULTS, later)
    assert (needed, reason) == (False, f"only 0 new labels since {report['trained_at']}")
    rows.append(label_row(CARDS[3], "W1", 9, "field_event", created_at=later.isoformat()))
    assert train_classifier.should_train(train_classifier.read_labels(label_engine(rows)), registry,
                                         train_classifier.DEFAULTS, later)[1] == (
        "new field-confirmed or human-reviewed labels"
    )
    assert train_classifier.should_train(pd.DataFrame(), registry, train_classifier.DEFAULTS,
                                         NOW + timedelta(days=31))[0]


def test_cli_status_untrust_and_latest_training(trained, monkeypatch, capsys):
    registry, report = trained
    registry.record_run({"status": "trained", "version": report["version"]})
    monkeypatch.setattr(classifier_cli, "_settings", lambda: None)
    monkeypatch.setattr(classifier_cli, "_registry", lambda settings: registry)
    assert classifier_cli.main(["status"]) == 0
    assert "original model file, not yet in the registry" in capsys.readouterr().out
    assert classifier_cli.main(["untrust", "healthy", "--by", "ald"]) == 0
    output = capsys.readouterr().out
    assert "Last changed" in output and "by ald" in output and "  healthy " not in output
    assert classifier_cli.main(["untrust", "healthy"]) == 2
    latest = classifier_cli.latest_training(registry)
    assert latest["run"]["version"] == report["version"]
    assert latest["report"]["version"] == report["version"]
    assert "healthy" not in latest["live"]["trusted"]


def test_promotion_is_recommended_with_the_exact_command(registry):
    registry.set_trust("fluid_pound_mild", False, "mo")
    registry.set_trust("fluid_pound_moderate", False, "mo")
    report = train_classifier.train(label_engine(training_rows()), registry, 0.9, OPTIONS, now=NOW)
    recommendation = report["recommendation"]
    assert recommendation["action"] == "promote_with_new_labels"
    assert recommendation["commands"]["promote"] == (
        "umai-card-classifier promote 20261108-020000 --trust gas_interference_severe --trust low_fluid_load"
    )
    assert recommendation["commands"]["promote_without_new_labels"] == "umai-card-classifier promote 20261108-020000"
    # The recommended command works as written.
    state, notes = registry.promote(report["version"], "mo", trust=["gas_interference_severe", "low_fluid_load"])
    assert sorted(state["trusted"]) == ["fluid_pound_severe", "gas_interference_severe", "healthy", "low_fluid_load"]
    assert notes == []
