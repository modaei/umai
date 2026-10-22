from umai.services.rod_pump_analysis_engine import (
    anomaly_score,
    fuse_diagnoses,
)


def test_anomaly_cannot_create_a_named_diagnosis():
    anomaly = anomaly_score(
        {"Pump Fillage": {"latest": 200}},
        [],
        {"Pump Fillage": [{"value": 80}] * 24},
        [],
    )
    assert anomaly["available"] is True
    assert anomaly["creates_diagnosis"] is False

    fused = fuse_diagnoses([], anomaly, {"available": False})
    assert fused == []


def test_anomaly_only_strengthens_existing_non_normal_rule():
    rules = [{
        "name": "Rod/tubing friction",
        "severity": "medium",
        "confidence": .6,
        "evidence": [],
        "contradictory_evidence": [],
        "affected_timestamps": [],
        "main_metrics": [],
        "recommended_action": "Review",
        "alternative_causes": [],
        "provenance": ["rule"],
    }]
    fused = fuse_diagnoses(
        rules,
        {"available": True, "score": .8},
        {"available": False},
    )
    assert fused[0]["confidence"] == .68
    assert fused[0]["provenance"] == ["rule", "anomaly"]


def test_validated_classifier_only_fuses_with_paraffin_candidate():
    base = {
        "severity": "medium", "confidence": .5, "evidence": [],
        "contradictory_evidence": [], "affected_timestamps": [],
        "main_metrics": [], "recommended_action": "Review",
        "alternative_causes": [], "provenance": ["rule"],
    }
    fused = fuse_diagnoses(
        [
            {"name": "Possible paraffin/wax buildup", **base},
            {"name": "Rod/tubing friction", **base},
        ],
        {"available": False},
        {"available": True, "probability": .9},
    )
    paraffin = next(row for row in fused if "paraffin" in row["name"].lower())
    friction = next(row for row in fused if row["name"] == "Rod/tubing friction")
    assert paraffin["confidence"] == .64
    assert "classifier" in paraffin["provenance"]
    assert friction["confidence"] == .5
