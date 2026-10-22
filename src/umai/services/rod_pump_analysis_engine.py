from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any


SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def extract_trend_features(series: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Summarize every telemetry series before diagnostic rules run."""
    return {name: _series_features(points) for name, points in series.items()}


def extract_card_features(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract full-card geometry and changes between pulls of the same category."""
    output = []
    previous: dict[str, dict[str, float]] = {}
    for card in cards:
        values = _card_values(card["points"])
        # Surface and downhole cards have independent histories. Comparing one
        # category to the other would turn normal load differences into false
        # shrinkage or spread-change signals.
        prior = previous.get(card["category"])
        values["area_change_ratio"] = _ratio(values["area"], prior["area"]) if prior else None
        values["load_spread_change_ratio"] = _ratio(values["load_spread"], prior["load_spread"]) if prior else None
        output.append({"pull_time": card["pull_time"], "category": card["category"], **values})
        previous[card["category"]] = values
    return output


def anomaly_score(
    trends: dict[str, dict[str, Any]],
    cards: list[dict[str, Any]],
    baseline_series: dict[str, list[dict[str, Any]]],
    baseline_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return robust per-well deviation evidence; never a named diagnosis."""
    deviations = []
    evidence = []
    baseline_sample_count = sum(len(points) for points in baseline_series.values())
    baseline_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_cards:
        baseline_by_category[row["category"]].append(row)

    for metric, feature in trends.items():
        values = [float(point["value"]) for point in baseline_series.get(metric, [])]
        if len(values) < 8 or feature.get("latest") is None:
            continue
        # Median/MAD scoring is less sensitive than mean/stddev to historical
        # failures that remain in an otherwise healthy well's baseline.
        score = _robust_z(float(feature["latest"]), values)
        deviations.append(score)
        if score >= 2.5:
            evidence.append({"source": "anomaly", "metric": metric, "score": round(score, 2)})

    latest_by_category = {row["category"]: row for row in cards}
    for category, current in latest_by_category.items():
        history = baseline_by_category.get(category, [])
        if len(history) < 3:
            continue
        for metric in ("area", "load_spread", "asymmetry", "impact_score"):
            values = [float(row[metric]) for row in history]
            score = _robust_z(float(current[metric]), values)
            deviations.append(score)
            if score >= 2.5:
                evidence.append({"source": "anomaly", "metric": f"{category}_{metric}", "score": round(score, 2)})

    card_coverage = {category: len(rows) for category, rows in baseline_by_category.items()}
    # Either telemetry or card history may establish a useful baseline. Do not
    # silently substitute another well's behavior when both are insufficient.
    if baseline_sample_count < 24 and max(card_coverage.values(), default=0) < 3:
        return {
            "available": False,
            "reason": "Insufficient per-well historical baseline coverage.",
            "baseline_samples": baseline_sample_count,
            "baseline_cards": card_coverage,
        }
    aggregate = mean(min(score / 5, 1) for score in deviations) if deviations else 0
    return {
        "available": True,
        "score": round(aggregate, 3),
        "category": "high" if aggregate >= .7 else "moderate" if aggregate >= .4 else "normal",
        "evidence": evidence,
        "baseline_samples": baseline_sample_count,
        "baseline_cards": card_coverage,
        "creates_diagnosis": False,
    }


def rule_diagnoses(
    trends: dict[str, dict[str, Any]],
    cards: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply explicit engineering heuristics to current trends and cards."""
    fillage = trends.get("Pump Fillage", {})
    peak = trends.get("Peak Load Last Stroke", {})
    minimum = trends.get("Min Load Last Stroke", {})
    spm = trends.get("Yesterday Strokes per minute", {})
    latest = {row["category"]: row for row in cards}
    down = latest.get("downhole", {})
    surface = latest.get("surface", {})
    results = []

    def add(name, severity, confidence, evidence, action, alternatives=None):
        results.append({
            "name": name,
            "severity": severity,
            "confidence": round(min(max(confidence, 0), 1), 2),
            "evidence": [{"source": "rule", "detail": item} for item in evidence],
            "contradictory_evidence": [],
            "affected_timestamps": [row["pull_time"] for row in cards[-2:]],
            "main_metrics": [item.split(":", 1)[0] for item in evidence],
            "recommended_action": action,
            "alternative_causes": alternatives or [],
            "provenance": ["rule"],
        })

    # These thresholds are intentionally visible and deterministic. They are
    # initial engineering heuristics and should be operator-calibrated before
    # their results are used for field decisions.
    fillage_low = fillage.get("latest", 100) < 70
    fillage_falling = (fillage.get("change_ratio") or 0) < -.1
    shrinking = down.get("area_change_ratio") is not None and down.get("area_change_ratio", 0) < -.15
    stable_spm = abs(spm.get("change_ratio") or 0) < .08
    load_rising = (peak.get("change_ratio") or 0) > .08 and (minimum.get("change_ratio") or 0) > .05

    if fillage_low and (fillage_falling or shrinking):
        add("Pump-off / low inflow", "high" if fillage.get("latest", 100) < 50 else "medium", .7, [f"Pump Fillage: latest {fillage.get('latest')}", f"Downhole area change: {down.get('area_change_ratio')}"], "Review inflow, controller settings, and recent downhole cards.")
    if fillage_low and fillage.get("stddev", 0) > 8 and down.get("asymmetry", 0) > .25:
        add("Gas interference", "medium", .62, [f"Pump Fillage variability: {fillage.get('stddev')}", f"Downhole asymmetry: {down.get('asymmetry')}"], "Review fluid level and gas-handling conditions.", ["Low inflow"])
    if down.get("load_spread", math.inf) < 100 and down.get("area", math.inf) < 100:
        add("Gas lock", "high", .7, [f"Downhole load spread: {down.get('load_spread')}", f"Downhole card area: {down.get('area')}"], "Verify gas lock in the field before changing operation.")
    if fillage_low and down.get("impact_score", 0) > .6:
        add("Fluid pound", "high", .72, [f"Pump Fillage: {fillage.get('latest')}", f"Downhole impact score: {down.get('impact_score')}"], "Review pumping speed and inflow before adjusting controls.")
    if down.get("asymmetry", 0) > .45 and down.get("upstroke_mean_load", 0) < down.get("downstroke_mean_load", 0):
        add("Possible traveling valve leak", "medium", .55, [f"Downhole stroke asymmetry: {down.get('asymmetry')}"], "Review the downhole upstroke pattern and compare with a valve test.")
    if down.get("asymmetry", 0) > .45 and down.get("downstroke_mean_load", 0) <= down.get("upstroke_mean_load", 0):
        add("Possible standing valve leak", "medium", .55, [f"Downhole stroke asymmetry: {down.get('asymmetry')}"], "Review the downhole compression pattern and compare with a valve test.")
    if load_rising and stable_spm:
        add("Rod/tubing friction", "medium", .66, [f"Peak-load change: {peak.get('change_ratio')}", f"Minimum-load change: {minimum.get('change_ratio')}", f"SPM change: {spm.get('change_ratio')}"], "Compare loading with prior interventions and inspect for drag.", ["Scale", "Paraffin", "Well deviation"])
    # A prior chart-note event corroborates recurrence but cannot create a
    # paraffin diagnosis without a current load/friction signature.
    paraffin_history = any(note["event"] == "paraffin" for note in notes)
    if load_rising and stable_spm and not fillage_falling and surface.get("load_spread_change_ratio", 0) >= 0:
        add("Possible paraffin/wax buildup", "medium", .7 if paraffin_history else .48, [f"Peak-load trend: {peak.get('trend')}", f"Minimum-load trend: {minimum.get('trend')}", f"Fillage trend: {fillage.get('trend')}", f"Prior paraffin chart note: {paraffin_history}"], "Review treatment history and verify the restriction before scheduling treatment.", ["Scale", "Rod/tubing wear", "Deviation", "Mechanical drag"])
    if peak.get("change_ratio") is not None and peak.get("change_ratio", 0) < -.5 and surface.get("load_spread", math.inf) < 200:
        add("Possible rod parting", "critical", .76, [f"Peak-load change: {peak.get('change_ratio')}", f"Surface load spread: {surface.get('load_spread')}"], "Escalate for immediate field verification.")
    if surface.get("impact_score", 0) > .75:
        add("Possible pump tagging", "high", .65, [f"Surface impact score: {surface.get('impact_score')}"], "Review end-of-stroke position and verify tagging in the field.")
    if not results:
        add("Normal pumping", "info", .75 if cards and fillage.get("count", 0) else .4, [f"Pump Fillage trend: {fillage.get('trend', 'unknown')}", "Dynograph abnormalities: none exceeded configured rules"], "Continue routine monitoring.")
    return sorted(results, key=lambda row: (-SEVERITY_ORDER[row["severity"]], -row["confidence"]))


def fuse_diagnoses(
    rules: list[dict[str, Any]],
    anomaly: dict[str, Any],
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fuse evidence deterministically; anomaly alone cannot create diagnoses."""
    output = []
    anomaly_support = anomaly.get("score", 0) if anomaly.get("available") else 0
    prediction_probability = prediction.get("probability") if prediction.get("available") else None
    for candidate in rules:
        item = {**candidate, "evidence": list(candidate["evidence"]), "provenance": list(candidate["provenance"])}
        confidence = float(item["confidence"])
        # Anomaly evidence can modestly strengthen an existing rule result. It
        # is never allowed to manufacture a named mechanical diagnosis.
        if anomaly_support >= .4 and item["name"] != "Normal pumping":
            confidence = min(1, confidence + min(.1, anomaly_support * .1))
            item["provenance"].append("anomaly")
            item["evidence"].append({"source": "anomaly", "detail": f"Per-well anomaly score: {anomaly_support:.3f}"})
        # Only a classifier that passed validation can expose `available=true`
        # and contribute probability to the paraffin candidate.
        if item["name"] == "Possible paraffin/wax buildup" and prediction_probability is not None:
            confidence = min(1, .65 * confidence + .35 * float(prediction_probability))
            item["provenance"].append("classifier")
            item["evidence"].append({"source": "classifier", "detail": f"Validated next-24-hour probability: {prediction_probability:.3f}"})
        item["confidence"] = round(confidence, 2)
        output.append(item)
    return sorted(output, key=lambda row: (-SEVERITY_ORDER[row["severity"]], -row["confidence"]))


def health_scores(trends: dict[str, dict], cards: list[dict], diagnoses: list[dict]) -> dict[str, Any]:
    """Combine interpretable component scores into the documented 0-100 score."""
    fill = trends.get("Pump Fillage", {})
    fillage = max(0, min(100, float(fill.get("latest", 0)))) if fill.get("count") else 50
    peak = trends.get("Peak Load Last Stroke", {})
    load_stability = max(0, 100-min(100, abs(float(peak.get("change_ratio") or 0))*300 + float(peak.get("stddev", 0))/100))
    abnormal = max((row.get("impact_score", 0)+row.get("asymmetry", 0) for row in cards), default=.5)
    shape = max(0, 100-abnormal*60)
    spm = trends.get("Yesterday Strokes per minute", {})
    cycles = trends.get("Yesterday Cycles", {})
    efficiency = 50 if not spm.get("count") else max(0, 100-min(50, abs(float(spm.get("change_ratio") or 0))*200)-min(50, abs(float(cycles.get("change_ratio") or 0))*100))
    risk = max((SEVERITY_ORDER[d["severity"]]*25*d["confidence"] for d in diagnoses), default=0)
    final = .2*fillage+.2*load_stability+.25*shape+.15*efficiency+.2*(100-risk)
    critical = any(d["severity"] == "critical" for d in diagnoses)
    category = "urgent intervention" if critical or final < 40 else "needs optimization" if final < 60 else "watchlist" if final < 80 else "healthy"
    return {"score": round(final, 1), "category": category, "components": {"pump_fillage": round(fillage, 1), "load_stability": round(load_stability, 1), "dynograph_shape": round(shape, 1), "operating_efficiency": round(efficiency, 1), "mechanical_risk": round(risk, 1)}}


def _series_features(points):
    values = [float(point["value"]) for point in points]
    if not values:
        return {"count": 0, "trend": "unknown"}
    change = _ratio(values[-1], values[0])
    return {"count": len(values), "first": values[0], "latest": values[-1], "minimum": min(values), "maximum": max(values), "mean": mean(values), "stddev": pstdev(values) if len(values)>1 else 0, "change_ratio": change, "trend": "rising" if change is not None and change > .05 else "falling" if change is not None and change < -.05 else "stable"}


def _card_values(points):
    if len(points) < 2:
        return {"area": 0, "width": 0, "height": 0, "peak_load": 0, "minimum_load": 0, "load_spread": 0, "asymmetry": 0, "impact_score": 0, "upstroke_mean_load": 0, "downstroke_mean_load": 0}
    xs = [p["position"] for p in points]; ys = [p["load"] for p in points]
    # Shoelace area requires the original traversal order; the averaging layer
    # preserves repeated positions on the upstroke and downstroke for this reason.
    area = abs(sum(xs[i]*ys[(i+1) % len(xs)]-xs[(i+1) % len(xs)]*ys[i] for i in range(len(xs)))/2)
    half = len(ys)//2; spread = max(ys)-min(ys)
    asymmetry = abs(mean(ys[:half])-mean(ys[half:]))/max(spread, 1) if half else 0
    deltas = [abs(ys[i]-ys[i-1]) for i in range(1, len(ys))]
    return {"area": area, "width": max(xs)-min(xs), "height": spread, "peak_load": max(ys), "minimum_load": min(ys), "load_spread": spread, "asymmetry": asymmetry, "impact_score": max(deltas, default=0)/max(spread, 1), "upstroke_mean_load": mean(ys[:half]) if half else mean(ys), "downstroke_mean_load": mean(ys[half:]) if half else mean(ys)}


def _ratio(current, previous):
    return (current-previous)/abs(previous) if previous else None


def _robust_z(value: float, values: list[float]) -> float:
    center = median(values)
    mad = median(abs(item-center) for item in values)
    if mad == 0:
        standard = pstdev(values) if len(values) > 1 else 0
        return abs(value-center)/standard if standard else 0
    return abs(.6745*(value-center)/mad)
