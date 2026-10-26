"""Directional branch comparison with bounded alignment and explicit feature gates; no LLMs."""

import math
from statistics import mean

from umai.rod_pump_health.matching import HEALTHY_TEMPLATE_ID, MIN_SCORE, reference_load_range
from umai.rod_pump_health.templates import TEMPLATES
from umai.rod_pump_health.shape_structure import directional_structure, structural_checks, template_variants

VERSION = "branch-features-v3"
BRANCH_SAMPLES = 48
DTW_BAND = 2
AMBIGUITY_MARGIN = 5.0
MIN_CANDIDATE_SCORE = 50.0


def interpolate(branch, position):
    """Interpolate along the recorded branch without swapping or sorting its stroke direction."""
    for (x1, y1), (x2, y2) in zip(branch, branch[1:]):
        if min(x1, x2) <= position <= max(x1, x2) and x1 != x2:
            return y1 + (position - x1) / (x2 - x1) * (y2 - y1)
    return None


def crossing(branch, level):
    """Locate the first load transition along the stroke's acquisition direction."""
    for (x1, y1), (x2, y2) in zip(branch, branch[1:]):
        if min(y1, y2) <= level <= max(y1, y2) and y1 != y2:
            return x1 + (level - y1) / (y2 - y1) * (x2 - x1)
    return None


def prepare_branches(points, point_order="unknown", *, template=False, load_reference=None):
    """Verify closure and direction, then resample each stroke at common position intervals."""
    if len(points) < 4 or not all(math.isfinite(v) for point in points for v in point):
        return None, ["Fewer than four usable position/load points."]
    xs, ys = zip(*points)
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    dx, dy = xmax - xmin, ymax - ymin
    if dx <= 0 or dy <= 0:
        return None, ["The curve has no usable stroke or load range."]
    reasons = []
    if not template and point_order != "sam1_reverse_time":
        reasons.append("Upstroke/downstroke direction is unverified.")
    closure = math.hypot((points[0][0] - points[-1][0]) / dx, (points[0][1] - points[-1][1]) / dy)
    if not template and closure > 0.08:
        reasons.append("The curve is not sufficiently closed.")
    scale = 1 if template else reference_load_range(load_reference) or dy
    midpoint = (ymin + ymax) / 2
    sequence = [((x - xmin) / dx, 0.5 + (y - midpoint) / scale) for x, y in points]
    if not template and point_order == "sam1_reverse_time":
        sequence.reverse()
    if sequence[0] == sequence[-1]:
        sequence.pop()
    start = min(range(len(sequence)), key=lambda i: (sequence[i][0], sequence[i][1]))
    sequence = sequence[start:] + sequence[:start]
    sequence.append(sequence[0])
    turn = max(range(len(sequence)), key=lambda i: sequence[i][0])
    branches = {"upstroke": sequence[:turn + 1], "downstroke": sequence[turn:]}
    samples = {}
    for name, branch in branches.items():
        sign = 1 if name == "upstroke" else -1
        if len(branch) < 2:
            return None, [*reasons, f"Cannot resolve the {name} branch."]
        monotonic = sum(sign * (b[0] - a[0]) >= -0.015 for a, b in zip(branch, branch[1:]))
        if monotonic / (len(branch) - 1) < 0.85:
            reasons.append(f"The {name} branch has excessive position reversals.")
        positions = [0.02 + 0.96 * i / (BRANCH_SAMPLES - 1) for i in range(BRANCH_SAMPLES)]
        if name == "downstroke":
            positions.reverse()
        sampled = [(x, interpolate(branch, x)) for x in positions]
        if any(y is None for _, y in sampled):
            return None, [*reasons, f"The {name} branch does not cover the stroke."]
        samples[name] = sampled
    samples["features"] = branch_features(branches, samples, sequence)
    samples["structure"] = directional_structure(samples, sequence)
    samples["physical_load_range"] = dy
    return samples, reasons


def branch_features(branches, samples, sequence):
    """Measure independent defining features before any warping can conceal their differences."""
    output = {}
    for name, branch in branches.items():
        central = [y for x, y in samples[name] if 0.2 <= x <= 0.8]
        output[name + "_level"] = mean(central)
        output[name + "_slope"] = (interpolate(branch, 0.8) - interpolate(branch, 0.2)) / 0.6
        positions = [crossing(branch, level) for level in (0.25, 0.5, 0.75)]
        output[name + "_transition"] = positions[1]
        output[name + "_transition_width"] = (
            abs(positions[2] - positions[0]) if positions[0] is not None and positions[2] is not None else None
        )
    upper = samples["upstroke"]
    lower = list(reversed(samples["downstroke"]))
    output["separation"] = mean(a[1] - b[1] for a, b in zip(upper, lower))
    central_min = min(y for name in branches for x, y in samples[name] if 0.2 <= x <= 0.8)
    output["end_spike"] = max(0, central_min - min(y for _, y in sequence))
    output["area"] = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(sequence, sequence[1:]))) / 2
    return output


def constrained_error(left, right):
    """Align within two position samples, penalizing non-diagonal steps to limit shape distortion."""
    count = len(left)
    costs = {(0, 0): (0.0, 0)}
    for i in range(1, count + 1):
        for j in range(max(1, i - DTW_BAND), min(count, i + DTW_BAND) + 1):
            candidates = []
            for previous, penalty in (((i - 1, j - 1), 0), ((i - 1, j), 0.002), ((i, j - 1), 0.002)):
                if previous in costs:
                    cost, length = costs[previous]
                    candidates.append((cost + penalty, length))
            if candidates:
                cost, length = min(candidates)
                costs[(i, j)] = (cost + math.dist(left[i - 1], right[j - 1]) ** 2, length + 1)
    cost, length = costs[(count, count)]
    return cost / length


FEATURES = (
    ("upstroke_level", "Upstroke load level", 0.18),
    ("downstroke_level", "Downstroke load level", 0.18),
    ("upstroke_slope", "Upstroke slope", 0.35),
    ("downstroke_slope", "Downstroke slope", 0.35),
    ("upstroke_transition", "Load pickup position", 0.15),
    ("downstroke_transition", "Load release position", 0.15),
    ("upstroke_transition_width", "Load pickup width", 0.15),
    ("downstroke_transition_width", "Load release width", 0.15),
    ("separation", "Branch separation", 0.22),
    ("end_spike", "End-of-stroke spike", 0.12),
    ("area", "Enclosed area", 0.18),
)


def compare_branches(actual, reference):
    """Combine branch distance with visible feature checks; failed checks prevent acceptance."""
    direct = mean(math.dist(a, b) ** 2 for name in ("upstroke", "downstroke")
                  for a, b in zip(actual[name], reference[name]))
    error, alignment = direct, "direct"
    if direct > 0.003:
        warped = mean(constrained_error(actual[name], reference[name]) for name in ("upstroke", "downstroke"))
        if warped < direct * 0.95:
            error, alignment = warped, "constrained_dtw"
    checks = []
    for key, label, tolerance in FEATURES:
        measured, expected = actual["features"][key], reference["features"][key]
        # A narrow low-load sketch may never cross a nominal transition threshold.
        if expected is None:
            continue
        delta = abs(measured - expected) if measured is not None else None
        checks.append({
            "feature": label, "actual": round(measured, 4) if measured is not None else None,
            "template": round(expected, 4), "tolerance": tolerance,
            "passed": delta is not None and delta <= tolerance,
            "similarity": max(0, 1 - delta / (2 * tolerance)) if delta is not None else 0,
        })
    curve_score = 100 * math.exp(-error / 0.16 ** 2)
    feature_score = 100 * mean(row["similarity"] for row in checks)
    return {
        "score": round(0.8 * curve_score + 0.2 * feature_score, 1),
        "curve_score": round(curve_score, 1), "feature_score": round(feature_score, 1),
        "feature_checks": [{key: value for key, value in row.items() if key != "similarity"} for row in checks],
        "features_passed": all(row["passed"] for row in checks), "alignment": alignment,
    }



def compare_template_branches(actual, template, load_reference=None):
    """Fit a bounded template family while enforcing independently measured directional structures."""
    scale = reference_load_range(load_reference)
    ratio = actual["physical_load_range"] / scale if scale else None
    checks = structural_checks(template["id"], actual["structure"], load_ratio=ratio)
    best = None
    for variant, parameters in template_variants(template, actual["structure"]):
        reference, _ = prepare_branches(variant, template=True)
        if reference is None:
            continue
        result = compare_branches(actual, reference)
        result.update({
            "structural_checks": checks,
            "structure_passed": all(row["passed"] for row in checks),
            "structural_score": round(100 * sum(row["passed"] for row in checks) / len(checks), 1),
            "features_passed": result["features_passed"] and all(row["passed"] for row in checks),
            "matched_template_points": variant, "family_parameters": parameters,
        })
        if best is None or result["score"] > best["score"]:
            best = result
    return best

def analyze_branches(packet):
    """Produce independent accepted or ambiguous rankings on the same selected cards, without LLM calls."""
    candidates, unavailable = [], []
    references = packet.calculation_snapshot.get("load_references", {})
    references = references if isinstance(references, dict) else {}
    for side in ("surface", "downhole"):
        recent = [card for card in packet.cards if "recent" in card.roles]
        card = max(recent or packet.cards, key=lambda row: row.timestamp)
        points = [(point.position, point.load) for point in getattr(card, side)]
        actual, issues = prepare_branches(points, card.point_order, load_reference=references.get(side))
        if actual is None:
            unavailable.append(side)
            continue
        for template in TEMPLATES:
            if template["side"] != side:
                continue
            span = max(y for _, y in template["points"]) - min(y for _, y in template["points"])
            if span < 0.3 and reference_load_range(references.get(side)) is None:
                continue
            candidate_actual = actual
            if template["id"] == "sf_friction" and reference_load_range(references.get(side)):
                candidate_actual, _ = prepare_branches(points, card.point_order)
            result = compare_template_branches(candidate_actual, template, references.get(side))
            if result is None:
                continue
            candidates.append({**result, "side": side, "template_id": template["id"],
                               "title": template["title"], "card_reference": card.reference,
                               "quality_reasons": issues})
    candidates.sort(key=lambda row: (-row["score"], row["template_id"] == HEALTHY_TEMPLATE_ID, row["template_id"]))
    # A failed defining structure is a rejection, never an ambiguous candidate or a competing match.
    eligible = [row for row in candidates if row["structure_passed"]]
    strong = [row for row in eligible if row["score"] >= MIN_SCORE and row["features_passed"]
              and not row["quality_reasons"]]
    reasons = []
    if strong:
        selected = strong
        healthy_leads = selected[0]["template_id"] == HEALTHY_TEMPLATE_ID
        retained = selected[:1] if healthy_leads else [
            row for row in selected if row["template_id"] != HEALTHY_TEMPLATE_ID
        ][:2]
        for leader in retained:
            # Compare rivals on the same physical card, ignoring a non-winning healthy pattern.
            rivals = [row for row in eligible if row["side"] == leader["side"]
                      and row["template_id"] != leader["template_id"] and row["score"] >= MIN_SCORE
                      and (healthy_leads or row["template_id"] != HEALTHY_TEMPLATE_ID)]
            if rivals and abs(leader["score"] - rivals[0]["score"]) < AMBIGUITY_MARGIN:
                reasons.append(
                    f"Competing templates are within {AMBIGUITY_MARGIN:g} score points on {leader['side']}."
                )
    else:
        selected = [row for row in eligible if row["score"] >= MIN_CANDIDATE_SCORE]
    if selected:
        healthy = selected[0]["template_id"] == HEALTHY_TEMPLATE_ID
        selected = selected[:1] if healthy else [
            row for row in selected if row["template_id"] != HEALTHY_TEMPLATE_ID
        ][:2]
        if not strong:
            reasons.extend(f"{row['side']}: {issue}" for row in selected for issue in row["quality_reasons"])
            if any(not row["features_passed"] for row in selected):
                reasons.append("Defining feature checks do not support a clear match.")
            if selected[0]["score"] < MIN_SCORE:
                reasons.append("The best candidate is below the acceptance cutoff.")
        status = "ambiguous" if reasons or not strong else "healthy" if healthy else "matches"
    else:
        status = "unavailable" if len(unavailable) == 2 else "no_match"
    return {
        "method": VERSION, "diagnosis_status": status, "matches": selected,
        "structural_rejections": [
            {"side": row["side"], "title": row["title"],
             "failed_checks": [check["feature"] for check in row["structural_checks"] if not check["passed"]]}
            for row in candidates if any(not check["passed"] for check in row["structural_checks"])
        ],
        "reasons": list(dict.fromkeys(reasons)), "unavailable_sides": unavailable,
        "minimum_match_score": MIN_SCORE, "ambiguity_margin": AMBIGUITY_MARGIN,
        "dtw_band_samples": DTW_BAND,
        "direction": "Preserved from SAM1 reverse-time provenance; unverified direction is ambiguous.",
    }
