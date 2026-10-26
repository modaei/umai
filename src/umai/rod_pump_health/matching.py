"""Local, axis-preserving similarity matching of closed dynograph outlines."""

import math

from umai.rod_pump_health.templates import TEMPLATES, VERSION as TEMPLATE_VERSION

VERSION = "local-dynograph-matching-v11"
MIN_SCORE = 75.0
SAMPLES = 64
HEALTHY_TEMPLATE_ID = "dh_full"


def normalized_outline(points, *, template=False, load_range=None):
    """Resample outlines without stretching template loads or independently scaling narrow bands."""
    points = [(float(x), float(y)) for x, y in points]
    if len(points) < 4 or not all(math.isfinite(v) for point in points for v in point):
        return []
    xs, ys = zip(*points)
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    if dx <= 0 or dy <= 0:
        return []
    midpoint = (max(ys) + min(ys)) / 2
    points = [
        ((x - min(xs)) / dx, 0.5 + (y - midpoint) / (1 if template else (load_range or dy)))
        for x, y in points
    ]
    if points[-1] != points[0]:
        points.append(points[0])
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    output, edge = [], 0
    for index in range(SAMPLES):
        target = lengths[-1] * index / SAMPLES
        while edge < len(points) - 2 and lengths[edge + 1] <= target:
            edge += 1
        fraction = (target - lengths[edge]) / (lengths[edge + 1] - lengths[edge] or 1)
        output.append(tuple(a + fraction * (b - a) for a, b in zip(points[edge], points[edge + 1])))
    return output


def reference_load_range(reference):
    """Accept a trusted expected range or validated normal-card range, never the current card's range.

    Optional calculation_snapshot.load_references entries are per side and contain source and
    load_range_lb. The current reader supplies none, so low-load patterns remain unevaluated.
    """
    if not isinstance(reference, dict) or reference.get("source") not in (
        "expected_full_load", "validated_normal_card",
    ):
        return None
    value = reference.get("load_range_lb")
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def match_score(actual, reference, load_reference=None):
    """Compare against fixed template loads; narrow patterns require an independent load scale."""
    scale = reference_load_range(load_reference)
    if not reference:
        return 0.0
    span = max(y for _, y in reference) - min(y for _, y in reference)
    # Gas-lock and unloading sketches encode low loading, which self-normalization cannot establish.
    if span < 0.3 and scale is None:
        return 0.0
    left = normalized_outline(actual, load_range=scale)
    right = normalized_outline(reference, template=True)
    if not left or not right:
        return 0.0
    error = min(
        sum(math.dist(point, candidate[(index + shift) % SAMPLES]) ** 2
            for index, point in enumerate(left)) / SAMPLES
        for candidate in (right, list(reversed(right))) for shift in range(SAMPLES)
    )
    # A geometric similarity index, not a diagnostic probability; RMS 0.107 gives about 75.
    return round(100 * math.exp(-error / 0.20 ** 2), 1)



def evaluate_outline(points, template, load_reference=None):
    """Gate each template by its structure and compare fluid-pound body/handle regions separately."""
    from umai.rod_pump_health.shape_structure import (
        outline_structure, regional_outline_score, structural_checks, template_variants, tagging_outline_score,
    )
    physical_scale = reference_load_range(load_reference)
    # Increased separation is verified against the reference; its remaining contour is compared by shape.
    shape_scale = None if template["id"] == "sf_friction" else physical_scale
    metrics = outline_structure(points, load_range=shape_scale)
    if metrics is None:
        return None
    loads = [y for _, y in points]
    ratio = (max(loads) - min(loads)) / physical_scale if physical_scale else None
    checks = structural_checks(template["id"], metrics, load_ratio=ratio, closure=metrics["closure"])
    best = None
    for variant, parameters in template_variants(template, metrics):
        score = match_score(
            points, variant, load_reference=None if template["id"] == "sf_friction" else load_reference,
        )
        result = {"score": score}
        if template["id"] == "dh_fluid_pound" and metrics["handle_length"] >= 0.08:
            result = regional_outline_score(
                normalized_outline(points, load_range=shape_scale), normalized_outline(variant, template=True),
                metrics["handle_start"],
            )
        if template["id"] == "dh_tagging":
            result = tagging_outline_score(points, variant)
        result.update({
            "structural_checks": checks, "structure_passed": all(row["passed"] for row in checks),
            "structural_score": round(100 * sum(row["passed"] for row in checks) / len(checks), 1),
            "matched_template_points": variant, "family_parameters": parameters,
        })
        if best is None or result["score"] > best["score"]:
            best = result
    return best

def analyze_local(packet):
    """Run local outline matching with structural gates and healthy-winner selection."""
    matches, selected, unavailable, rejections = [], {}, [], []
    missing_references = []
    references = packet.calculation_snapshot.get("load_references", {})
    references = references if isinstance(references, dict) else {}
    for side in ("surface", "downhole"):
        recent = [card for card in packet.cards if "recent" in card.roles]
        candidates = recent or packet.cards
        card = max(candidates, key=lambda row: row.timestamp)
        selected[side] = card.reference
        points = [(point.position, point.load) for point in getattr(card, side)]
        if not normalized_outline(points):
            unavailable.append(side)
            continue
        load_reference = references.get(side)
        if reference_load_range(load_reference) is None:
            missing_references.append(side)
        for template in TEMPLATES:
            if template["side"] != side:
                continue
            assessment = evaluate_outline(points, template, load_reference)
            if assessment is None:
                continue
            if assessment["score"] >= MIN_SCORE and assessment["structure_passed"]:
                matches.append({
                    **assessment, "side": side, "template_id": template["id"], "title": template["title"],
                    "card_reference": card.reference,
                })
            elif not assessment["structure_passed"]:
                rejections.append({"side": side, "title": template["title"],
                                   "failed_checks": [row["feature"] for row in assessment["structural_checks"]
                                                     if not row["passed"]]})
    # A tied problem score wins the tie; healthy must lead the well's combined ranking.
    matches.sort(key=lambda row: (
        -row["score"], row["template_id"] == HEALTHY_TEMPLATE_ID, row["template_id"],
    ))
    healthy = bool(matches and matches[0]["template_id"] == HEALTHY_TEMPLATE_ID)
    if healthy:
        matches = matches[:1]
    else:
        matches = [row for row in matches if row["template_id"] != HEALTHY_TEMPLATE_ID][:2]
    return {
        "structural_rejections": rejections,
        "status": "partial" if unavailable else "complete", "review_status": "complete",
        "review_error": None, "analyzer_version": VERSION, "template_version": TEMPLATE_VERSION,
        "diagnosis_status": "healthy" if healthy else "matches" if matches else "no_match", "findings": [],
        "matches": matches, "matching_method": VERSION, "minimum_match_score": MIN_SCORE,
        "matched_cards": selected, "unavailable_sides": unavailable,
        "measurements": {}, "templates": TEMPLATES, "token_usage": None,
        "limitations": ["Scores measure normalized shape similarity, not fault probability."] + (
            ["Load-dependent patterns (low load / increased separation) were not evaluated for "
             + ", ".join(missing_references)
             + ": no validated full-load reference is available."] if missing_references else []
        ),
    }
