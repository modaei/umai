"""Local topology and template-specific shape checks; no models or network access."""

import math
from statistics import mean, median

PROFILE_SAMPLES = 128
LOAD_DEPENDENT = {"dh_gas_lock", "sf_unloading", "sf_friction"}


def normalize_points(points, *, template=False, load_range=None):
    """Preserve every vertex and fixed template load separation when normalizing physical axes."""
    if len(points) < 4 or not all(math.isfinite(v) for point in points for v in point):
        return []
    xs, ys = zip(*points)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return []
    scale = 1 if template else load_range or ymax - ymin
    middle = (ymin + ymax) / 2
    return [((x - xmin) / (xmax - xmin), 0.5 + (y - middle) / scale) for x, y in points]


def quantile(values, fraction):
    """Interpolate a robust range statistic independently of original sampling density."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def transition(profile, key, level):
    """Locate an increasing geometric transition, including ideal vertical boundary transitions."""
    if profile[0][key] >= level:
        return 0.0
    for a, b in zip(profile, profile[1:]):
        if a[key] <= level <= b[key] and a[key] != b[key]:
            return a["x"] + (level - a[key]) / (b[key] - a[key]) * (b["x"] - a["x"])
    return 1.0


def slope(rows, key):
    """Estimate a plateau slope without letting a single corner dominate its value."""
    xmean, ymean = mean(row["x"] for row in rows), mean(row[key] for row in rows)
    variance = sum((row["x"] - xmean) ** 2 for row in rows)
    return sum((row["x"] - xmean) * (row[key] - ymean) for row in rows) / (variance or 1)


def localized_excursion(excursions, amplitude):
    """Measure separate narrow excursions so distant spikes cannot masquerade as one gradual rise."""
    regions, region = [], []
    for value, x in excursions:
        if amplitude > 0 and value >= amplitude / 2:
            region.append((value, x))
        elif region:
            regions.append(region)
            region = []
    if region:
        regions.append(region)
    # The curve is cyclic; an excursion crossing the acquisition start is still one region.
    if len(regions) > 1 and excursions[0] in regions[0] and excursions[-1] in regions[-1]:
        regions = [regions[-1] + regions[0], *regions[1:-1]]
    return max((max(value for value, _ in rows) for rows in regions
                if max(x for _, x in rows) - min(x for _, x in rows) <= 0.15), default=0.0)


def edge_lean(sequence):
    """Measure signed side-edge displacement between quarter and three-quarter normalized load."""
    closed = sequence if sequence[0] == sequence[-1] else [*sequence, sequence[0]]
    crossings = []
    for level in (0.25, 0.75):
        xs = [x1 + (level - y1) * (x2 - x1) / (y2 - y1)
              for (x1, y1), (x2, y2) in zip(closed, closed[1:])
              if y1 != y2 and min(y1, y2) <= level <= max(y1, y2)]
        if not xs:
            return 0.0, 0.0
        crossings.append((min(xs), max(xs)))
    return tuple(crossings[1][index] - crossings[0][index] for index in (0, 1))


def profile_metrics(profile, sequence):
    """Measure the right upper extension, body plateaus, transition widths and localized excursions."""
    if len(profile) < 8:
        return None
    profile = sorted(profile, key=lambda row: row["x"])
    handle = []
    for row in reversed(profile):
        if min(row["high"], row["low"]) < 0.78 or abs(row["high"] - row["low"]) > 0.15:
            break
        handle.append(row)
    start = min(row["x"] for row in handle) if handle else 1.0
    body_end = max(0.35, min(0.8, start - 0.06))
    body = [row for row in profile if 0.2 <= row["x"] <= body_end]
    central = [row for row in profile if 0.2 <= row["x"] <= 0.8]
    late = [row for row in profile if 0.45 <= row["x"] <= 0.8]
    floor, ceiling = median(row["low"] for row in body), median(row["high"] for row in body)
    excursions = [(max(floor - y, y - ceiling, 0), x) for x, y in sequence]
    amplitude, position = max(excursions)
    spike_y = max((y for x, y in sequence if x == position), key=lambda y: max(floor - y, y - ceiling, 0))
    spike_positions = [x for value, x in excursions if amplitude > 0 and value >= amplitude / 2]
    pickup25, pickup75 = (transition(profile, "high", level) for level in (0.25, 0.75))
    release25, release75 = (transition(profile, "low", level) for level in (0.25, 0.75))
    left_lean, right_lean = edge_lean(sequence)
    # Fluid-pound ramps belong to the transitions, not to the main-body plateaus.
    interior = [row for row in profile
                if pickup75 + 0.03 <= row["x"] <= release25 - 0.03]
    plateau_span = interior[-1]["x"] - interior[0]["x"] if len(interior) >= 2 else 0
    return {
        "body_floor": floor, "body_ceiling": ceiling,
        "spike_position": position,
        "spike_polarity": -1 if spike_y < floor else 1,
        "pickup_lean": left_lean, "release_lean": right_lean,
        "parallel_edge_lean": min(left_lean, right_lean),
        "tubing_edge_pair": (0.06 <= left_lean <= 0.35 and 0.04 <= right_lean <= 0.35
                             and 0.3 <= right_lean / left_lean <= 3.0),
        "edge_lean_ratio": right_lean / left_lean if left_lean > 0 else 0,
        "fluid_plateau_span": plateau_span,
        "fluid_upper_flatness": (quantile([row["high"] for row in interior], 0.95)
                                 - quantile([row["high"] for row in interior], 0.05)) if interior else 1,
        "fluid_lower_flatness": (quantile([row["low"] for row in interior], 0.95)
                                 - quantile([row["low"] for row in interior], 0.05)) if interior else 1,
        "handle_start": start, "handle_length": 1 - start,
        "handle_thickness": max(abs(row["high"] - row["low"]) for row in handle) if handle else 1.0,
        "handle_level": mean((row["high"] + row["low"]) / 2 for row in handle) if handle else 0.0,
        "body_gap": median(row["high"] - row["low"] for row in body),
        "upper_flatness": quantile([row["high"] for row in body], 0.95)
                          - quantile([row["high"] for row in body], 0.05),
        "lower_flatness": quantile([row["low"] for row in body], 0.95)
                          - quantile([row["low"] for row in body], 0.05),
        "upper_slope": slope(central, "high"), "lower_slope": slope(central, "low"),
        "center_slope": (slope(central, "high") + slope(central, "low")) / 2,
        "late_upper_slope": slope(late, "high"),
        "pickup25": pickup25, "pickup75": pickup75, "pickup_width": pickup75 - pickup25,
        "release25": release25, "release75": release75, "release_width": release75 - release25,
        "localized_excursion": localized_excursion(excursions, amplitude),
        "end_spike": amplitude,
        "spike_at_end": position <= 0.15 or position >= 0.85,
        "spike_width": max(spike_positions) - min(spike_positions) if spike_positions else 0.0,
    }


def outline_structure(points, *, template=False, load_range=None):
    """Intersect the geometric contour vertically; traversal and stroke labels do not affect the result."""
    sequence = normalize_points(points, template=template, load_range=load_range)
    if not sequence:
        return None
    closed = sequence if sequence[0] == sequence[-1] else [*sequence, sequence[0]]
    profile = []
    for index in range(PROFILE_SAMPLES):
        x = (index + 0.5) / PROFILE_SAMPLES
        loads = []
        for (x1, y1), (x2, y2) in zip(closed, closed[1:]):
            if min(x1, x2) <= x <= max(x1, x2) and x1 != x2:
                loads.append(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
        if loads:
            profile.append({"x": x, "high": max(loads), "low": min(loads)})
    result = profile_metrics(profile, sequence)
    if result is not None:
        xs, ys = zip(*sequence)
        result["closure"] = math.hypot(sequence[0][0] - sequence[-1][0],
                                       (sequence[0][1] - sequence[-1][1]) / (max(ys) - min(ys)))
    return result


def directional_structure(samples, sequence):
    """Measure the handle from physical upstroke/downstroke branches without exchanging their labels."""
    rows = [{"x": up[0], "high": up[1], "low": down[1]}
            for up, down in zip(samples["upstroke"], reversed(samples["downstroke"]))]
    return profile_metrics(rows, sequence)


def structural_checks(template_id, metrics, *, load_ratio=None, closure=None):
    """Require each template's defining structure instead of accepting general outline resemblance."""
    checks = []
    def check(label, key, minimum=None, maximum=None):
        value = metrics.get(key)
        passed = (value is not None and (minimum is None or value >= minimum)
                  and (maximum is None or value <= maximum))
        expected = (f"{minimum:g}–{maximum:g}" if minimum is not None and maximum is not None
                    else f"≥ {minimum:g}" if minimum is not None else f"≤ {maximum:g}")
        checks.append({"feature": label, "actual": round(value, 4) if value is not None else None,
                       "expected": expected, "passed": passed})
    if closure is not None:
        metrics = {**metrics, "closure": closure}
        check("Closed curve", "closure", maximum=0.08)
    if template_id in LOAD_DEPENDENT:
        metrics = {**metrics, "load_ratio": load_ratio}
        check("Load range / independent reference", "load_ratio",
              minimum=1.2 if template_id == "sf_friction" else 0.03,
              maximum=3 if template_id == "sf_friction" else 0.30)
    if template_id == "dh_fluid_pound":
        check("Upper-right handle present / length", "handle_length", 0.08, 0.65)
        check("Handle thickness", "handle_thickness", maximum=0.15)
        check("Handle near upper load level", "handle_level", minimum=0.78)
        check("Main-loop branch separation", "body_gap", minimum=0.40)
        check("Abrupt load release", "release_width", maximum=0.10)
        check("Main-body plateau width", "fluid_plateau_span", minimum=0.15)
        check("Upper main-body plateau", "fluid_upper_flatness", maximum=0.15)
        check("Lower main-body plateau", "fluid_lower_flatness", maximum=0.15)
    elif template_id == "dh_full":
        check("Main-loop branch separation", "body_gap", minimum=0.40)
        checks.append({"feature": "No parallel leaning edges", "actual": metrics["tubing_edge_pair"],
                       "expected": "No", "passed": not metrics["tubing_edge_pair"]})
        check("No substantial upper-right handle", "handle_length", maximum=0.07)
        check("Flat upper plateau", "upper_flatness", maximum=0.12)
        check("Flat lower plateau", "lower_flatness", maximum=0.12)
        check("Crisp pickup", "pickup_width", maximum=0.12)
        check("Early upper plateau", "pickup75", maximum=0.20)
        check("Crisp release", "release_width", maximum=0.09)
        check("No end-of-stroke excursion", "localized_excursion", maximum=0.09)
        check("Bounded gradual end-load change", "end_spike", maximum=0.15)
    elif template_id == "dh_gas":
        check("No fluid-pound handle", "handle_length", maximum=0.07)
        check("Progressive load release", "release_width", 0.08, 0.45)
        check("No extended pickup ramp", "pickup_width", maximum=0.12)
    elif template_id == "dh_traveling_valve":
        check("Sustained upper-branch load decline", "upper_slope", maximum=-0.18)
        check("Flat lower branch", "lower_flatness", maximum=0.12)
        check("No fluid-pound handle", "handle_length", maximum=0.07)
    elif template_id == "dh_standing_valve":
        check("Gradual load pickup", "pickup_width", 0.10, 0.35)
        check("Later upper plateau is flat", "late_upper_slope", -0.15, 0.15)
        check("Crisp release", "release_width", maximum=0.09)
        check("No fluid-pound handle", "handle_length", maximum=0.07)
    elif template_id == "dh_tagging":
        check("Significant localized excursion", "end_spike", minimum=0.12)
        checks.append({"feature": "Excursion at stroke end", "actual": metrics["spike_at_end"],
                       "expected": "Yes", "passed": metrics["spike_at_end"]})
        check("Localized excursion width", "spike_width", maximum=0.15)
        check("No fluid-pound handle", "handle_length", maximum=0.07)
    elif template_id == "dh_tubing":
        check("Leaning pickup edge", "pickup_lean", 0.06, 0.35)
        check("Leaning release edge", "release_lean", 0.04, 0.35)
        check("Comparable edge lean", "edge_lean_ratio", 0.3, 3.0)
        check("No fluid-pound handle", "handle_length", maximum=0.07)
    elif template_id in ("sf_undertravel", "sf_overtravel"):
        positive = template_id == "sf_undertravel"
        for key, label in (("center_slope", "Overall loop slope"), ("upper_slope", "Upper-branch slope"),
                           ("lower_slope", "Lower-branch slope")):
            check(label, key, minimum=0.12 if positive else None, maximum=None if positive else -0.12)
    elif template_id in LOAD_DEPENDENT:
        check("No fluid-pound handle", "handle_length", maximum=0.07)
    return checks


def template_variants(template, metrics):
    """Fit bounded fluid-pound transitions and reflect tagging spikes across the two stroke reversals."""
    variants = [(template["points"], {})]
    if template["id"] == "dh_tagging":
        # End spikes may appear at either reversal and on either load branch.
        for flip_x, flip_y in ((True, False), (False, True), (True, True)):
            points = [[1 - x if flip_x else x, 1 - y if flip_y else y] for x, y in template["points"]]
            variants.append((points, {"reflect_position": flip_x, "reflect_load": flip_y}))
        return variants
    if template["id"] != "dh_fluid_pound" or metrics is None:
        return variants
    pickup_width = metrics["pickup75"] - metrics["pickup25"]
    release_width = metrics["release75"] - metrics["release25"]
    pickup_start = max(0, metrics["pickup25"] - pickup_width / 2)
    pickup_end = min(0.35, metrics["pickup75"] + pickup_width / 2)
    release_top = min(0.97, metrics["release75"] + release_width / 2)
    release_bottom = max(0.25, metrics["release25"] - release_width / 2)
    if not (0.35 <= release_top <= 0.97 and release_bottom > pickup_end + 0.10
            and 0 <= release_top - release_bottom <= 0.20):
        return variants
    points = [[0, 0], [pickup_start, 0], [pickup_end, 1], [1, 1],
              [release_top, 1], [release_bottom, 0], [0, 0]]
    variants.append((points, {"pickup_start": round(pickup_start, 4), "pickup_end": round(pickup_end, 4),
                              "release_top": round(release_top, 4), "release_bottom": round(release_bottom, 4)}))
    # A narrow high-load handle may sit slightly above the main loop; retain that separation.
    floor, ceiling = metrics["body_floor"], metrics["body_ceiling"]
    level = metrics["handle_level"]
    if (0 <= floor <= 0.15 and 0.80 <= ceiling <= 1 and ceiling - floor >= 0.65
            and ceiling <= level <= 1 and level - ceiling <= 0.15):
        points = [[0, floor], [pickup_start, floor], [pickup_end, ceiling], [release_top, ceiling],
                  [1, level], [release_top, ceiling], [release_bottom, floor], [0, floor]]
        variants.append((points, {**variants[-1][1], "body_lower": round(floor, 4),
                                  "body_upper": round(ceiling, 4), "handle_load": round(level, 4)}))
    return variants


def regional_outline_score(actual, reference, handle_start):
    """Give the main loop and right-hand extension separate geometric distances and explicit weights."""
    def region_score(left, right):
        if not left or not right:
            return 0.0
        error = (mean(min(math.dist(a, b) ** 2 for b in right) for a in left)
                 + mean(min(math.dist(b, a) ** 2 for a in left) for b in right)) / 2
        return 100 * math.exp(-error / 0.12 ** 2)
    body = region_score([p for p in actual if p[0] <= handle_start], [p for p in reference if p[0] <= handle_start])
    handle = region_score(
        [p for p in actual if p[0] >= handle_start], [p for p in reference if p[0] >= handle_start],
    )
    return {"score": round(0.65 * body + 0.35 * handle, 1),
            "body_score": round(body, 1), "handle_score": round(handle, 1)}



def tagging_outline_score(points, reference):
    """Compare the clipped main loop and a localized spike independently in a shared body load scale."""
    actual_metrics, reference_metrics = outline_structure(points), outline_structure(reference)
    if actual_metrics is None or reference_metrics is None:
        return {"score": 0.0, "body_score": 0.0, "spike_score": 0.0}

    def main_loop(curve, metrics):
        floor, ceiling = metrics["body_floor"], metrics["body_ceiling"]
        span = ceiling - floor
        if span <= 0:
            return []
        return [(x, max(0, min(1, (y - floor) / span))) for x, y in normalize_points(curve)]

    # Resampling clipped contours removes vertex-density effects while retaining pickup/release geometry.
    from umai.rod_pump_health.matching import normalized_outline
    left = normalized_outline(main_loop(points, actual_metrics), template=True)
    right = normalized_outline(main_loop(reference, reference_metrics), template=True)
    if not left or not right:
        return {"score": 0.0, "body_score": 0.0, "spike_score": 0.0}
    error = (mean(min(math.dist(a, b) ** 2 for b in right) for a in left)
             + mean(min(math.dist(b, a) ** 2 for a in left) for b in right)) / 2
    body = 100 * math.exp(-error / 0.12 ** 2)
    amplitude = actual_metrics["end_spike"] / max(actual_metrics["body_ceiling"] - actual_metrics["body_floor"], 1e-9)
    expected = reference_metrics["end_spike"] / max(
        reference_metrics["body_ceiling"] - reference_metrics["body_floor"], 1e-9)
    spike_error = ((amplitude - expected) / 0.20) ** 2
    spike_error += ((actual_metrics["spike_width"] - reference_metrics["spike_width"]) / 0.10) ** 2
    spike_error += ((actual_metrics["spike_position"] - reference_metrics["spike_position"]) / 0.15) ** 2
    spike = 100 * math.exp(-spike_error / 3) if actual_metrics["spike_polarity"] == reference_metrics["spike_polarity"] else 0
    return {"score": round(0.65 * body + 0.35 * spike, 1),
            "body_score": round(body, 1), "spike_score": round(spike, 1)}
