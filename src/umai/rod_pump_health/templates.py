"""Versioned, illustrative references; these sketches are not field measurements."""

VERSION = "dynograph-templates-v3"
PDF_SOURCE = "Downhole Dynamometer Card Guide and Card Review, supplied October 2026, section 2"
ECHOMETER = "https://www.echometer.com/technical-papers"


def template(key, side, title, points, features, alternatives, source):
    return dict(
        id=key,
        side=side,
        title=title,
        points=points,
        features=features,
        alternatives=alternatives,
        source=source,
    )


TEMPLATES = [
    template(
        "dh_full",
        "downhole",
        "Healthy pump",
        [[0, 0], [0.06, 1], [0.94, 1], [1, 0], [0, 0]],
        "Flat top and bottom with crisp corners; valves open and close cleanly.",
        "A healthy shape match is a similarity assessment, not confirmation of every operating condition.",
        "User-supplied Healthy pump reference image",
    ),
    template(
        "dh_fluid_pound",
        "downhole",
        "Fluid-pound pattern",
        [[0, 0], [0, 1], [1, 1], [0.62, 1], [0.6, 0], [0, 0]],
        "Load remains high into downstroke then falls abruptly.",
        "Gas interference, mixed gas/liquid, stroke orientation or calculation distortion.",
        PDF_SOURCE,
    ),
    template(
        "dh_gas",
        "downhole",
        "Gas-interference pattern",
        [
            [0, 0],
            [0, 1],
            [0.96, 1],
            [1, 0.8],
            [0.98, 0.55],
            [0.91, 0.3],
            [0.8, 0.1],
            [0.65, 0],
            [0, 0],
        ],
        "Rounded, gradual downstroke release rather than abrupt collapse.",
        "Fluid pound, smoothing or mixed conditions.",
        PDF_SOURCE,
    ),
    template(
        "dh_gas_lock",
        "downhole",
        "Gas-lock / low fluid-load pattern",
        [[0, 0.45], [0.08, 0.56], [0.95, 0.56], [1, 0.5], [0.95, 0.44], [0.08, 0.44], [0, 0.45]],
        "Very small branch separation relative to the expected full-load card shape.",
        (
            "Valve stuck open, unloading, rod failure or measurement/calculation issues; "
            "not uniquely gas lock."
        ),
        PDF_SOURCE,
    ),
    template(
        "dh_traveling_valve",
        "downhole",
        "Traveling-valve-leak pattern",
        [[0, 0], [0, 1], [1, 0.68], [1, 0], [0, 0]],
        "Upper branch loses load through the upstroke.",
        "Gas, changing fluid load or calculation distortion.",
        PDF_SOURCE,
    ),
    template(
        "dh_standing_valve",
        "downhole",
        "Standing-valve-leak pattern",
        [[0, 0], [0.35, 1], [1, 1], [1, 0], [0, 0]],
        "Delayed or gradual load pickup on the upstroke.",
        "Gas compression, fluid compressibility or calculation smoothing.",
        PDF_SOURCE,
    ),
    template(
        "dh_tagging",
        "downhole",
        "Tagging pattern",
        [[0, 0], [0, 1], [1, 1], [1, 0], [0.12, 0], [0.02, -0.25], [0, 0]],
        "Localized end-of-stroke spike.",
        "Rod ringing, sampling or reversal artifacts.",
        PDF_SOURCE,
    ),
    template(
        "dh_tubing",
        "downhole",
        "Tubing-movement pattern",
        [[0, 0], [0.3, 1], [1, 1], [0.7, 0], [0, 0]],
        "Sheared / leaning pump card.",
        "Rod dynamics or incorrect tubing correction/settings.",
        PDF_SOURCE,
    ),
    template(
        "sf_undertravel",
        "surface",
        "Undertravel slope",
        [[0, 0], [0.2, 0.65], [1, 1], [0.8, 0.35], [0, 0]],
        "Overall positive left-to-right slope.",
        (
            "Large fluid load, rod stretch, friction, speed or equipment geometry; "
            "compare calculated pump stroke."
        ),
        ECHOMETER + " — Over-Travel Can Occur on Both the Upstroke and Down Stroke (2017)",
    ),
    template(
        "sf_overtravel",
        "surface",
        "Overtravel slope",
        [[0, 0.35], [0.2, 1], [1, 0.65], [0.8, 0], [0, 0.35]],
        "Overall negative left-to-right slope.",
        "Rod dynamics, speed, unloading, valve conditions or rod failure; not a unique diagnosis.",
        ECHOMETER + " — Over-Travel Can Occur on Both the Upstroke and Down Stroke (2017)",
    ),
    template(
        "sf_friction",
        "surface",
        "Increased friction / load separation",
        [[0, 0], [0.08, 1], [1, 1], [0.92, 0], [0, 0]],
        "Expanded branch separation / work relative to same-well history.",
        "Fluid load, operating speed, stroke changes or calibration also change area.",
        ECHOMETER + " — Examples of Forces Not Accounted For By the Wave Equation (2018)",
    ),
    template(
        "sf_unloading",
        "surface",
        "Unloading pattern",
        [[0, 0.45], [0.15, 0.58], [1, 0.53], [0.85, 0.42], [0, 0.45]],
        "Reduced branch separation and loading relative to earlier cards.",
        (
            "Valve stuck open, pump unseated, rod part, flowing conditions or "
            "calibration; require corroboration."
        ),
        ECHOMETER
        + " — Reference Load Lines Aid in Analysis of the Downhole Dynamometer Pump Card (2009)",
    ),
]
