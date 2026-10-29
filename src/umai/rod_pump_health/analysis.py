"""Single-card rod-pump health review: measured geometry plus coordinates sent as text."""

from __future__ import annotations

import json
import math
import logging
import re
from datetime import datetime
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError

logger = logging.getLogger(__name__)

VERSION = "rod-pump-health-analysis-v12"


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: float
    load: float


class Card(BaseModel):
    """The last eligible current card selected by UMetriqs, with its calculated downhole card."""

    model_config = ConfigDict(extra="forbid")
    archive_id: int | None = Field(default=None, gt=0)
    reference: str = Field(min_length=1)
    roles: list[Literal["recent"]] = Field(default_factory=lambda: ["recent"])
    pull_timestamp: int | None = None
    pumping_window: dict = Field(default_factory=dict)
    timestamp: datetime
    source_card_id: int | None = None
    header: dict = Field(default_factory=dict)
    surface: list[Point] = Field(max_length=2048)
    downhole: list[Point] = Field(default_factory=list, max_length=2048)
    downhole_summary: dict = Field(default_factory=dict)
    period_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    warnings: list[str] = Field(default_factory=list, max_length=30)
    point_order: Literal["unknown", "sam1_reverse_time"] = "unknown"
    downhole_error: str | None = Field(default=None, max_length=2000)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_id: int = Field(gt=0)
    well_id: int = Field(gt=0)
    card: Card
    calculation_snapshot: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def aware_timestamp(self):
        if self.card.timestamp.tzinfo is None:
            raise ValueError("Card timestamps require a timezone.")
        return self


class Finding(BaseModel):
    """One diagnosis for the reviewed card, as returned by the model."""

    model_config = ConfigDict(extra="ignore")
    side: Literal["surface", "downhole"]
    title: str = Field(min_length=1, max_length=200)
    reasoning: str = Field(max_length=4000)
    actions: str = Field(max_length=4000)
    confidence: Literal["low", "medium", "high"]


class Assessment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    health_status: Literal["healthy", "watch", "problem"]
    summary: str = Field(min_length=1, max_length=1500)
    findings: list[Finding] = Field(default_factory=list, max_length=2)
    limitations: list[str] = Field(default_factory=list)


# Strict JSON schema for the provider; array sizes are enforced after parsing instead.
ASSESSMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["health_status", "summary", "findings", "limitations"],
    "properties": {
        "health_status": {"type": "string", "enum": ["healthy", "watch", "problem"]},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["side", "title", "reasoning", "actions", "confidence"],
                "properties": {
                    "side": {"type": "string", "enum": ["surface", "downhole"]},
                    "title": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "actions": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}


def features(points: list[Point], point_order: str = "unknown") -> dict:
    """Measure physical geometry; preserve offset and flag unreliable open curves."""
    if len(points) < 4:
        return {"status": "invalid", "reason": "Fewer than four numeric points."}
    x, y = [p.position for p in points], [p.load for p in points]
    span, spread = max(x) - min(x), max(y) - min(y)
    if span <= 0:
        return {"status": "invalid", "reason": "Zero stroke span."}
    closure = math.hypot((x[0] - x[-1]) / span, (y[0] - y[-1]) / (spread or 1))
    area = abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i] for i in range(len(x)))) / 2
    # Geometric high/low branches are safe even when time direction is unknown.
    bins = []
    for index in range(10):
        loads = [
            load for pos, load in zip(x, y) if index / 10 <= (pos - min(x)) / span < (index + 1) / 10
        ]
        if len(loads) >= 2:
            bins.append((index / 10 + 0.05, min(loads), max(loads)))
    thickness = median([hi - lo for _, lo, hi in bins]) if bins else None
    centers = [(pos, (lo + hi) / 2) for pos, lo, hi in bins]
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for i, a in enumerate(centers) for b in centers[i + 1 :]]
    result = {
        "status": "valid" if closure <= 0.05 and spread > 0 else "questionable",
        "orientation": "unknown",
        "stroke_in": span,
        "peak_load_lb": max(y),
        "min_load_lb": min(y),
        "load_range_lb": spread,
        "area_in_lb": area,
        "median_branch_separation_lb": thickness,
        "geometric_slope_lb_per_stroke": median(slopes) if slopes else None,
        "closure_ratio": closure,
        "load_offset_lb": median(y),
        "normalized_area": area / (span * spread) if spread else 0,
    }
    result["spike_ratio"] = max(
        abs(y[(i + 1) % len(y)] - 2 * y[i] + y[i - 1]) for i in range(len(y))
    ) / (spread or 1)
    if point_order == "sam1_reverse_time":
        result.update(branch_features(points, span, spread))
    return result


def branch_features(points, span, spread):
    """SAM1 reverse-time provenance plus monotonicity verifies displacement branches."""
    sequence = list(reversed(points))
    start = min(range(len(sequence)), key=lambda i: sequence[i].position)
    sequence = sequence[start:] + sequence[:start]
    sequence.append(sequence[0])
    turn = max(range(len(sequence)), key=lambda i: sequence[i].position)
    up, down = sequence[: turn + 1], sequence[turn:]
    if min(len(up), len(down)) < 4:
        return {}
    for branch, sign in ((up, 1), (down, -1)):
        if (
            sum(sign * (b.position - a.position) >= -span * 0.002 for a, b in zip(branch, branch[1:]))
            / (len(branch) - 1)
            < 0.8
        ):
            return {}
    low = min(p.load for p in points)
    xmin = min(p.position for p in points)

    def cross(branch, threshold):
        for a, b in zip(branch, branch[1:]):
            target = low + spread * threshold
            if (a.load - target) * (b.load - target) <= 0 and a.load != b.load:
                return (
                    (
                        a.position
                        + (target - a.load) / (b.load - a.load) * (b.position - a.position)
                        - xmin
                    )
                    / span
                    * 100
                )
        return None

    pick25, pick75 = cross(up, 0.25), cross(up, 0.75)
    release75, release25 = cross(down, 0.75), cross(down, 0.25)
    result = {"orientation": "verified_sam1_reverse_time"}
    for key, value in [
        ("pickup_position_percent", pick75),
        ("release_position_percent", release75),
        (
            "upstroke_transition_width_percent",
            None if pick25 is None or pick75 is None else abs(pick75 - pick25),
        ),
        (
            "downstroke_transition_width_percent",
            None if release25 is None or release75 is None else abs(release75 - release25),
        ),
    ]:
        if value is not None:
            result[key] = value
    for name, branch in [("upstroke", up), ("downstroke", down)]:
        interior = [p for p in branch if 0.2 <= (p.position - xmin) / span <= 0.8]
        slopes = [
            (b.load - a.load) / ((b.position - a.position) / span)
            for i, a in enumerate(interior)
            for b in interior[i + 1 :]
            if abs(b.position - a.position) > span * 0.02
        ]
        if slopes:
            result[name + "_slope_lb_per_stroke"] = median(slopes)
    return result



def reduce_points(pairs: list[list[float]], limit: int | None) -> list[list[float]]:
    """Largest-triangle-three-buckets down-sampling that keeps corners and peaks.

    Both axes are scaled to 0..1 first so inches and pounds weigh equally; the first and
    last points are always kept.
    """
    if limit is None or limit < 3 or len(pairs) <= limit:
        return list(pairs)
    xs, ys = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    x_span, y_span = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
    scaled = [((x - min(xs)) / x_span, (y - min(ys)) / y_span) for x, y in pairs]
    bucket = (len(pairs) - 2) / (limit - 2)
    kept, anchor = [0], 0
    for index in range(limit - 2):
        start, end = int(index * bucket) + 1, int((index + 1) * bucket) + 1
        following = scaled[end : min(int((index + 2) * bucket) + 1, len(pairs))] or [scaled[-1]]
        next_x = sum(point[0] for point in following) / len(following)
        next_y = sum(point[1] for point in following) / len(following)
        ax, ay = scaled[anchor]
        anchor = max(
            range(start, end),
            key=lambda j: abs((ax - next_x) * (scaled[j][1] - ay) - (ax - scaled[j][0]) * (next_y - ay)),
        )
        kept.append(anchor)
    kept.append(len(pairs) - 1)
    return [pairs[index] for index in kept]


def coordinates(points: list[Point], point_order: str, limit: int | None = None) -> list[list[float]]:
    """Compact [position_in, load_lb] pairs in time order, down-sampled to at most limit points."""
    ordered = list(reversed(points)) if point_order == "sam1_reverse_time" else points
    return reduce_points([[round(point.position, 1), round(point.load)] for point in ordered], limit)


# Surface load check fields the LLM gets; the trend, comparison card and hint are for the page.
SURFACE_LOAD_EVIDENCE = (
    "status", "load_range", "normal", "range_ratio", "since", "cards", "last_normal_card_at",
    "card_level_pct", "recovering", "after_shutdown",
)


def evidence(
    card: Card, measurements: dict, point_limits: dict | None = None, surface_load: dict | None = None
) -> dict:
    """Build the text-only packet: header, measurements and coordinates for both card sides."""
    limits = point_limits or {}
    header = {
        key: value
        for key, value in card.header.items()
        if key != "stroke_period" and isinstance(value, (int, float, str)) and value is not None
    }
    controller = {
        **header,
        "stroke_period_s": card.period_s,
        "strokes_per_minute": round(60 / card.period_s, 2) if card.period_s else None,
    }
    order = (
        "time order through one stroke"
        if card.point_order == "sam1_reverse_time"
        else "controller order; time direction unverified"
    )
    downhole = {"measurements": measurements["downhole"]}
    if card.downhole:
        downhole["calculation"] = card.downhole_summary
        downhole["points"] = coordinates(card.downhole, card.point_order, limits.get("downhole"))
    else:
        downhole["unavailable"] = card.downhole_error or "No calculated downhole card."
    surface = {
        "measurements": measurements["surface"],
        "points": coordinates(card.surface, card.point_order, limits.get("surface")),
    }
    if surface_load:
        surface["load_history"] = {
            key: surface_load[key] for key in SURFACE_LOAD_EVIDENCE if key in surface_load
        }
    return {
        "card_reference": card.reference,
        "captured_at": card.timestamp.isoformat(),
        "pumping_confirmed": "Well state was pumping from 10 minutes before to 5 minutes after capture.",
        "controller_header": controller,
        "point_format": (
            f"[position_in, load_lb] pairs in {order}, down-sampled to keep the card shape; "
            "measurements use every controller point"
        ),
        "surface": surface,
        "downhole": downhole,
        "calculation_warnings": card.warnings,
    }


def review_failure_message(exception, stage):
    """Report actionable failures without logging authentication tokens."""
    if isinstance(exception, ValidationError):
        issues = exception.errors(include_input=False, include_url=False)
        fields = [".".join(map(str, issue["loc"])) + ": " + issue["type"] for issue in issues[:5]]
        message = "Invalid structured LLM response: " + "; ".join(fields)
    else:
        message = str(exception) or type(exception).__name__
    message = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "<redacted key>", message)
    message = re.sub(r"(?i)Bearer\s+[^\s'\"]+", "Bearer <redacted>", message)
    return f"{stage}: {type(exception).__name__}: " + " ".join(message.split())[:600]


def measure(card: Card) -> tuple[dict, list[str]]:
    """Measure both full curves and list the data-quality limitations they show."""
    measurements = {}
    limitations = []
    for side in ("surface", "downhole"):
        points = getattr(card, side)
        if points:
            measurements[side] = features(points, card.point_order)
        else:
            reason = card.downhole_error if side == "downhole" else "No surface points."
            measurements[side] = {"status": "invalid", "reason": reason or "No points."}
        if measurements[side]["status"] == "questionable":
            limitations.append(side + ": the curve does not close; measurements are approximate.")
    if not card.downhole:
        limitations.append("downhole: calculated card unavailable; surface card reviewed alone.")
    return measurements, limitations


def analyze(
    packet: AnalysisRequest, reviewer, point_limits: dict | None = None, surface_load: dict | None = None
) -> dict:
    """Measure the full card, send down-sampled coordinates as text and keep the assessment."""
    card = packet.card
    measurements, limitations = measure(card)
    findings = []
    health_status = None
    summary = None
    review_status = "unavailable"
    review_error = None
    token_usage = None
    if any(measurements[side]["status"] != "invalid" for side in measurements):
        content = [{"type": "text", "text": json.dumps(evidence(card, measurements, point_limits, surface_load), default=str)}]
        stage = "LLM request"
        try:
            raw_review = reviewer(content)
            stage = "Response handling"
            if not isinstance(raw_review, dict):
                raise ValueError("LLM response must be a JSON object.")
            token_usage = raw_review.get("token_usage")
            raw_findings = raw_review.get("findings")
            if isinstance(raw_findings, list):
                raw_review = {**raw_review, "findings": raw_findings[:2]}
            review = Assessment.model_validate(raw_review)
            health_status = review.health_status
            summary = " ".join(review.summary.split())
            limitations.extend(" ".join(str(value).split()) for value in review.limitations)
            for finding in review.findings:
                findings.append(
                    {
                        "side": finding.side,
                        "kind": "shape",
                        "scope": "last_eligible_card",
                        "title": " ".join(finding.title.split()),
                        "reasoning": " ".join(finding.reasoning.split()),
                        "actions": " ".join(finding.actions.split()),
                        "confidence": finding.confidence,
                        "card_ids": [card.reference],
                    }
                )
            review_status = "complete"
        except Exception as exception:
            review_error = review_failure_message(exception, stage)
            logger.warning("Rod pump health review failed well_id=%s: %s", packet.well_id, review_error)
            findings = []
            review_status = "failed"
            limitations.append("AI review failed. No healthy verdict is inferred.")
    else:
        review_error = "No usable card curve; LLM was not called."
        logger.warning(
            "Rod pump health review unavailable well_id=%s: %s", packet.well_id, review_error
        )
    return {
        "review_error": review_error,
        "analyzer_version": VERSION,
        "status": "complete" if review_status == "complete" and not limitations else "partial",
        "review_status": review_status,
        "diagnosis_status": (
            "candidates"
            if findings
            else "healthy"
            if review_status == "complete"
            else "unavailable"
        ),
        "health_status": health_status,
        "summary": summary,
        "measurements": measurements,
        "findings": findings,
        "token_usage": token_usage,
        "limitations": limitations,
    }


SYSTEM_PROMPT = (
    "You assess the health of a rod-pumped oil well from one dynamometer card. The well runs a "
    "conventional beam pumping unit with a fixed-speed motor and no VFD, controlled by a Lufkin SAM1 "
    "well manager that stops the unit on pump-off and idle time instead of changing speed. The card "
    "is the controller's most recent card captured while the well was confirmed pumping. "
    "The input JSON has the SAM1 card header, the measured surface card and the downhole pump card "
    "calculated from it with the wave equation, each as [position_in, load_lb] coordinates with "
    "measurements derived from the full curves; the coordinates are down-sampled, so take peak and "
    "minimum loads from the measurements. Base the diagnosis mainly on the downhole card shape "
    "(pump fillage, fluid pound, gas interference, gas lock, traveling or standing valve leak, worn "
    "pump, tagging, unanchored tubing, sticking, parted rods) and use the surface card for load "
    "range, rod loading and data quality. "
    "surface.load_history, when present, compares this card's surface load range (peak minus "
    "minimum) with the median of the well's recent good cards: collapse means the range fell "
    "below half of normal or under 3,000 lb, load_dropping below 80%. A collapse while pumping "
    "means the pump lost its fluid load (tubing leak, rod part, unseated or worn pump, pump not "
    "lifting); weigh it with the downhole card, and note when it started right after a shutdown, "
    "since the fluid column may still be refilling. "
    "Return health_status: healthy for a normal full or near-full pump card that needs no action; "
    "watch for a developing or minor issue worth monitoring or a setpoint review; problem for a "
    "clear fault that needs field action. Give a one or two sentence summary for an operator. "
    "Report up to two findings, none when healthy. Each finding names the side whose shape shows "
    "it, a short title, one reasoning paragraph citing the card features and explaining the "
    "confidence, one actions paragraph with practical field steps and relevant SAM1 setpoint "
    "changes, and confidence low, medium or high based on how clear the defining features are and "
    "how plausible the alternatives are. Separate suspected causes from confirmed faults and give "
    "numeric values only when the data supports them. If data quality prevents a reliable "
    "assessment, say so in limitations and the summary rather than guessing."
)


def _response_text(response) -> str:
    """Join text parts when a provider returns content blocks instead of a string."""
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content


def _structured_reviewer(model, settings, system_prompt):
    """Invoke a bound chat model and return its JSON object with normalized token usage."""
    from langchain_core.messages import HumanMessage, SystemMessage

    def review(content):
        try:
            response = model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=content)]
            )
            result = json.loads(_response_text(response))
            if not isinstance(result, dict):
                raise ValueError("LLM response must be a JSON object.")
            token_usage = _response_token_usage(response)
            if token_usage is not None:
                result["token_usage"] = token_usage
            return result
        except Exception as exception:
            if settings.llm_api_key and settings.llm_api_key in str(exception):
                message = str(exception).replace(settings.llm_api_key, "<redacted key>")
                raise RuntimeError(message) from None
            raise

    return review


def health_reviewer(settings, effort=None):
    """Use the dedicated health-check model with schema-constrained JSON and a reasoning effort
    (the configured one unless another is given, e.g. the routine effort)."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.rod_pump_health_llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        reasoning_effort=effort or settings.rod_pump_health_llm_effort,
        max_tokens=16000,
        timeout=300,
        max_retries=0,
    )
    structured = model.bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rod_pump_health_assessment",
                "strict": True,
                "schema": ASSESSMENT_SCHEMA,
            },
        }
    )
    return _structured_reviewer(structured, settings, SYSTEM_PROMPT)


def llm_reviewer(settings, system_prompt):
    """Shared-model JSON reviewer kept for the disabled parameter-trend and conclusion stages."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=150,
        max_retries=0,
    )
    structured = model.bind(response_format={"type": "json_object"})
    return _structured_reviewer(structured, settings, system_prompt)


def _response_token_usage(response):
    """Normalize LangChain or OpenAI usage metadata for persistence and cost estimates."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict) or not usage:
        metadata = getattr(response, "response_metadata", {})
        usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    details = {}
    for target, keys in {
        "input": ("input_token_details", "prompt_tokens_details"),
        "output": ("output_token_details", "completion_tokens_details"),
    }.items():
        value = next((usage[key] for key in keys if usage.get(key)), None)
        if value is not None:
            details[target] = value
    if details:
        normalized["details"] = details
    return normalized
