"""Measured card changes and bounded multimodal review, without health scores."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import logging
import re
from datetime import datetime
from statistics import median
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError
from umai.rod_pump_health.templates import TEMPLATES, VERSION as TEMPLATE_VERSION

logger = logging.getLogger(__name__)

VERSION = "rod-pump-health-analysis-v10"


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: float
    load: float


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid")
    archive_id: int | None = Field(default=None, gt=0)
    reference: str = ""
    roles: list[Literal["baseline", "recent"]] = Field(default_factory=lambda: ["baseline"])
    pull_timestamp: int | None = None
    pumping_window: dict = Field(default_factory=dict)
    timestamp: datetime
    source_card_id: int | None = None
    header: dict = Field(default_factory=dict)
    surface: list[Point] = Field(max_length=2048)
    downhole: list[Point] = Field(default_factory=list, max_length=2048)
    period_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    warnings: list[str] = Field(default_factory=list, max_length=30)
    point_order: Literal["unknown", "sam1_reverse_time"] = "unknown"
    downhole_error: str | None = Field(default=None, max_length=2000)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_id: int = Field(gt=0)
    well_id: int = Field(gt=0)
    cards: list[Card] = Field(min_length=1, max_length=15)
    calculation_snapshot: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_history(self):
        times = [card.timestamp for card in self.cards]
        if any(t.tzinfo is None for t in times):
            raise ValueError("Card timestamps require a timezone.")
        for card in self.cards:
            if not card.reference:
                if card.archive_id is None:
                    raise ValueError("Card source reference is required.")
                card.reference = f"archive:{card.archive_id}"
            if card.reference.startswith("archive:") and card.reference != f"archive:{card.archive_id}":
                raise ValueError("Archive reference does not match its database ID.")
        if len({card.reference for card in self.cards}) != len(self.cards):
            raise ValueError("Duplicate card references.")
        baseline = [card.timestamp for card in self.cards if "baseline" in card.roles]
        if len(set(baseline)) != len(baseline):
            raise ValueError("Duplicate baseline capture timestamps.")
        if len(baseline) > 14 or sum("recent" in card.roles for card in self.cards) > 1:
            raise ValueError("At most fourteen baseline cards and one recent card are allowed.")
        if baseline and (max(baseline) - min(baseline)).total_seconds() > 14 * 86400:
            raise ValueError("Card window exceeds 14 days.")
        return self


class Finding(BaseModel):
    """Keep display structure while tolerating omitted optional fields in the first version."""

    model_config = ConfigDict(extra="ignore")
    side: Literal["surface", "downhole"]
    kind: Literal["shape", "trend"] = "shape"
    scope: Literal["last_eligible_card", "historical"]
    title: str = Field(default="Observation", max_length=120)
    suspected_problem: str = ""
    template_ids: list[str] = Field(default_factory=list)
    card_ids: list[str | int] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=1800)
    actions: str = Field(default="", max_length=1800)
    confidence: Literal["low", "medium", "high"] | None = None
    supporting_features: list[str] = Field(default_factory=list)
    contradicting_features: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    matching_features: list[str] = Field(default_factory=list)
    mismatching_features: list[str] = Field(default_factory=list)
    match_strength: Literal["strong", "moderate", "weak"] = "weak"
    evidence_keys: list[str] = Field(default_factory=list)


class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    findings: list = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


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


def trends(rows: list[dict]) -> dict:
    """Theil–Sen slopes use elapsed days, not row indices or interpolated gaps."""
    usable = [row for row in rows if row["features"]["status"] != "invalid"]
    if len(usable) < 3:
        return {"status": "insufficient_history", "valid_cards": len(usable), "metrics": {}}
    output = {}
    for key in (
        "stroke_in",
        "peak_load_lb",
        "min_load_lb",
        "load_range_lb",
        "area_in_lb",
        "median_branch_separation_lb",
        "geometric_slope_lb_per_stroke",
        "load_offset_lb",
        "pickup_position_percent",
        "release_position_percent",
        "upstroke_transition_width_percent",
        "downstroke_transition_width_percent",
        "spike_ratio",
    ):
        values = [
            (datetime.fromisoformat(row["timestamp"]), row["features"].get(key))
            for row in usable
            if row["features"].get(key) is not None
        ]
        if len(values) < 3:
            continue
        slopes = [
            (b[1] - a[1]) / ((b[0] - a[0]).total_seconds() / 86400)
            for i, a in enumerate(values)
            for b in values[i + 1 :]
            if b[0] > a[0]
        ]
        group = min(3, max(1, len(values) // 3))
        first = median([v for _, v in values[:group]])
        last = median([v for _, v in values[-group:]])
        output[key] = {
            "slope_per_day": median(slopes),
            "early_median": first,
            "recent_median": last,
            "absolute_change": last - first,
            "percent_change": (last - first) / abs(first) * 100 if abs(first) > 1e-9 else None,
            "first_at": values[0][0].isoformat(),
            "last_at": values[-1][0].isoformat(),
        }
    span_days = (
        datetime.fromisoformat(usable[-1]["timestamp"]) - datetime.fromisoformat(usable[0]["timestamp"])
    ).total_seconds() / 86400
    return {
        "status": "available",
        "valid_cards": len(usable),
        "metrics": output,
        "preliminary": len(usable) < 7 or span_days < 10,
    }


def render(
    curves: list[tuple[str, list]],
    title: str,
    template: bool = False,
    reference_curves: list | None = None,
) -> str:
    """Use common physical axes; templates retain their illustrative coordinate scale."""
    img = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(img)
    pts = [p for _, curve in (reference_curves or curves) for p in curve]
    if not pts:
        return ""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if template:
        xmin, xmax, ymin, ymax = 0, 1, min(0, ymin), max(1, ymax)
    dx, dy = xmax - xmin or 1, ymax - ymin or 1
    draw.text((40, 15), title, fill="black")
    draw.line([(90, 70), (90, 535), (950, 535)], fill="black", width=2)
    for tick in range(6):
        draw.text((90 + tick * 172, 545), f"{xmin+dx*tick/5:.2f}", fill="black")
        draw.text(
            (5, 530 - tick * 90),
            format(ymin + dy * tick / 5, ".2f" if template else ".0f"),
            fill="black",
        )
    draw.text((330, 580), "Position (inches for field cards; unitless for templates)", fill="black")
    draw.text((5, 40), "Illustrative load (unitless)" if template else "Load (lb)", fill="black")
    colors = ["#94a3b8", "#64748b", "#0ea5e9", "#f97316", "#7c3aed", "#15803d"]
    for index, (label, curve) in enumerate(curves):
        if len(curve) > 1:
            draw.line(
                [(90 + (x - xmin) / dx * 850, 530 - (y - ymin) / dy * 450) for x, y in curve],
                fill=colors[index % 6],
                width=3,
            )
            draw.text(
                (95 + (index % 4) * 220, 610 + (index // 4) * 12), label[:30], fill=colors[index % 6]
            )
    stream = io.BytesIO()
    img.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def evidence_catalog(measurements: dict) -> dict:
    """Expose exact numerical facts the reviewer may reference without inventing values."""
    facts = {}
    for side, data in measurements.items():
        for row in data["cards"]:
            for key, value in row["features"].items():
                if isinstance(value, (int, float)):
                    facts[f'{side}.card.{row["reference"]}.{key}'] = value
        for key, values in data["trends"]["metrics"].items():
            for metric, value in values.items():
                if isinstance(value, (int, float)):
                    facts[f"{side}.trend.{key}.{metric}"] = value
    return facts


def combined_panels(left, right, title):
    """Combine two physical-axis plots into one image, including unavailable-data placeholders."""
    canvas = Image.new("RGB", (2000, 750), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 15), title, fill="black")
    for index, encoded in enumerate((left, right)):
        if encoded:
            panel = Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
            canvas.paste(panel, (index * 1000, 50))
        else:
            draw.text((index * 1000 + 50, 150), "No usable card available for this side", fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def comparison_sheet(cards, recent_ids, side):
    """Show one recent field card beside each normalized template for its side."""
    recent_card = next(
        (
            card for card in reversed(cards)
            if card.reference in recent_ids[side] and getattr(card, side)
        ),
        None,
    )
    rows = [template for template in TEMPLATES if template["side"] == side]
    row_height = 240
    image = Image.new("RGB", (1600, 100 + len(rows) * row_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), f"Pairwise {side} comparisons: recent card vs one template", fill="black")

    def normalized(points, point_order="unknown"):
        if len(points) < 2:
            return []
        points = list(reversed(points)) if point_order == "sam1_reverse_time" else list(points)
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
        if dx <= 0 or dy <= 0:
            return []
        start = min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
        points = points[start:] + points[:start]
        xmin = min(xs)
        reverse = [points[0], *reversed(points[1:])]

        def first_branch_load(sequence):
            return next(
                (y for x, y in sequence[1:] if x > xmin + dx * 0.002),
                min(ys),
            )

        if first_branch_load(reverse) > first_branch_load(points):
            points = reverse
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        xmin, ymin = min(xs), min(ys)
        dx, dy = max(xs) - xmin, max(ys) - ymin
        return [((x - min(xs)) / dx, (y - min(ys)) / dy) for x, y in points]

    def panel(left, top, label, points):
        right, bottom = left + 740, top + 205
        draw.text((left, top), label, fill="black")
        draw.line([(left + 35, top + 30), (left + 35, bottom), (right, bottom)], fill="#475569")
        draw.text((left + 38, bottom + 5), "Position 0–1", fill="#475569")
        draw.text((left + 38, top + 32), "Load 0–1", fill="#475569")
        if len(points) > 1:
            curve = [
                (left + 35 + x * (right - left - 45), bottom - y * (bottom - top - 45))
                for x, y in points
            ]
            draw.line(curve, fill="#0f766e", width=3)

    for index, template in enumerate(rows):
        top = 60 + index * row_height
        card = recent_card
        points = (
            [(point.position, point.load) for point in getattr(card, side)]
            if card else []
        )
        panel(
            24,
            top,
            f"{side.title()} field card: {card.reference if card else 'unavailable'}",
            normalized(points, card.point_order if card else "unknown"),
        )
        draw.text(
            (800, top - 18),
            f"Compare only this pair: {template['id']} | {template['title']}",
            fill="#7c3aed",
        )
        panel(800, top, f"{side.title()} template: {template['id']}", normalized(template["points"]))

    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def review_failure_message(exception, stage):
    """Report actionable failures without logging image blobs or authentication tokens."""
    if isinstance(exception, ValidationError):
        issues = exception.errors(include_input=False, include_url=False)
        fields = [".".join(map(str, issue["loc"])) + ": " + issue["type"] for issue in issues[:5]]
        message = "Invalid structured LLM response: " + "; ".join(fields)
    else:
        message = str(exception) or type(exception).__name__
    message = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image omitted>", message)
    message = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "<redacted key>", message)
    message = re.sub(r"(?i)Bearer\s+[^\s'\"]+", "Bearer <redacted>", message)
    return f"{stage}: {type(exception).__name__}: " + " ".join(message.split())[:600]


def analyze(packet: AnalysisRequest, reviewer) -> dict:
    """Send history and pairwise current/template images plus measured JSON."""
    cards = sorted(packet.cards, key=lambda c: c.timestamp)
    measurements = {}
    historical_images = []
    reviewed_ids = {}
    recent_ids = {}
    limitations = []
    for side in ("surface", "downhole"):
        rows = []
        curves = []
        for card in cards:
            points = getattr(card, side)
            if not points:
                continue
            measured = features(points, card.point_order)
            rows.append(
                {
                    "archive_id": card.archive_id,
                    "reference": card.reference,
                    "roles": card.roles,
                    "timestamp": card.timestamp.isoformat(),
                    "features": measured,
                    "period_s": card.period_s,
                }
            )
            if measured["status"] != "invalid":
                curves.append(
                    (
                        card.reference + " " + card.timestamp.date().isoformat(),
                        [[p.position, p.load] for p in points],
                    )
                )
        historical = [row for row in rows if "baseline" in row["roles"]]
        recent_ids[side] = {
            row["reference"]
            for row in rows
            if "recent" in row["roles"] and row["features"]["status"] != "invalid"
        }
        measurements[side] = {"cards": rows, "trends": trends(historical)}
        if measurements[side]["trends"].get("preliminary"):
            limitations.append(side + ": short history; measured trends are provisional.")
        if any(row["features"]["status"] == "questionable" for row in rows):
            limitations.append(side + ": questionable curves are included for first-version review.")
        reviewed_ids[side] = {
            row["reference"] for row in rows if row["features"]["status"] != "invalid"
        }
        history_curves = [
            curve for curve in curves if curve[0].split()[0] in {row["reference"] for row in historical}
        ]
        historical_images.append(
            render(history_curves, side + " historical daily baseline") if history_curves else ""
        )
        if not recent_ids[side]:
            limitations.append(side + ": no recent qualifying pumping card available.")
    images = [
        combined_panels(*historical_images, "Historical overlays: surface and downhole"),
        comparison_sheet(cards, recent_ids, "surface"),
        comparison_sheet(cards, recent_ids, "downhole"),
    ]
    facts = evidence_catalog(measurements)
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "measurements": measurements,
                    "required_shape_card_ids": {side: sorted(ids) for side, ids in recent_ids.items()},
                    "evidence_catalog": facts,
                    "templates": [
                        {key: template[key] for key in ("id", "side", "title", "points")}
                        for template in TEMPLATES
                    ],
                    "calculation_snapshot": packet.calculation_snapshot,
                    "warnings": {c.reference: c.warnings for c in cards},
                    "pumping_evidence": {
                        c.reference: c.pumping_window for c in cards if "recent" in c.roles
                    },
                    "downhole_errors": {
                        c.reference: c.downhole_error for c in cards if c.downhole_error
                    },
                },
                default=str,
            ),
        }
    ]
    content.extend({"type": "image_url", "image_url": {"url": image}} for image in images if image)
    findings = []
    historical_ids = {
        side: {
            row["reference"]
            for row in measurements[side]["cards"]
            if "baseline" in row["roles"] and row["reference"].startswith("archive:")
        }
        for side in ("surface", "downhole")
    }
    review_status = "unavailable"
    review_error = None
    token_usage = None
    if any(reviewed_ids.values()):
        stage = "LLM request"
        try:
            raw_review = reviewer(content)
            stage = "Response handling"
            if isinstance(raw_review, dict):
                raw_findings = raw_review.get("findings") or []
                raw_limitations = raw_review.get("limitations") or []
                token_usage = raw_review.get("token_usage")
            else:
                raw_findings = [raw_review]
                raw_limitations = []
            if isinstance(raw_findings, dict):
                raw_findings = [raw_findings]
            if not isinstance(raw_findings, list):
                raw_findings = [raw_findings]
            if isinstance(raw_limitations, str):
                raw_limitations = [raw_limitations]
            if isinstance(raw_limitations, list):
                limitations.extend(str(value) for value in raw_limitations)
            for item in raw_findings[:2]:
                output = dict(item) if isinstance(item, dict) else {"title": str(item)}
                side = str(output.get("side", "surface")).lower()
                output["side"] = side if side in ("surface", "downhole") else "surface"
                title = output.get("title") or output.get("suspected_problem") or "Observation"
                output["title"] = str(title)
                reasoning = output.get("reasoning", "")
                output["reasoning"] = " ".join(str(reasoning).split())
                actions = output.get("actions", "")
                output["actions"] = " ".join(str(actions).split())
                findings.append(output)
            review_status = "complete"
        except Exception as exception:
            review_error = review_failure_message(exception, stage)
            logger.warning("Rod pump health review failed well_id=%s: %s", packet.well_id, review_error)
            findings = []
            review_status = "failed"
            limitations.append("AI review failed. No healthy verdict is inferred.")
    else:
        counts = []
        for side, data in measurements.items():
            statuses = [row["features"]["status"] for row in data["cards"]]
            counts.append(
                side
                + ": "
                + ", ".join(
                    f"{status}={statuses.count(status)}"
                    for status in ("valid", "questionable", "invalid")
                )
            )
        review_error = "No usable cards; LLM was not called. " + "; ".join(counts)
        logger.warning(
            "Rod pump health review unavailable well_id=%s: %s", packet.well_id, review_error
        )
    return {
        "review_error": review_error,
        "analyzer_version": VERSION,
        "template_version": TEMPLATE_VERSION,
        "status": (
            "complete"
            if review_status == "complete"
            and not limitations
            and all(len(reviewed_ids[s]) == len(cards) and recent_ids[s] for s in reviewed_ids)
            else "partial"
        ),
        "review_status": review_status,
        "diagnosis_status": (
            "candidates"
            if findings
            else "healthy"
            if review_status == "complete"
            else "unavailable"
        ),
        "measurements": measurements,
        "findings": findings,
        "token_usage": token_usage,
        "limitations": limitations,
        "templates": TEMPLATES,
    }


SYSTEM_PROMPT = (
    "Analyze Lufkin SAM1 rod-pump data without a VFD. Review recent surface/downhole cards, archived "
    "dynographs, measurements, and templates. Identify up to two problems from shape matches, "
    "historical trends, or rod-pump knowledge; no findings means Healthy. Return separate concise "
    "reasoning and actions paragraphs per diagnosis. Put practical field steps and relevant SAM1 "
    "setpoint changes in actions; give numeric values only when data supports them. Separate "
    "suspected causes from confirmed faults. Return JSON findings: side, kind, scope, title, "
    "reasoning, actions, template_ids, card_ids, confidence. Use scope=last_eligible_card "
    "for latest findings and scope=historical for archive-only findings. "
    "Confidence is low/medium/high based on defining-feature clarity, template similarity, "
    "alternative explanations and consistency across cards. A small clear anomaly can have high "
    "confidence; a large ambiguous anomaly can have low confidence. Size alone must not increase "
    "confidence or imply equipment damage. Explain confidence briefly in reasoning. Healthy "
    "findings need no confidence field. Do not add a separate pattern-match rating."
)


def llm_reviewer(settings, system_prompt=SYSTEM_PROMPT):
    """Use the configured model's native structured response and image input."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=150,
        max_retries=0,
    )
    structured = model.bind(response_format={"type": "json_object"})

    def review(content):
        try:
            response = structured.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=content)]
            )
            result = json.loads(response.content)
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
