"""Review a month of controller telemetry independently of dynograph interpretation."""

import base64
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
from statistics import median

import httpx
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import text

from umai.clients.data_point_client import DataPointClient
from umai.rod_pump_health.analysis import llm_reviewer
from umai.rod_pump_health.reader import analyzer_from_settings, decoded


PARAMETERS = (
    "Pump Fillage", "Max Load LS", "Min Load LS",
    "Yesterday Strokes /Min", "Yesterday Percent", "Yesterday Cycles",
)
SYSTEM_PROMPT = (
    "Analyze six parameter graphs covering the last 30 days of Lufkin SAM1 rod-pump operation "
    "without a VFD. Use your oilfield knowledge to identify unusual overall trends, sustained "
    "changes or abnormal variability, considering the parameters together. Each graph shows up "
    "to 720 consolidated monitoring samples, not raw strokes or daily median/min/max summaries. "
    "Values are unscaled telemetry; do not assume percentages versus fractions. Yesterday "
    "channels report the preceding day. Gaps are unknown, not zero; shutdowns and sampling gaps "
    "can explain changes. Sparse data does not establish a fault. Return JSON with trends: "
    "[{parameter, title}] for supplied parameters with an unusual overall trend. Title must be a "
    "short overall trend such as 'Sustained decline', 'Gradual increase', 'Increasing variability' "
    "or 'Persistent step decrease'. Do not include dates, measured values or detailed episodes "
    "in these titles. Then return "
    "interpretation: one short paragraph explaining what these trends together suggest in SAM1 "
    "oilfield terms, including plausible causes and uncertainty, without dates or numeric values. "
    "Do not infer VFD adjustments or production changes. Use your own knowledge; no dynographs "
    "or templates are supplied. If no unusual trend is supported, return an empty trends list "
    "and say so in interpretation."
)


def monitoring_samples(datapoints, start, end):
    """Keep ordered in-window measurements and explicit null gaps, without daily aggregation."""
    samples = {}
    for point in datapoints:
        try:
            value, timestamp = point
            if timestamp is None:
                continue
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                continue
            captured = datetime.fromtimestamp(timestamp, start.tzinfo)
            if start <= captured < end:
                value = float(value) if value is not None else None
                samples[timestamp] = value if value is not None and math.isfinite(value) else None
        except (ValueError, TypeError, OverflowError, OSError):
            continue
    return [
        {"timestamp": timestamp, "value": value} for timestamp, value in sorted(samples.items())
    ]


def render_parameter_graph(name, samples, start, end):
    """Render the original monitoring series on time/value axes, breaking lines across gaps."""
    image = Image.new("RGB", (1100, 450), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        heading = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        font = heading = ImageFont.load_default()
    draw.text((95, 12), name, fill="#0f172a", font=heading)
    draw.text((95, 45), "Unscaled monitoring values", fill="#475569", font=font)
    values = [point["value"] for point in samples if point["value"] is not None]
    xmin, xmax = start.timestamp(), end.timestamp()
    ymin, ymax = min(values), max(values)
    padding = (ymax - ymin) * 0.08 or max(abs(ymin) * 0.02, 0.01)
    ymin, ymax = ymin - padding, ymax + padding
    for tick in range(5):
        x, y = 100 + tick * 240, 355 - tick * 65
        value = ymin + (ymax - ymin) * tick / 4
        timestamp = xmin + (xmax - xmin) * tick / 4
        draw.line([(100, y), (1060, y)], fill="#e2e8f0")
        draw.text((3, y - 9), f"{value:.4g}", fill="#475569", font=font)
        label = datetime.fromtimestamp(timestamp, start.tzinfo).strftime("%b %d")
        draw.text((x - 30, 370), label, fill="#475569", font=font)
    draw.line([(100, 95), (100, 355), (1060, 355)], fill="#64748b", width=2)
    draw.text((430, 414), f"Last 30 days ({start.tzinfo})", fill="#475569", font=font)
    steps = [b["timestamp"] - a["timestamp"] for a, b in zip(samples, samples[1:])]
    gap_limit = 3 * median(steps) if steps else float("inf")
    previous = None
    for sample in samples:
        if sample["value"] is None:
            previous = None
            continue
        x = 100 + (sample["timestamp"] - xmin) / (xmax - xmin) * 960
        y = 355 - (sample["value"] - ymin) / (ymax - ymin) * 260
        if previous and sample["timestamp"] - previous[0] <= gap_limit:
            draw.line([previous[1], (x, y)], fill="#0f766e", width=2)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#0f766e")
        previous = (sample["timestamp"], (x, y))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class ParameterTrendAnalyzer:
    """Read six scoped parameter series and review their graphs independently of dynographs."""

    def __init__(self, health_analyzer, monitoring_url, reviewer, http_get=httpx.get, timeout=20):
        self.health_analyzer = health_analyzer
        self.engine = health_analyzer.engine
        self.timezone = health_analyzer.timezone
        self.monitoring_url = monitoring_url
        self.reviewer = reviewer
        self.http_get = http_get
        self.timeout = timeout

    def _points(self, well):
        """Read this well's configured tags, including points with disabled alarm evaluation."""
        with self.engine.connect() as connection:
            rows = connection.execute(text(
                "SELECT dp.data_point_name, dp.tag, s.`key` AS site_key, "
                "COALESCE(parent.facility_name, dp.facility_name) AS facility_name "
                "FROM wells w JOIN sites s ON s.id=w.site_id "
                "JOIN data_points dp ON dp.site_id=w.site_id "
                "LEFT JOIN data_points parent ON parent.id=dp.facility_id "
                "WHERE w.id=:well AND w.site_id=:site "
                "AND COALESCE(parent.facility_name, dp.facility_name)=w.`key` "
                "ORDER BY dp.id"
            ), {"well": well["id"], "site": well["site_id"]}).mappings().all()
        targets = {}
        for row in rows:
            if row["data_point_name"] not in PARAMETERS or not row["tag"]:
                continue
            tag = DataPointClient.ROD_PUMP_TAG_MAPPING.get(row["tag"], row["tag"])
            facility = row["facility_name"].replace(" ", "_")
            targets[f"MI3.{row['site_key']}.{facility}.{tag}"] = row["data_point_name"]
        return targets

    def analyze(self, well_id, now=None):
        """Analyze the latest thirty days independently of card availability and cached card reviews."""
        well = self.health_analyzer._well(well_id)
        evaluated_at = (now or datetime.now(self.timezone)).astimezone(self.timezone)
        end = evaluated_at.replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=30)
        targets = self._points(well)
        parameters = {}
        if targets:
            params = [("target", target) for target in targets]
            params.extend([
                ("from", str(int(start.timestamp()))), ("until", str(int(end.timestamp()))),
                ("format", "json"), ("maxDataPoints", "720"), ("tz", str(self.timezone)),
            ])
            response = self.http_get(self.monitoring_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Monitoring service did not return parameter series.")
            for series in payload:
                name = targets.get(series.get("target")) if isinstance(series, dict) else None
                if name:
                    samples = monitoring_samples(series.get("datapoints", []), start, end)
                    if any(point["value"] is not None for point in samples):
                        parameters[name] = samples
        packet = {
            "well_id": well_id, "site_id": well["site_id"],
            "window_start_at": start.isoformat(), "window_end_at": end.isoformat(),
            "parameters": parameters,
            "missing_parameters": sorted(set(PARAMETERS) - parameters.keys()),
        }
        signature = hashlib.sha256(json.dumps([packet, SYSTEM_PROMPT], sort_keys=True).encode())
        metadata = {
            **packet, "input_signature": signature.hexdigest(),
            "analyzed_at": evaluated_at.isoformat(),
        }
        if not any(len({
            datetime.fromtimestamp(point["timestamp"], self.timezone).date()
            for point in samples if point["value"] is not None
        }) >= 2 for samples in parameters.values()):
            return {**metadata, "status": "insufficient_data", "trends": [],
                    "interpretation": "Insufficient parameter history to assess monthly trends."}
        with self.engine.connect() as connection:
            saved = connection.execute(text(
                "SELECT findings FROM rod_pump_health_analyses WHERE well_id=:well"
            ), {"well": well_id}).scalar()
        cached = decoded(saved, {}).get("parameter_trends") or {}
        if cached.get("input_signature") == metadata["input_signature"]:
            if cached.get("status") == "complete":
                return {
                    **cached, "well_id": well_id, "site_id": well["site_id"],
                    "unchanged": True, "last_refresh_error": None,
                }
        content = [{"type": "text", "text": (
            "Review the following 30-day SAM1 parameter graphs. "
            f"Unavailable parameters: {', '.join(packet['missing_parameters']) or 'none'}."
        )}]
        content.extend({"type": "image_url", "image_url": {"url": render_parameter_graph(
            name, parameters[name], start, end,
        )}} for name in PARAMETERS if name in parameters)
        review = self.reviewer(content)
        if not isinstance(review, dict):
            raise ValueError("Parameter review must be a JSON object.")
        raw_trends = review.get("trends", [])
        if not isinstance(raw_trends, list) or not isinstance(review.get("interpretation"), str):
            raise ValueError("Parameter review is missing trends or interpretation.")
        by_name = {
            row["parameter"]: str(row.get("title", "")).strip()
            for row in raw_trends if isinstance(row, dict) and isinstance(row.get("parameter"), str)
        }
        trends = [
            {"parameter": name, "title": by_name[name]}
            for name in PARAMETERS if name in by_name and name in parameters
        ]
        return {
            **metadata, "status": "complete", "trends": trends,
            "interpretation": " ".join(review["interpretation"].split()),
            "token_usage": review.get("token_usage"),
        }


def parameter_analyzer_from_settings(settings):
    """Reuse scoped database access and review parameter graphs with their own prompt and LLM call."""
    return ParameterTrendAnalyzer(
        analyzer_from_settings(settings), settings.monitoring_data_api_url,
        lambda content: llm_reviewer(settings, system_prompt=SYSTEM_PROMPT)(content),
        timeout=settings.data_point_trend_timeout_seconds,
    )
