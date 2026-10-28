"""Read the selected rod-pump card and calculate its downhole curve before text-only review."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from umai.rod_pump_health.analysis import AnalysisRequest, VERSION, analyze, health_reviewer, measure
from umai.rod_pump_health.classifier import TRUSTED_LABELS, load_classifier

logger = logging.getLogger(__name__)

# Defaults for when the classifier may stand in for a fresh LLM review.
DEFAULT_ROUTING = {"min_confidence": 0.9, "refresh_days": 7, "min_downhole_load_range_lb": 3000.0}


class RecentCardReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["live", "archive"]
    id: int = Field(gt=0)
    monitoring_data_id: int | None = Field(default=None, gt=0)
    pull_timestamp: int = Field(gt=0)
    card_timestamp: int = Field(gt=0)
    pumping_window_start: int
    pumping_window_end: int


class WellAnalysisRequest(BaseModel):
    """The web service supplies only the globally unique well database ID."""

    model_config = ConfigDict(extra="forbid")
    well_id: int = Field(gt=0)
    recent_card: RecentCardReference | None = None


def decoded(value, default):
    """Accept driver-decoded JSON and JSON strings; malformed data stays unavailable."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def physical_points(points, position_scale=1, declared_empty=False):
    """Convert units without repairing malformed curves by removing their points."""
    if declared_empty or not isinstance(points, list):
        return []
    output = []
    for point in points:
        if not isinstance(point, dict):
            return []
        values = [point.get("position"), point.get("load")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return []
        if not all(math.isfinite(value) for value in values):
            return []
        output.append({"position": values[0] / position_scale, "load": values[1]})
    return output


class RodPumpHealthAnalyzer:
    """Own data preparation and interpretation; derived records are written by UMetriqs."""

    def __init__(
        self,
        engine,
        archive_engine,
        timezone,
        report_url,
        reviewer,
        http_post=httpx.post,
        point_limits=None,
        classifier=None,
        routine_reviewer=None,
        routing=None,
    ):
        self.engine = engine
        self.archive_engine = archive_engine
        self.timezone = ZoneInfo(timezone)
        self.report_url = report_url
        self.reviewer = reviewer
        self.http_post = http_post
        # Maximum coordinates sent to the LLM per curve; None sends every point.
        self.point_limits = point_limits
        # Downhole classifier and the cheaper reviewer for routine healthy cards; without a
        # classifier every card goes to the LLM.
        self.classifier = classifier
        self.routine_reviewer = routine_reviewer or reviewer
        self.routing = {**DEFAULT_ROUTING, **(routing or {})}

    def _well(self, well_id):
        """Resolve site ownership and configured scope from the operational database."""
        with self.engine.connect() as connection:
            well = (
                connection.execute(
                    text("SELECT id, site_id FROM wells WHERE id=:id AND LOWER(pump_type)='rod'"),
                    {"id": well_id},
                )
                .mappings()
                .first()
            )
            if well is None:
                raise ValueError("Configured rod well not found.")
            configuration = connection.execute(
                text(
                    "SELECT related_entities FROM report_configurations "
                    "WHERE site_id=:site AND function_name='rod_pump_monitoring'"
                ),
                {"site": well["site_id"]},
            ).scalar()
        configuration = decoded(configuration, {})
        ids = configuration.get("well_ids", []) if isinstance(configuration, dict) else []
        if well_id not in {int(value) for value in ids if str(value).isdigit()}:
            raise ValueError("Well is outside the configured monitoring scope.")
        return dict(well)

    def _recent_card(self, well_id, reference):
        """Resolve the validated source reference without trusting cross-well card identifiers."""
        resolved_source = reference.source
        if reference.source == "live":
            with self.engine.connect() as connection:
                row = (
                    connection.execute(
                        text(
                            "SELECT c.id, c.timestamp AS card_timestamp, c.id AS source_card_id, "
                            "d.pull_timestamp, c.card_type, c.info, c.data_points "
                            "FROM rod_pump_cards c JOIN rod_pump_monitoring_data_cards link "
                            "ON link.rod_pump_card_id=c.id JOIN rod_pump_monitoring_data d "
                            "ON d.id=link.rod_pump_monitoring_data_id "
                            "WHERE c.id=:id AND d.id=:pull AND d.well_id=:well "
                            "AND LOWER(c.card_type)='current'"
                        ),
                        {"id": reference.id, "pull": reference.monitoring_data_id, "well": well_id},
                    )
                    .mappings()
                    .first()
                )
        else:
            with self.archive_engine.connect() as connection:
                row = (
                    connection.execute(
                        text(
                            "SELECT id, card_timestamp, source_card_id, pull_timestamp, card_type, "
                            "info, data_points FROM rod_pump_card_archive "
                            "WHERE id=:id AND well_id=:well AND LOWER(card_type)='current'"
                        ),
                        {"id": reference.id, "well": well_id},
                    )
                    .mappings()
                    .first()
                )
        if row is None and reference.source == "live":
            # Live pruning may run between UMetriqs selection and this read.
            with self.archive_engine.connect() as connection:
                row = (
                    connection.execute(
                        text(
                            "SELECT id, card_timestamp, source_card_id, pull_timestamp, card_type, "
                            "info, data_points FROM rod_pump_card_archive WHERE well_id=:well "
                            "AND source_card_id=:id AND pull_timestamp=:pull AND card_timestamp=:capture "
                            "AND LOWER(card_type)='current'"
                        ),
                        {
                            "well": well_id,
                            "id": reference.id,
                            "pull": reference.pull_timestamp,
                            "capture": reference.card_timestamp,
                        },
                    )
                    .mappings()
                    .first()
                )
            resolved_source = "archive"
        if row is None:
            raise ValueError("The selected current card is unavailable for this well. Refresh again.")
        row = dict(row)
        timestamp = int(row["card_timestamp"])
        if (
            timestamp != reference.card_timestamp
            or int(row["pull_timestamp"]) != reference.pull_timestamp
        ):
            raise ValueError("Selected card reference changed. Refresh again.")
        if (
            reference.pumping_window_start != timestamp - 600
            or reference.pumping_window_end != timestamp + 300
            or reference.pumping_window_end > datetime.now(self.timezone).timestamp()
        ):
            raise ValueError("Invalid or incomplete pumping window.")
        row["info"] = decoded(row["info"], {})
        row["data_points"] = decoded(row["data_points"], [])
        row["reference"] = f"{resolved_source}:{row['id']}"
        row["archive_id"] = row["id"] if resolved_source == "archive" else None
        row["roles"] = ["recent"]
        row["pumping_window"] = {
            "start": reference.pumping_window_start,
            "end": reference.pumping_window_end,
            "confirmed_by": "UMetriqs pumping eligibility service",
        }
        return row

    def _downhole(self, well, cards):
        """Use the existing UMreports inline-card API; failures still allow surface analysis."""
        try:
            response = self.http_post(
                self.report_url,
                json={
                    "site_id": well["site_id"],
                    "function_name": "rod_pump_downhole_cards",
                    "params": {
                        "well_id": well["id"],
                        "pull_timestamp": cards[-1]["pull_timestamp"],
                        "surface_cards": [
                            {
                                "id": card["reference"],
                                **{key: card[key] for key in ("card_type", "info", "data_points")},
                            }
                            for card in cards
                        ],
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            result = decoded(response.json(), {})
            if not isinstance(result, dict):
                raise ValueError("Invalid report response.")
            return result
        except (httpx.HTTPError, ValueError, TypeError):
            return {"cards": [], "error": "Downhole calculation service unavailable."}

    def prepare(self, well_id, recent_card=None):
        """Resolve the selected card to a single-card packet, including its downhole card."""
        well = self._well(well_id)
        if recent_card is None:
            raise ValueError("No eligible pumping card was selected for this well.")
        card = self._recent_card(well_id, recent_card)
        calculation = self._downhole(well, [card])
        curve = next(
            (
                row
                for row in calculation.get("cards", [])
                if isinstance(row, dict) and row.get("card_id") == card["reference"]
            ),
            {},
        )
        downhole = []
        positions, loads = curve.get("positions", []), curve.get("loads", [])
        if curve.get("ok") and isinstance(positions, list) and isinstance(loads, list):
            if len(positions) == len(loads):
                downhole = physical_points(
                    [{"position": position, "load": load} for position, load in zip(positions, loads)]
                )
        # Calculated pump values give the model fillage-related context without extra curves.
        summary = {
            key: curve[key]
            for key in (
                "gross_stroke_in",
                "fluid_load_lb",
                "min_load_lb",
                "max_load_lb",
                "tubing_stretch_in",
            )
            if downhole and isinstance(curve.get(key), (int, float))
        }
        header = card["info"] if isinstance(card["info"], dict) else {}
        period = header.get("stroke_period")
        period = period / 100 if isinstance(period, (int, float)) and period > 0 else None
        packet_card = {
            "archive_id": card["archive_id"],
            "reference": card["reference"],
            "roles": ["recent"],
            "pull_timestamp": card["pull_timestamp"],
            "pumping_window": card.get("pumping_window", {}),
            "source_card_id": card["source_card_id"],
            "timestamp": datetime.fromtimestamp(card["card_timestamp"], self.timezone),
            "header": header,
            "surface": physical_points(card["data_points"], 100, header.get("number_points") == 0),
            "downhole": downhole,
            "downhole_summary": summary,
            "period_s": period,
            "point_order": "sam1_reverse_time",
            "warnings": curve.get("warnings", []),
            "downhole_error": (
                None
                if downhole
                else curve.get("error", calculation.get("error", "No calculated curve available."))
            ),
        }
        return AnalysisRequest(site_id=well["site_id"], well_id=well_id, card=packet_card)

    def _classify(self, packet):
        """Classify the downhole card; None when there is no classifier or no downhole card."""
        if self.classifier is None or not packet.card.downhole:
            return None
        try:
            return self.classifier.classify(packet.card.downhole)
        except Exception:
            logger.exception("Dynograph classifier failed well_id=%s", packet.well_id)
            return None

    def _route(self, packet, classification, saved_findings):
        """Decide whether the last LLM review still applies to this card.

        Returns (reuse, routine, reason): reuse the saved review, or call the LLM, with the
        routine (cheaper) effort when the card is a confident, ordinary healthy card.
        """
        if classification is None:
            reason = "no downhole card" if not packet.card.downhole else "classifier unavailable"
            return False, False, reason
        loads = [point.load for point in packet.card.downhole]
        if max(loads) - min(loads) < self.routing["min_downhole_load_range_lb"]:
            return False, False, "collapsed downhole card"
        label = classification["label"]
        if label not in TRUSTED_LABELS:
            return False, False, f"{label} needs an LLM review"
        confidence = min(classification["health_confidence"], classification["label_confidence"])
        if confidence < self.routing["min_confidence"]:
            return False, False, f"classifier confidence {confidence:.2f} is below the threshold"
        routine = label == "healthy"
        review = saved_findings.get("llm_review") if isinstance(saved_findings, dict) else None
        if saved_findings.get("review_status") != "complete" or not isinstance(review, dict):
            return False, routine, "no previous LLM review"
        if review.get("classifier_model_version") != classification["model_version"]:
            return False, routine, "classifier model changed since the last LLM review"
        if review.get("classifier_label") != label:
            return False, False, f"pattern changed from {review.get('classifier_label')} to {label}"
        # Compare card times, not the clock, so as-of runs only reuse reviews of earlier cards.
        try:
            reviewed_at = datetime.fromisoformat(review["card_captured_at"])
        except (KeyError, TypeError, ValueError):
            return False, routine, "previous LLM review has no card time"
        age_days = (packet.card.timestamp - reviewed_at).total_seconds() / 86400
        if age_days < 0:
            return False, routine, "previous LLM review is of a later card"
        if age_days >= self.routing["refresh_days"]:
            return False, routine, f"previous LLM review is {age_days:.0f} days old"
        return True, routine, f"same {label} pattern as the LLM review of {review.get('card_reference')}"

    def analyze(self, well_id, recent_card=None):
        """Read the selected card, reuse an unchanged saved review, or return a new result."""
        packet = self.prepare(well_id, recent_card)
        classification = self._classify(packet)
        signature = hashlib.sha256(
            json.dumps(
                [
                    packet.model_dump(mode="json"),
                    VERSION,
                    self.point_limits,
                    classification and classification["model_version"],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self.engine.connect() as connection:
            saved = (
                connection.execute(
                    text(
                        "SELECT input_signature, findings FROM rod_pump_health_analyses "
                        "WHERE well_id=:id"
                    ),
                    {"id": well_id},
                )
                .mappings()
                .first()
            )
        metadata = {
            "well_id": well_id,
            "site_id": packet.site_id,
            "input_signature": signature,
            "analyzer_version": VERSION,
        }
        saved_findings = decoded(saved["findings"], {}) if saved else {}
        saved_findings = saved_findings if isinstance(saved_findings, dict) else {}
        if saved and saved["input_signature"] == signature:
            if saved_findings.get("review_status") == "complete":
                return {**metadata, "unchanged": True, "review_status": "complete"}
        reuse, routine, reason = self._route(packet, classification, saved_findings)
        logger.info(
            "Rod pump health well_id=%s: %s (%s)",
            well_id, "reusing LLM review" if reuse else "LLM review", reason,
        )
        if reuse:
            result = self._reused_review(packet, saved_findings)
        else:
            effort = "routine" if routine else "default"
            result = analyze(
                packet, self.routine_reviewer if routine else self.reviewer, self.point_limits
            )
            result["review_source"] = "llm"
            if result["review_status"] == "complete":
                result["llm_review"] = {
                    "card_reference": packet.card.reference,
                    "card_captured_at": packet.card.timestamp.isoformat(),
                    "effort": effort,
                    "classifier_label": classification and classification["label"],
                    "classifier_model_version": classification and classification["model_version"],
                }
                if classification:
                    # Disagreements are the cases to review when retraining the classifier.
                    logger.info(
                        "Classifier vs LLM well_id=%s: LLM %s, classifier %s/%s (%.2f)",
                        well_id, result["health_status"], classification["health"],
                        classification["label"], classification["label_confidence"],
                    )
        captured = packet.card.timestamp.isoformat()
        return {
            **result,
            **metadata,
            "classifier": classification,
            "routing": {"llm_called": not reuse, "reason": reason},
            "unchanged": False,
            "window_start_at": captured,
            "window_end_at": captured,
            "latest_card_at": captured,
            "input_cards": [packet.card.model_dump(mode="json")],
            "calculation_snapshot": {},
        }

    @staticmethod
    def _reused_review(packet, saved_findings):
        """Carry the last LLM verdict onto the new card, with that card's own measurements."""
        measurements, limitations = measure(packet.card)
        findings = saved_findings.get("findings") or []
        return {
            "review_error": None,
            "analyzer_version": VERSION,
            "status": "complete" if not limitations else "partial",
            "review_status": "complete",
            "review_source": "classifier",
            "diagnosis_status": "candidates" if findings else "healthy",
            "health_status": saved_findings.get("health_status"),
            "summary": saved_findings.get("summary"),
            "measurements": measurements,
            "findings": findings,
            "token_usage": None,
            "limitations": limitations,
            "llm_review": saved_findings["llm_review"],
        }


@lru_cache(maxsize=4)
def analyzer_from_settings(settings):
    """Reuse operational/archive read connections; archive options fall back to operational values."""

    def engine(host, port, user, password, database):
        return create_engine(
            URL.create(
                "mysql+pymysql",
                username=user,
                password=password,
                host=host,
                port=port,
                database=database,
            ),
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
        )

    if not all((settings.db_host, settings.db_user, settings.db_name)):
        raise ValueError("Operational database settings are incomplete.")
    operational = engine(
        settings.db_host, settings.db_port, settings.db_user, settings.db_password, settings.db_name
    )
    archive = engine(
        settings.db_archive_host or settings.db_host,
        settings.db_archive_port or settings.db_port,
        settings.db_archive_user or settings.db_user,
        (
            settings.db_archive_password
            if settings.db_archive_password is not None
            else settings.db_password
        ),
        settings.db_archive_name or settings.db_name,
    )
    return RodPumpHealthAnalyzer(
        operational,
        archive,
        settings.timezone,
        settings.umreports_api_url,
        lambda content: health_reviewer(settings)(content),
        point_limits={
            "surface": settings.rod_pump_health_surface_points,
            "downhole": settings.rod_pump_health_downhole_points,
        },
        classifier=load_classifier(settings.rod_pump_health_classifier_model),
        routine_reviewer=lambda content: health_reviewer(
            settings, settings.rod_pump_health_routine_effort
        )(content),
        routing={
            "min_confidence": settings.rod_pump_health_classifier_min_confidence,
            "refresh_days": settings.rod_pump_health_llm_refresh_days,
            "min_downhole_load_range_lb": settings.rod_pump_health_min_downhole_load_range_lb,
        },
    )
