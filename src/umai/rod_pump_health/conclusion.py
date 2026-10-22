"""Synthesize historical card diagnoses and monthly parameter trends in a separate LLM call."""

from datetime import datetime
import hashlib
import json
import re

from sqlalchemy import text

from umai.rod_pump_health.analysis import llm_reviewer
from umai.rod_pump_health.reader import analyzer_from_settings, decoded


SYSTEM_PROMPT = (
    "Write a final historical well-health conclusion for a Lufkin SAM1 rod-pump unit without a "
    "VFD. You receive historical surface and downhole diagnoses plus a separate monthly parameter "
    "trend interpretation, not raw data or dynographs. Synthesize their combined oilfield meaning "
    "in JSON: title (one concise final assessment) and body (one short paragraph). Explain the "
    "most likely overall operating condition, whether the findings corroborate or conflict, and "
    "the relevant uncertainty. Do not merely list the source diagnoses or invent additional "
    "confirmed faults. Missing or failed reviews are unknown, not healthy; qualify the conclusion "
    "when inputs are incomplete. An empty successful review indicates no supported problem from "
    "that source. If both reviews completed and neither identifies a problem, use title Healthy. "
    "Avoid dates, "
    "numeric values, VFD explanations and unsupported operational changes."
)


def historical_findings(review, cards):
    """Respect explicit scope, with the UI's legacy inference, and exclude low-confidence findings."""
    recent = {
        card.get("reference") for card in cards if "recent" in card.get("roles", [])
    }
    output = {"surface": [], "downhole": []}
    for finding in review.get("findings", []):
        if not isinstance(finding, dict):
            continue
        if str(finding.get("kind", "")).strip().lower() == "healthy":
            continue
        if str(finding.get("confidence", "")).strip().lower() == "low":
            continue
        scope = finding.get("scope")
        if scope in ("last_eligible_card", "historical"):
            historical = scope == "historical"
        else:
            prose = str(finding.get("title", "")) + " " + str(finding.get("reasoning", ""))
            historical = finding.get("kind") == "trend" or bool(re.search(
                r"\bhistorical\b|\bnot present (?:on|in) (?:the )?(?:latest|current)\b", prose, re.I,
            ))
            ids = finding.get("card_ids", [])
            ids = ids if isinstance(ids, list) else []
            references = {f"archive:{value}" if str(value).isdigit() else str(value) for value in ids}
            historical = historical or bool(references and not references.intersection(recent))
        side = finding.get("side", "surface")
        if historical and side in output:
            output[side].append({
                key: finding[key] for key in ("title", "reasoning", "confidence") if key in finding
            })
    return output


class ConclusionAnalyzer:
    """Review only retained historical assessments, with an independent signature and token usage."""

    def __init__(self, health_analyzer, reviewer):
        self.health_analyzer = health_analyzer
        self.engine = health_analyzer.engine
        self.timezone = health_analyzer.timezone
        self.reviewer = reviewer

    def analyze(self, well_id, card_result, parameter_trends):
        """Resolve cached card assessments before running the final text-only synthesis."""
        well = self.health_analyzer._well(well_id)
        with self.engine.connect() as connection:
            saved = connection.execute(text(
                "SELECT findings, input_cards FROM rod_pump_health_analyses WHERE well_id=:well"
            ), {"well": well_id}).mappings().first()
        saved_review = decoded(saved["findings"], {}) if saved else {}
        if card_result.get("unchanged"):
            review = saved_review
            cards = decoded(saved["input_cards"], []) if saved else []
        else:
            review = card_result
            cards = card_result.get("input_cards", [])
        diagnoses = historical_findings(review, cards)
        packet = {
            **diagnoses,
            "card_review_status": review.get("review_status", "unavailable"),
            "parameter_trends": {
                key: parameter_trends.get(key) for key in ("status", "trends", "interpretation")
            },
        }
        metadata = {
            "well_id": well_id, "site_id": well["site_id"],
            "analyzed_at": datetime.now(self.timezone).isoformat(),
            "input_signature": hashlib.sha256(
                json.dumps([packet, SYSTEM_PROMPT], sort_keys=True).encode()
            ).hexdigest(),
        }
        cards_complete = packet["card_review_status"] == "complete"
        trends_complete = parameter_trends.get("status") == "complete"
        if not cards_complete and not trends_complete:
            return {**metadata, "status": "insufficient_data"}
        cached = saved_review.get("conclusion") or {}
        if cached.get("input_signature") == metadata["input_signature"]:
            if cached.get("status") in ("complete", "partial"):
                return {**cached, "last_refresh_error": None, "unchanged": True}
        result = self.reviewer([{"type": "text", "text": json.dumps(packet)}])
        if not isinstance(result, dict) or not all(
            isinstance(result.get(key), str) and result[key].strip() for key in ("title", "body")
        ):
            raise ValueError("Conclusion review must include a title and body.")
        return {
            **metadata, "status": "complete" if cards_complete and trends_complete else "partial",
            "title": " ".join(result["title"].split()),
            "body": " ".join(result["body"].split()),
            "token_usage": result.get("token_usage"),
        }


def conclusion_analyzer_from_settings(settings):
    """Give final synthesis its own prompt and LLM invocation within the health request."""
    return ConclusionAnalyzer(
        analyzer_from_settings(settings),
        lambda content: llm_reviewer(settings, system_prompt=SYSTEM_PROMPT)(content),
    )
