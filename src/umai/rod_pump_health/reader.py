"""Read archived daily cards and calculate pump curves before multimodal review."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from umai.rod_pump_health.analysis import AnalysisRequest, VERSION, analyze, llm_reviewer
from umai.rod_pump_health.templates import VERSION as TEMPLATE_VERSION


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

    def __init__(self, engine, archive_engine, timezone, report_url, reviewer, http_post=httpx.post):
        self.engine = engine
        self.archive_engine = archive_engine
        self.timezone = ZoneInfo(timezone)
        self.report_url = report_url
        self.reviewer = reviewer
        self.http_post = http_post

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

    def _daily_cards(self, well_id):
        """Stream retained cards by actual capture time, keeping only daily baseline evidence."""
        by_day = defaultdict(list)
        seen = set()
        end = start = None
        with self.archive_engine.connect() as connection:
            rows = (
                connection.execution_options(stream_results=True)
                .execute(
                    text(
                        "SELECT id, well_id, card_timestamp, pull_timestamp, source_card_id, "
                        "card_type, data_points, info, archive_reasons FROM rod_pump_card_archive "
                        "WHERE well_id=:well AND retention_state='retained' "
                        "AND LOWER(card_type)='current' ORDER BY card_timestamp DESC, id ASC"
                    ),
                    {"well": well_id},
                )
                .mappings()
            )
            for raw in rows:
                timestamp = datetime.fromtimestamp(int(raw["card_timestamp"]), self.timezone)
                if start is not None and timestamp < start:
                    break
                reasons = decoded(raw["archive_reasons"], [])
                if not isinstance(reasons, list) or "daily_baseline" not in reasons:
                    continue
                if end is None:
                    end = timestamp
                    start = end.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=13)
                if int(raw["card_timestamp"]) in seen:
                    continue
                seen.add(int(raw["card_timestamp"]))
                row = dict(raw)
                row["info"] = decoded(row["info"], {})
                row["data_points"] = decoded(row["data_points"], [])
                by_day[timestamp.date()].append(row)
        if not by_day:
            raise ValueError("No daily archived cards are available.")
        cards = []
        for day in sorted(by_day):
            candidates = sorted(by_day[day], key=lambda row: (row["card_timestamp"], row["id"]))
            cards.append(candidates[(len(candidates) - 1) // 2])
        return cards, start, end

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
        """Use the existing Omreports inline-card API; failures still allow surface analysis."""
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
        """Resolve a well ID to a bounded analysis packet, including locally fetched surface cards."""
        well = self._well(well_id)
        try:
            cards, start, end = self._daily_cards(well_id)
        except ValueError:
            if recent_card is None:
                raise
            cards, start, end = [], None, None
        for card in cards:
            card.update(reference=f"archive:{card['id']}", archive_id=card["id"], roles=["baseline"])
        if recent_card is not None:
            recent = self._recent_card(well_id, recent_card)
            matched = next(
                (
                    card
                    for card in cards
                    if card["card_timestamp"] == recent["card_timestamp"]
                    and card["source_card_id"] == recent["source_card_id"]
                ),
                None,
            )
            if matched:
                recent["archive_id"] = matched["archive_id"]
                recent["roles"] = ["baseline", "recent"]
                cards[cards.index(matched)] = recent
            else:
                cards.append(recent)
        if start is None:
            start = end = datetime.fromtimestamp(cards[-1]["card_timestamp"], self.timezone)
        calculation = self._downhole(well, cards)
        calculated = {
            row["card_id"]: row
            for row in calculation.get("cards", [])
            if isinstance(row, dict) and row.get("card_id") is not None
        }
        packet_cards = []
        for card in cards:
            curve = calculated.get(card["reference"], {})
            downhole = []
            positions, loads = curve.get("positions", []), curve.get("loads", [])
            if curve.get("ok") and isinstance(positions, list) and isinstance(loads, list):
                if len(positions) == len(loads):
                    downhole = physical_points(
                        [
                            {"position": position, "load": load}
                            for position, load in zip(positions, loads)
                        ]
                    )
            header = card["info"] if isinstance(card["info"], dict) else {}
            period = header.get("stroke_period")
            period = period / 100 if isinstance(period, (int, float)) and period > 0 else None
            packet_cards.append(
                {
                    "archive_id": card["archive_id"],
                    "reference": card["reference"],
                    "roles": card["roles"],
                    "pull_timestamp": card["pull_timestamp"],
                    "pumping_window": card.get("pumping_window", {}),
                    "source_card_id": card["source_card_id"],
                    "timestamp": datetime.fromtimestamp(card["card_timestamp"], self.timezone),
                    "header": header,
                    "surface": physical_points(
                        card["data_points"], 100, header.get("number_points") == 0
                    ),
                    "downhole": downhole,
                    "period_s": period,
                    "point_order": "sam1_reverse_time",
                    "warnings": curve.get("warnings", []),
                    "downhole_error": (
                        None
                        if downhole
                        else curve.get(
                            "error", calculation.get("error", "No calculated curve available.")
                        )
                    ),
                }
            )
        return AnalysisRequest(site_id=well["site_id"], well_id=well_id, cards=packet_cards), start, end

    def analyze(self, well_id, recent_card=None):
        """Read source data, reuse an unchanged saved review, or return a new result for persistence."""
        packet, start, end = self.prepare(well_id, recent_card)
        signature = hashlib.sha256(
            json.dumps(
                [packet.model_dump(mode="json"), VERSION, TEMPLATE_VERSION], sort_keys=True
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
            "template_version": TEMPLATE_VERSION,
        }
        if saved and saved["input_signature"] == signature:
            findings = decoded(saved["findings"], {})
            if isinstance(findings, dict) and findings.get("review_status") == "complete":
                return {**metadata, "unchanged": True, "review_status": "complete"}
        result = analyze(packet, self.reviewer)
        return {
            **result,
            **metadata,
            "unchanged": False,
            "window_start_at": start.isoformat(),
            "window_end_at": end.isoformat(),
            "latest_card_at": next(
                (card.timestamp for card in packet.cards if "recent" in card.roles),
                max(card.timestamp for card in packet.cards),
            ).isoformat(),
            "input_cards": [card.model_dump(mode="json") for card in packet.cards],
            "calculation_snapshot": {},
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
        settings.omreports_api_url,
        lambda content: llm_reviewer(settings)(content),
    )
