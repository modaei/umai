from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings
from umai.services.rod_pump_analysis_engine import (
    anomaly_score,
    extract_card_features,
    extract_trend_features,
    fuse_diagnoses,
    health_scores,
    rule_diagnoses,
)


METRICS = {
    "min_load_last_stroke": "Min Load Last Stroke",
    "peak_load_last_stroke": "Peak Load Last Stroke",
    "pump_fillage": "Pump Fillage",
    "yesterday_cycles": "Yesterday Cycles",
    "yesterday_stroke_min": "Yesterday Strokes per minute",
    "max_load": "Max Load Set Point",
    "min_load": "Min Load Set Point",
}
SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class RodPumpAnalysisError(RuntimeError):
    pass


class RodPumpAnalysisClient:
    """Retrieve and deterministically analyze rod-pump trends and dynographs."""

    def __init__(
        self,
        engine: Engine,
        monitoring_url: str,
        timezone: str = "UTC",
        timeout_seconds: float = 20,
        max_data_points: int = 96,
        http_get: Callable[..., Any] = httpx.get,
        prediction_metadata_path: str | None = None,
        batch_workers: int = 2,
    ):
        self.engine = engine
        self.monitoring_url = monitoring_url
        self.timezone = ZoneInfo(timezone)
        self.timeout_seconds = timeout_seconds
        self.max_data_points = max_data_points
        self.http_get = http_get
        self.prediction_metadata_path = prediction_metadata_path
        self.batch_workers = max(1, batch_workers)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RodPumpAnalysisClient":
        settings.validate_database()
        engine = create_engine(
            URL.create(
                "mysql+pymysql",
                username=settings.db_user,
                password=settings.db_password,
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
            ),
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
        )
        return cls(
            engine,
            os.getenv(
                "MONITORING_DATA_API_URL",
                "http://metrics1.ultimatesys.com/render",
            ).strip(),
            timezone=settings.timezone,
            timeout_seconds=float(os.getenv("ROD_PUMP_TIMEOUT_SECONDS", "20")),
            max_data_points=int(os.getenv("ROD_PUMP_MAX_DATA_POINTS", "96")),
            prediction_metadata_path=os.getenv("ROD_PUMP_MODEL_METADATA") or None,
            batch_workers=int(os.getenv("ROD_PUMP_BATCH_WORKERS", "2")),
        )

    def analyze(
        self,
        site_id: int,
        well_name: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Run retrieval, feature extraction, parallel diagnostics, and fusion."""
        start, end = self._interval(start_time, end_time)
        well = self._well(site_id, well_name)
        series, warnings = self._trend_series(well["site_key"], well["well_key"], start, end)
        cards = self._averaged_cards(well["well_id"], start, end)
        notes = self._chart_notes(well["well_id"], end)
        # The active interval drives the report. Older data is used only to
        # determine whether current behavior is unusual for this specific well.
        baseline_start = start - timedelta(days=30)
        baseline_series, baseline_warnings = self._trend_series(
            well["site_key"], well["well_key"], baseline_start, start
        )
        baseline_cards = self._averaged_cards(
            well["well_id"], start - timedelta(days=14), start
        )
        baseline_series, baseline_cards = _exclude_intervention_windows(
            baseline_series, baseline_cards, notes
        )
        card_features = extract_card_features(cards)
        baseline_card_features = extract_card_features(baseline_cards)
        trends = extract_trend_features(series)
        # Rules and anomaly scoring consume the same precomputed evidence. The
        # LLM receives their fused result and is not part of this decision path.
        rules = rule_diagnoses(trends, card_features, notes)
        anomaly = anomaly_score(
            trends,
            card_features,
            baseline_series,
            baseline_card_features,
        )
        prediction = self._prediction_gate()
        diagnoses = fuse_diagnoses(rules, anomaly, prediction)
        scores = health_scores(trends, card_features, diagnoses)
        if baseline_warnings:
            warnings.append(
                f"Historical baseline is partial ({len(baseline_warnings)} metric retrieval failures)."
            )
        if not cards:
            warnings.append("No rod-pump dynographs were found in the analysis interval.")
        if not any(series.values()):
            warnings.append("No rod-pump trend samples were returned by the monitoring service.")
        return {
            "site_id": site_id,
            "well": {"id": well["well_id"], "name": well["well_name"], "key": well["well_key"]},
            "interval": {"start": start.isoformat(), "end": end.isoformat()},
            "current_status": self._current_status(site_id, well["well_key"]),
            "trend_series": series,
            "trend_features": trends,
            # Feature extraction above always uses full-resolution averages.
            # Compact only the LLM-facing representation so every pull remains
            # visible without flooding the tool context with thousands of points.
            "averaged_dynographs": [_compact_card(card) for card in cards],
            "dynograph_features": card_features,
            "chart_note_events": notes,
            "diagnoses": diagnoses,
            "health": scores,
            "anomaly": anomaly,
            "paraffin_prediction": prediction,
            "warnings": warnings,
        }

    def rank_wells(
        self,
        site_id: int,
        as_of_time: str | None = None,
        well_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Analyze every site-scoped rod well and apply deterministic ordering."""
        end = self._parse_time(as_of_time) if as_of_time else datetime.now(self.timezone)
        configured_scope = well_ids is not None
        requested_ids = {int(well_id) for well_id in (well_ids or [])}
        try:
            with self.engine.connect() as connection:
                wells = connection.execute(
                    text("SELECT id, name FROM wells WHERE site_id=:site_id AND LOWER(pump_type)='rod' ORDER BY name"),
                    {"site_id": site_id},
                ).mappings().all()
        except SQLAlchemyError as exc:
            raise RodPumpAnalysisError(f"Could not list rod-pump wells: {exc}") from exc
        available_ids = {int(well["id"]) for well in wells}
        ignored_well_ids = sorted(requested_ids - available_ids)
        if configured_scope:
            wells = [well for well in wells if int(well["id"]) in requested_ids]
        rows = []

        def analyze_well(well):
            name = str(well["name"])
            try:
                result = self.analyze(site_id, name, (end - timedelta(hours=24)).isoformat(), end.isoformat())
                leading = result["diagnoses"][0]
                return {
                    "well": result["well"],
                    "health": result["health"],
                    "leading_diagnosis": leading,
                    "warnings": result["warnings"],
                }
            except Exception as exc:
                return {"well": {"id": int(well["id"]), "name": name}, "health": {"category": "unknown"}, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
            futures = [executor.submit(analyze_well, well) for well in wells]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=_ranking_key)
        return {
            "site_id": site_id,
            "as_of_time": end.isoformat(),
            "well_scope": "configured" if configured_scope else "all_rod_pump_wells",
            "ignored_well_ids": ignored_well_ids,
            "wells": rows,
        }

    def _interval(self, start_raw: str | None, end_raw: str | None) -> tuple[datetime, datetime]:
        end = self._parse_time(end_raw) if end_raw else datetime.now(self.timezone)
        start = self._parse_time(start_raw) if start_raw else end - timedelta(hours=24)
        if start >= end:
            raise RodPumpAnalysisError("start_time must be before end_time.")
        if end - start > timedelta(days=31):
            raise RodPumpAnalysisError("Rod-pump analysis is limited to 31 days.")
        return start, end

    def _parse_time(self, value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RodPumpAnalysisError("Times must use ISO-8601 format.") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=self.timezone)
        return parsed.astimezone(self.timezone)

    def _well(self, site_id: int, well_name: str) -> dict[str, Any]:
        normalized = re.sub(
            r"^well\s+",
            "",
            well_name.strip(),
            flags=re.IGNORECASE,
        ).strip()
        if not normalized:
            raise RodPumpAnalysisError("well_name is required.")
        query = text("""
            SELECT w.id well_id, w.name well_name, w.`key` well_key, s.`key` site_key
            FROM wells w JOIN sites s ON s.id=w.site_id
            WHERE w.site_id=:site_id AND LOWER(w.pump_type)='rod'
            ORDER BY w.name
        """)
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(query, {"site_id": site_id}).mappings().all()
        except SQLAlchemyError as exc:
            raise RodPumpAnalysisError(f"Could not resolve the rod-pump well: {exc}") from exc

        requested = normalized.casefold()
        exact = [
            row for row in rows
            if requested in {
                str(row["well_name"]).strip().casefold(),
                str(row["well_key"]).strip().casefold(),
            }
        ]
        if len(exact) == 1:
            return dict(exact[0])

        # Operators commonly use the complete field identifier (for example
        # 5823) instead of the stored display name HARTZOG DRAW UNIT 5823.
        # Match only a complete terminal token; arbitrary substrings remain
        # invalid so 823 cannot resolve 5823.
        identifier_matches = [
            row for row in rows
            if requested in _rod_pump_well_identifiers(row)
        ]
        if len(identifier_matches) == 1:
            return dict(identifier_matches[0])
        if len(identifier_matches) > 1:
            candidates = ", ".join(str(row["well_name"]) for row in identifier_matches)
            raise RodPumpAnalysisError(
                f"Rod-pump well identifier is ambiguous: {well_name}. "
                f"Use one of these exact names: {candidates}"
            )
        raise RodPumpAnalysisError(
            "Rod-pump well not found. Use an exact well name, telemetry key, "
            f"or complete field identifier: {well_name}"
        )

    def _trend_series(self, site_key: str, well_key: str, start: datetime, end: datetime) -> tuple[dict[str, list], list[str]]:
        """Retrieve the same Graphite POC metrics used by OMetrics trend charts."""
        result: dict[str, list] = {label: [] for label in METRICS.values()}
        warnings: list[str] = []
        prefix = f"MI3.{site_key}.{well_key.replace(' ', '_')}."
        target_to_label = {prefix + metric: label for metric, label in METRICS.items()}
        params = [("target", target) for target in target_to_label]
        params.extend([
            ("from", start.strftime("%H:%M_%Y%m%d")),
            ("until", end.strftime("%H:%M_%Y%m%d")),
            ("format", "json"), ("noNullPoints", "true"),
            ("maxDataPoints", str(self.max_data_points)),
            ("tz", str(self.timezone)),
        ])
        try:
            response = self.http_get(self.monitoring_url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Graphite response is not a series list")
            for index, item in enumerate(payload):
                target = item.get("target") if isinstance(item, dict) else None
                if target not in target_to_label and len(payload) == len(target_to_label):
                    target = list(target_to_label)[index]
                label = target_to_label.get(target)
                if label is None:
                    continue
                result[label] = [
                    {"time": datetime.fromtimestamp(point[1], self.timezone).isoformat(), "value": float(point[0])}
                    for point in item.get("datapoints", [])
                    if len(point) >= 2 and point[0] is not None and point[1] is not None
                ]
            for label, points in result.items():
                if not points:
                    warnings.append(f"{label} returned no samples.")
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            warnings.append(f"Rod-pump trends could not be retrieved: {exc}")
        return result, warnings

    def _averaged_cards(self, well_id: int, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Produce one surface and one downhole average for every pull time."""
        query = text("""
            SELECT d.pull_timestamp, c.card_type, c.down_hole, c.data_points
            FROM rod_pump_monitoring_data d
            JOIN rod_pump_monitoring_data_cards dc ON dc.rod_pump_monitoring_data_id=d.id
            JOIN rod_pump_cards c ON c.id=dc.rod_pump_card_id
            WHERE d.well_id=:well_id AND d.pull_timestamp>=:start_ts AND d.pull_timestamp<=:end_ts
            ORDER BY d.pull_timestamp, c.down_hole, c.id
        """)
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(query, {"well_id": well_id, "start_ts": int(start.timestamp()), "end_ts": int(end.timestamp())}).mappings().all()
        except SQLAlchemyError as exc:
            raise RodPumpAnalysisError(f"Could not retrieve rod-pump cards: {exc}") from exc
        grouped: dict[tuple[int, bool], list] = defaultdict(list)
        types: dict[tuple[int, bool], list[str]] = defaultdict(list)
        for row in rows:
            points = row["data_points"]
            if isinstance(points, str):
                try:
                    points = json.loads(points)
                except json.JSONDecodeError:
                    continue
            if not isinstance(points, list):
                continue
            key = (int(row["pull_timestamp"]), bool(row["down_hole"]))
            grouped[key].append(points)
            types[key].append(str(row["card_type"]))
        cards = []
        for (timestamp, down_hole), card_sets in sorted(grouped.items()):
            # Positions repeat on the upstroke and downstroke. Keying only by
            # position would collapse the two branches and destroy the card
            # shape, so retain the occurrence ordinal and the first card's
            # traversal order while averaging corresponding samples.
            samples: dict[tuple[float, int], dict[str, list[float]]] = {}
            ordered_keys: list[tuple[float, int]] = []
            for points in card_sets:
                occurrences: dict[float, int] = defaultdict(int)
                for point in points:
                    try:
                        position = float(point["position"]) / 100
                        occurrence = occurrences[position]
                        occurrences[position] += 1
                        key = (position, occurrence)
                        if key not in samples:
                            samples[key] = {"positions": [], "loads": []}
                            ordered_keys.append(key)
                        samples[key]["positions"].append(position)
                        samples[key]["loads"].append(float(point["load"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            averaged = [
                {
                    "position": mean(samples[key]["positions"]),
                    "load": mean(samples[key]["loads"]),
                }
                for key in ordered_keys
            ]
            cards.append({
                "pull_time": datetime.fromtimestamp(timestamp, self.timezone).isoformat(),
                "category": "downhole" if down_hole else "surface",
                "contributing_card_count": len(card_sets),
                "card_types": sorted(set(types[(timestamp, down_hole)])),
                "points": averaged,
            })
        return cards

    def _chart_notes(self, well_id: int, end: datetime) -> list[dict[str, Any]]:
        """Load authoritative well events used for context and model labels."""
        query = text("""
            SELECT x_axis_value, chart_name, note
            FROM chart_notes
            WHERE object_type='RodPumpOilWell' AND object_id=:well_id
              AND x_axis_value<=:end_time AND x_axis_value>=:history_start
            ORDER BY x_axis_value
        """)
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(query, {"well_id": well_id, "end_time": end.replace(tzinfo=None), "history_start": (end-timedelta(days=730)).replace(tzinfo=None)}).mappings().all()
        except SQLAlchemyError:
            return []
        return [{"time": str(row["x_axis_value"]), "chart": row["chart_name"], "text": row["note"], "event": _note_event(row["note"] or "")} for row in rows]

    def _current_status(self, site_id: int, well_key: str) -> dict[str, Any]:
        query = text("""
            SELECT dp.data_point_name, d.data
            FROM data_point_data d
            JOIN data_points dp ON dp.id=d.data_point_id
            LEFT JOIN data_points parent ON parent.id=dp.facility_id
            WHERE dp.site_id=:site_id
              AND COALESCE(parent.facility_name, dp.facility_name)=:well_key
              AND dp.data_point_name IN ('stat','status','pump_status')
        """)
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(query, {"site_id": site_id, "well_key": well_key}).mappings().all()
        except SQLAlchemyError:
            return {"available": False}
        values = {}
        for row in rows:
            try:
                payload = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            except (TypeError, json.JSONDecodeError):
                payload = {}
            values[row["data_point_name"]] = {
                "value": payload.get("value") if isinstance(payload, dict) else None,
                "timestamp": payload.get("timestamp") if isinstance(payload, dict) else None,
            }
        return {"available": bool(values), "values": values}

    def _prediction_gate(self) -> dict[str, Any]:
        """Prevent an unvalidated paraffin artifact from influencing diagnosis."""
        if not self.prediction_metadata_path:
            return {"available": False, "reason": "No validated paraffin prediction artifact is configured."}
        try:
            metadata = json.loads(Path(self.prediction_metadata_path).read_text())
            metrics = metadata.get("test_metrics", {})
            precision = float(metrics.get("precision", 0))
            recall = float(metrics.get("recall", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"available": False, "reason": f"Prediction metadata is invalid: {exc}"}
        passed = precision >= 0.50 and recall >= 0.60
        return {
            "available": False,
            "validation_passed": passed,
            "precision": precision,
            "recall": recall,
            "reason": "Validated model scoring is not configured." if passed else "Model did not meet the 50% precision and 60% recall gate.",
        }


class UnavailableRodPumpAnalysisClient:
    def __init__(self, reason: str): self.reason = reason
    def analyze(self, *args, **kwargs): raise RodPumpAnalysisError(self.reason)
    def rank_wells(self, *args, **kwargs): raise RodPumpAnalysisError(self.reason)


def _note_event(note: str) -> str:
    value = note.lower()
    if "paraffin" in value or "wax" in value: return "paraffin"
    if "chemical" in value: return "chemical_treatment"
    if "hot water" in value or "hot oil" in value: return "thermal_treatment"
    if "calibrat" in value or "reset" in value: return "controller_service"
    if any(word in value for word in ("power", "facility", "communication")): return "interruption"
    if any(word in value for word in ("repair", "failure", "drag", "tag")): return "mechanical_event"
    return "unknown"


def _rod_pump_well_identifiers(row: Any) -> set[str]:
    """Return complete terminal identifiers without enabling substring search."""
    identifiers = set()
    for value in (row["well_name"], row["well_key"]):
        tokens = [token for token in re.split(r"[\s_]+", str(value).strip()) if token]
        if tokens:
            identifiers.add(tokens[-1].casefold())
    return identifiers


def _compact_card(card: dict[str, Any], max_points: int = 48) -> dict[str, Any]:
    points = card["points"]
    if len(points) <= max_points:
        compact = points
    else:
        # Evenly retain the full traversal, including both stroke branches and
        # the final point. This is representation-only, not analysis input.
        indices = {
            round(index * (len(points) - 1) / (max_points - 1))
            for index in range(max_points)
        }
        compact = [point for index, point in enumerate(points) if index in indices]
    return {**card, "raw_point_count": len(points), "points": compact}


def _exclude_intervention_windows(
    series: dict[str, list[dict[str, Any]]],
    cards: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Exclude the 24 hours after treatment/service events from baselines."""
    windows = []
    for note in notes:
        if note.get("event") not in {
            "chemical_treatment",
            "thermal_treatment",
            "controller_service",
        }:
            continue
        try:
            start = datetime.fromisoformat(str(note["time"]).replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            windows.append((start, start + timedelta(hours=24)))
        except ValueError:
            continue

    def included(raw_time: str) -> bool:
        try:
            value = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return not any(start <= value <= end for start, end in windows)
        except ValueError:
            return True

    return (
        {
            name: [point for point in points if included(str(point["time"]))]
            for name, points in series.items()
        },
        [card for card in cards if included(str(card["pull_time"]))],
    )


def _ranking_key(row: dict[str, Any]):
    if "error" in row: return (1, 0, 101, 0, row["well"]["name"])
    diagnosis=row["leading_diagnosis"]
    return (0, -SEVERITY_ORDER[diagnosis["severity"]], row["health"]["score"], -diagnosis["confidence"], row["well"]["name"])
