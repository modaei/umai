from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from statistics import mean, median, stdev
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from omai.config.settings import Settings


class DataPointClientError(RuntimeError):
    """Raised when current telemetry values cannot be resolved or queried."""


class DataPointClient:
    """Read current site-scoped values from `data_point_data`."""

    EQUIPMENT_TABLES = {
        "tank": "tanks",
        "pump": "pumps",
        "treater": "treaters",
    }
    ROD_PUMP_TAG_MAPPING = {
        "peak_load_sp": "max_load",
        "min_load_sp": "min_load",
        "peak_load_ls": "peak_load_last_stroke",
        "min_load_ls": "min_load_last_stroke",
    }

    def __init__(
        self,
        engine: Engine,
        timezone_name: str = "UTC",
        monitoring_data_api_url: str = "http://metrics1.ultimatesys.com/render",
        trend_timeout_seconds: float = 20,
        trend_max_data_points: int = 300,
        http_get: Callable[..., Any] = httpx.get,
    ):
        self.engine = engine
        self.monitoring_data_api_url = monitoring_data_api_url
        self.trend_timeout_seconds = trend_timeout_seconds
        self.trend_max_data_points = trend_max_data_points
        self.http_get = http_get
        try:
            self.timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise DataPointClientError(f"Invalid timezone: {timezone_name}") from exc

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataPointClient":
        settings.validate_database()
        url = URL.create(
            drivername="mysql+pymysql",
            username=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
        )
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        )
        return cls(
            engine,
            settings.timezone,
            settings.monitoring_data_api_url,
            settings.data_point_trend_timeout_seconds,
            settings.data_point_trend_max_data_points,
        )

    def get_values(
        self,
        site_id: int,
        facility_name: str | None = None,
        data_point_name: str | None = None,
        device_name: str | None = None,
        facility_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read one point across facilities, or all points for selected facilities."""
        if facility_name is not None and facility_names is not None:
            raise DataPointClientError("Supply facility_name or facility_names, not both.")
        selectors = [facility_name] if facility_name is not None else facility_names
        if selectors is not None and (not selectors or any(not _normalize_name(name) for name in selectors)):
            raise DataPointClientError("Facility selectors must be non-empty names.")
        if data_point_name is not None and not _normalize_name(data_point_name):
            raise DataPointClientError("data_point_name must be a non-empty name.")
        if selectors is None and data_point_name is None:
            raise DataPointClientError("Provide a facility selector or data_point_name (or both).")
        if device_name is not None and not _normalize_name(device_name):
            raise DataPointClientError("device_name must be a non-empty name.")

        rows = self._site_data_points(site_id)
        available = {str(row["effective_facility_name"]) for row in rows if row.get("effective_facility_name")}
        facilities = []
        # Resolve every explicit selector before returning values: an ambiguous
        # shorthand must never silently expand to several facilities.
        for selector in selectors or []:
            matches = _ranked_matches(_normalize_name(selector), available, mode="suffix")
            if not matches:
                return _not_found("facility", selector)
            if len(matches) > 1:
                return _ambiguous("facility", selector, matches)
            if matches[0] not in facilities:
                facilities.append(matches[0])
        if selectors is None:
            facilities = sorted(available)

        # Preserve the established single-facility/point default device.
        # Broad queries include every device unless explicitly filtered.
        legacy_scope = facility_name is not None and data_point_name is not None
        results = [
            self._current_facility_values(rows, facility, data_point_name, device_name, legacy_scope)
            for facility in facilities
        ]
        if legacy_scope:
            return results[0]
        values = [value for result in results for value in result.get("values", [])]
        failures = [result for result in results if result["status"] != "found"]
        return {
            "status": "partial" if values and failures else "found" if values else "not_found",
            "scope": "selected_facilities" if selectors is not None else "all_facilities",
            "facility_names": facilities,
            "requested_data_point_name": data_point_name,
            "match_count": len(values),
            "values": values,
            "facility_results": [
                {key: value for key, value in result.items() if key != "values"}
                for result in results
            ],
        }

    def _current_facility_values(
        self,
        rows: list[dict[str, Any]],
        facility: str,
        point: str | None,
        device: str | None,
        default_device: bool,
    ) -> dict[str, Any]:
        selected = [row for row in rows if row["effective_facility_name"] == facility]
        device_defaulted = default_device and device is None
        requested_device = facility if device_defaulted else device
        metadata = {"facility_name": facility, "device_defaulted": device_defaulted}
        if requested_device is not None:
            matches = _ranked_matches(
                _normalize_name(requested_device),
                {str(row["device_name"]) for row in selected if row.get("device_name")},
                mode="exact",
            )
            if not matches:
                return {**_not_found("device", requested_device), **metadata}
            if len(matches) > 1:
                return {**_ambiguous("device", requested_device, matches), **metadata}
            metadata["device_name"] = matches[0]
            selected = [row for row in selected if _normalize_name(row.get("device_name")) == _normalize_name(matches[0])]
        # Match independently per device so an exact name on one controller
        # cannot suppress valid prefix matches on another controller.
        matched_rows = []
        devices = sorted({row.get("device_name") for row in selected}, key=lambda name: str(name or ""))
        for device_name in devices:
            device_rows = [row for row in selected if row.get("device_name") == device_name]
            if point is None:
                matched_rows.extend(device_rows)
                continue
            matches = _ranked_matches(
                _normalize_name(point), {str(row["data_point_name"]) for row in device_rows}, mode="prefix"
            )
            names = {_normalize_name(name) for name in matches}
            matched_rows.extend(row for row in device_rows if _normalize_name(row["data_point_name"]) in names)
        if not matched_rows:
            return {**_not_found("data_point", point or "all data points"), **metadata}
        values = [self._compact_value(row) for row in matched_rows]
        values.sort(key=lambda item: (str(item["device_name"] or ""), item["data_point_name"]))
        return {
            "status": "found", **metadata,
            "requested_data_point_name": point,
            "match_count": len(values), "values": values,
        }

    def analyze_trend(
        self,
        site_id: int,
        facility_name: str,
        data_point_name: str,
        days: int,
        device_name: str | None = None,
        equipment_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a telemetry point, fetch its Graphite trend, and summarize it.

        Facility/device/data-point selection is site scoped. Equipment selectors
        such as "tank 10-1 Oil level" first resolve the equipment row, then use
        that row's configured data-point facility and device.
        """
        if days <= 0 or days > 366:
            raise DataPointClientError("days must be between 1 and 366.")
        if (start_date and not end_date) or (end_date and not start_date):
            raise DataPointClientError("Both start_date and end_date are required for a date range.")
        resolved = self._resolve_data_point(
            site_id=site_id,
            facility_name=facility_name,
            data_point_name=data_point_name,
            device_name=device_name,
            equipment_type=equipment_type,
            allow_fuzzy=True,
        )
        if resolved.get("status") != "found":
            return resolved

        if start_date and end_date:
            start_day = _parse_iso_date(start_date)
            end_day = _parse_iso_date(end_date)
            if start_day > end_day:
                raise DataPointClientError("start_date must be before or equal to end_date.")
            if end_day - start_day > timedelta(days=366):
                raise DataPointClientError("Trend date ranges are limited to 366 days.")
            start = datetime.combine(start_day, time.min, tzinfo=self.timezone)
            end = datetime.combine(end_day, time.max.replace(microsecond=0), tzinfo=self.timezone)
            interval_label = f"{start.strftime('%m/%d/%Y')} to {end.strftime('%m/%d/%Y')}"
        else:
            end = datetime.now(self.timezone)
            start = end - timedelta(days=days)
            interval_label = f"last {days} day(s)"
        samples = self._fetch_trend(site_id, resolved["data_point"], start, end)
        return {
            "status": "found",
            "facility_name": resolved["facility_name"],
            "device_name": resolved["device_name"],
            "device_defaulted": resolved["device_defaulted"],
            "equipment": resolved.get("equipment"),
            "data_point_name": resolved["data_point"]["data_point_name"],
            "tag": resolved["data_point"].get("tag"),
            "days": days,
            "start_date": start_date,
            "end_date": end_date,
            "interval_label": interval_label,
            "interval": {
                "start": start.strftime("%m/%d/%Y %H:%M:%S %Z"),
                "end": end.strftime("%m/%d/%Y %H:%M:%S %Z"),
            },
            "sample_count": len(samples),
            "summary": self._summarize_samples(samples),
        }

    def _site_data_points(self, site_id: int) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT COALESCE(parent.facility_name, dp.facility_name) AS effective_facility_name,
                dp.id,
                dp.facility_id,
                COALESCE(parent.facility_name, dp.facility_name) AS graphite_facility_name,
                dp.device_name,
                dp.data_point_name,
                dp.tag,
                dp.type AS data_point_type,
                dp.active,
                current_value.data,
                current_value.last_update
            FROM data_points dp
            LEFT JOIN data_points parent ON parent.id = dp.facility_id
            LEFT JOIN data_point_data current_value ON current_value.data_point_id = dp.id
            WHERE dp.site_id = :site_id
              AND dp.data_point_name IS NOT NULL
            ORDER BY effective_facility_name, dp.device_name, dp.data_point_name
            """
        )
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, {"site_id": site_id})
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as exc:
            raise DataPointClientError(f"Database query failed: {exc}") from exc

    def _resolve_data_point(
        self,
        *,
        site_id: int,
        facility_name: str,
        data_point_name: str,
        device_name: str | None,
        equipment_type: str | None,
        allow_fuzzy: bool,
    ) -> dict[str, Any]:
        """Resolve selectors to one data-point row without reading Graphite."""
        facility_query = _normalize_name(facility_name)
        data_point_query = _normalize_name(data_point_name)
        if not facility_query:
            raise DataPointClientError("facility_name is required.")
        if not data_point_query:
            raise DataPointClientError("data_point_name is required.")

        rows = self._site_data_points(site_id)
        equipment = self._resolve_equipment(site_id, facility_name, equipment_type, allow_fuzzy)
        if equipment is not None:
            facility_rows = [
                row
                for row in rows
                if row.get("facility_id") == equipment["data_point_facility_id"]
                and _normalize_name(row.get("device_name"))
                == _normalize_name(equipment["data_point_device_name"])
            ]
            if not facility_rows:
                return {
                    **_not_found("equipment data points", equipment["name"]),
                    "equipment": equipment,
                }
            resolved_facility = str(facility_rows[0]["effective_facility_name"])
            resolved_device = str(equipment["data_point_device_name"])
            device_defaulted = device_name is None or not str(device_name).strip()
        else:
            facility_matches = _ranked_matches(
                facility_query,
                {str(row["effective_facility_name"]) for row in rows},
                mode="suffix",
                allow_fuzzy=allow_fuzzy,
            )
            if not facility_matches:
                return _not_found("facility", facility_name)
            if len(facility_matches) > 1:
                return _ambiguous("facility", facility_name, facility_matches)
            resolved_facility = facility_matches[0]
            facility_rows = [
                row
                for row in rows
                if _normalize_name(row["effective_facility_name"])
                == _normalize_name(resolved_facility)
            ]

            device_defaulted = device_name is None or not str(device_name).strip()
            requested_device = resolved_facility if device_defaulted else str(device_name)
            device_matches = _ranked_matches(
                _normalize_name(requested_device),
                {str(row["device_name"]) for row in facility_rows if row.get("device_name")},
                mode="exact",
                allow_fuzzy=allow_fuzzy,
            )
            if not device_matches:
                return {
                    **_not_found("device", requested_device),
                    "facility_name": resolved_facility,
                    "device_defaulted": device_defaulted,
                }
            if len(device_matches) > 1:
                return {
                    **_ambiguous("device", requested_device, device_matches),
                    "facility_name": resolved_facility,
                    "device_defaulted": device_defaulted,
                }
            resolved_device = device_matches[0]
            facility_rows = [
                row
                for row in facility_rows
                if _normalize_name(row.get("device_name")) == _normalize_name(resolved_device)
            ]

        point_matches = _ranked_matches(
            data_point_query,
            {str(row["data_point_name"]) for row in facility_rows if row.get("data_point_name")},
            mode="prefix",
            allow_fuzzy=allow_fuzzy,
        )
        if not point_matches:
            return {
                **_not_found("data_point", data_point_name),
                "facility_name": resolved_facility,
                "device_name": resolved_device,
                "device_defaulted": device_defaulted,
                **({"equipment": equipment} if equipment else {}),
            }
        if len(point_matches) > 1:
            return {
                **_ambiguous("data_point", data_point_name, point_matches),
                "facility_name": resolved_facility,
                "device_name": resolved_device,
                "device_defaulted": device_defaulted,
                **({"equipment": equipment} if equipment else {}),
            }
        point_name = point_matches[0]
        data_point = next(
            row
            for row in facility_rows
            if _normalize_name(row.get("data_point_name")) == _normalize_name(point_name)
        )
        return {
            "status": "found",
            "facility_name": resolved_facility,
            "device_name": resolved_device,
            "device_defaulted": device_defaulted,
            "data_point": data_point,
            **({"equipment": equipment} if equipment else {}),
        }

    def _resolve_equipment(
        self,
        site_id: int,
        equipment_name: str,
        equipment_type: str | None,
        allow_fuzzy: bool,
    ) -> dict[str, Any] | None:
        """Resolve configured monitored equipment when a known type is supplied."""
        normalized_type = _normalize_name(equipment_type)
        if normalized_type not in self.EQUIPMENT_TABLES:
            return None
        table = self.EQUIPMENT_TABLES[normalized_type]
        query = text(
            f"""
            SELECT id, name, `key` AS equipment_key,
                data_point_facility_id, data_point_device_name
            FROM {table}
            WHERE site_id = :site_id
              AND data_point_facility_id IS NOT NULL
              AND data_point_device_name IS NOT NULL
            ORDER BY name
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = [dict(row._mapping) for row in connection.execute(query, {"site_id": site_id})]
        except SQLAlchemyError:
            return None
        names = {
            str(row["name"])
            for row in rows
            if row.get("name")
        } | {
            str(row["equipment_key"])
            for row in rows
            if row.get("equipment_key")
        }
        matches = _ranked_matches(
            _normalize_name(equipment_name),
            names,
            mode="suffix",
            allow_fuzzy=allow_fuzzy,
        )
        matched_names = {_normalize_name(match) for match in matches}
        matched_rows = [
            item
            for item in rows
            if _normalize_name(item.get("name")) in matched_names
            or _normalize_name(item.get("equipment_key")) in matched_names
        ]
        unique_rows = {int(item["id"]): item for item in matched_rows}
        if len(unique_rows) != 1:
            return None
        row = next(iter(unique_rows.values()))
        return {
            "type": normalized_type,
            "id": row["id"],
            "name": row["name"],
            "key": row.get("equipment_key"),
            "data_point_facility_id": row["data_point_facility_id"],
            "data_point_device_name": row["data_point_device_name"],
        }

    def _fetch_trend(
        self,
        site_id: int,
        data_point: dict[str, Any],
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch numeric samples from the same Graphite path Ometrics uses."""
        tag = str(data_point.get("tag") or "").strip()
        if not tag:
            raise DataPointClientError("The matched data point has no telemetry tag.")
        site_key = self._site_key(site_id)
        target_tag = self.ROD_PUMP_TAG_MAPPING.get(tag, tag)
        object_name = str(data_point["graphite_facility_name"]).replace(" ", "_")
        target = f"MI3.{site_key}.{object_name}.{target_tag}"
        response = self.http_get(
            self.monitoring_data_api_url,
            params={
                "target": target,
                "from": start.strftime("%H:%M_%Y%m%d"),
                "until": end.strftime("%H:%M_%Y%m%d"),
                "format": "json",
                "noNullPoints": "true",
                "maxDataPoints": self.trend_max_data_points,
                "tz": self.timezone.key,
            },
            timeout=self.trend_timeout_seconds,
        )
        try:
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise DataPointClientError(f"Monitoring data API request failed: {exc}") from exc
        samples = []
        for value, timestamp in (payload[0].get("datapoints", []) if payload else []):
            if value is None or timestamp is None:
                continue
            try:
                numeric_value = float(value)
                sample_time = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).astimezone(self.timezone)
            except (TypeError, ValueError, OSError, OverflowError):
                continue
            samples.append(
                {
                    "time": sample_time.strftime("%m/%d/%Y %H:%M:%S %Z"),
                    "timestamp": sample_time.isoformat(),
                    "value": numeric_value,
                }
            )
        return samples

    def _site_key(self, site_id: int) -> str:
        try:
            with self.engine.connect() as connection:
                row = connection.execute(
                    text("SELECT `key` FROM sites WHERE id=:site_id"),
                    {"site_id": site_id},
                ).first()
        except SQLAlchemyError as exc:
            raise DataPointClientError(f"Could not resolve site key: {exc}") from exc
        if row is None or not row[0]:
            raise DataPointClientError(f"No site key was found for site_id={site_id}.")
        return str(row[0])

    def _is_rod_pump_data_point(self, site_id: int, data_point: dict[str, Any]) -> bool:
        try:
            with self.engine.connect() as connection:
                row = connection.execute(
                    text(
                        """
                        SELECT 1
                        FROM wells
                        WHERE site_id=:site_id
                          AND `key`=:facility_name
                          AND LOWER(COALESCE(pump_type, ''))='rod'
                        LIMIT 1
                        """
                    ),
                    {
                        "site_id": site_id,
                        "facility_name": data_point.get("graphite_facility_name"),
                    },
                ).first()
        except SQLAlchemyError:
            return False
        return row is not None

    def _summarize_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Create a deterministic statistical trend summary for LLM/user output."""
        if not samples:
            return {"status": "no_data", "anomalies": ["No trend samples were returned."]}
        values = [float(sample["value"]) for sample in samples]
        first = samples[0]
        latest = samples[-1]
        change = values[-1] - values[0]
        change_percent = None if values[0] == 0 else (change / abs(values[0])) * 100
        typical_low = _percentile(values, 10)
        typical_high = _percentile(values, 90)
        typical_width = typical_high - typical_low
        center = median(values)
        pattern = _trend_pattern(typical_width, center)
        variability = _variability_level(typical_width, center)
        segments = _trend_segments(samples, values)
        significant_changes = _significant_changes(samples, values)
        trend = "stable"
        if change_percent is not None and abs(change_percent) >= 5:
            trend = "rising" if change > 0 else "falling"
        elif change_percent is None and abs(change) > 0:
            trend = "rising" if change > 0 else "falling"
        anomalies = self._detect_anomalies(samples, values)
        average_value = mean(values)
        average_distorted = bool(
            anomalies
            and center
            and abs(average_value - center) / abs(center) >= 0.05
        )
        return {
            "status": "ok",
            "first": first,
            "latest": latest,
            "min": min(values),
            "max": max(values),
            "average": average_value,
            "median": center,
            "typical_low": typical_low,
            "typical_high": typical_high,
            "typical_band": {
                "low": typical_low,
                "high": typical_high,
            },
            "pattern": pattern,
            "change": change,
            "change_percent": change_percent,
            "trend": trend,
            "variability": variability,
            "segments": segments,
            "significant_changes": significant_changes,
            "overall_shape": _overall_shape(
                trend=trend,
                pattern=pattern,
                variability=variability,
                segments=segments,
                change_percent=change_percent,
                significant_changes=significant_changes,
            ),
            "average_distorted_by_outliers": average_distorted,
            "anomalies": anomalies,
        }

    def _detect_anomalies(
        self, samples: list[dict[str, Any]], values: list[float]
    ) -> list[str]:
        """Flag simple trend anomalies without pretending to diagnose root cause."""
        anomalies: list[str] = []
        if len(values) < 5:
            return ["Too few samples were returned for anomaly detection."]
        normal_floor = _percentile(values, 10)
        normal_center = median(values)
        zero_events = [
            sample
            for sample in samples
            if abs(float(sample["value"])) < 0.000001
        ]
        if zero_events and normal_center > 0 and normal_floor > 0:
            first_zero = zero_events[0]
            count_text = "" if len(zero_events) == 1 else f" ({len(zero_events)} samples)"
            anomalies.append(
                f"Drop to zero at {first_zero.get('time')}{count_text}; this looks abnormal compared with the normal band."
            )
        high_outliers = _high_outlier_windows(samples, values)
        for window in high_outliers:
            if window["start_time"] == window["end_time"]:
                anomalies.append(
                    f"Abnormal high spike at {window['start_time']}, peaking at {_format_raw_number(window['peak_value'])}."
                )
            else:
                anomalies.append(
                    f"Abnormal high spike/plateau from {window['start_time']} to {window['end_time']}, peaking at {_format_raw_number(window['peak_value'])}."
                )
        if max(values) == min(values):
            anomalies.append("The value was flat for the whole returned interval.")
        timestamps = [
            datetime.fromisoformat(str(sample["timestamp"]))
            for sample in samples
            if sample.get("timestamp")
        ]
        if len(timestamps) >= 3:
            gaps = [
                (timestamps[index] - timestamps[index - 1]).total_seconds()
                for index in range(1, len(timestamps))
            ]
            typical_gap = median(gaps)
            if typical_gap > 0 and max(gaps) > max(3600, typical_gap * 3):
                anomalies.append("The returned samples contain a larger-than-normal data gap.")
        if len(values) >= 6:
            deltas = [
                abs(values[index] - values[index - 1])
                for index in range(1, len(values))
            ]
            normal_delta = median(delta for delta in deltas if delta > 0) if any(delta > 0 for delta in deltas) else 0
            if normal_delta > 0:
                largest_delta = max(deltas)
                if largest_delta >= normal_delta * 5:
                    index = deltas.index(largest_delta) + 1
                    if not (
                        _is_near_reported_high_outlier(samples[index], high_outliers)
                        or _is_near_reported_high_outlier(samples[index - 1], high_outliers)
                    ):
                        direction = "jump" if values[index] > values[index - 1] else "drop"
                        anomalies.append(
                            f"Large point-to-point {direction} near {samples[index].get('time')}."
                        )
        if len(values) >= 6:
            baseline = values[:-1]
            spread = stdev(baseline) if len(set(baseline)) > 1 else 0
            if spread > 0:
                z_score = (values[-1] - mean(baseline)) / spread
                if abs(z_score) >= 3:
                    direction = "spike" if z_score > 0 else "drop"
                    anomalies.append(f"The latest value is a statistical {direction} versus earlier samples.")
        return anomalies

    def _compact_value(self, row: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        data_error = None
        if row.get("data"):
            try:
                decoded = json.loads(row["data"])
                if isinstance(decoded, dict):
                    payload = decoded
                else:
                    data_error = "Current value payload is not a JSON object."
            except (TypeError, ValueError):
                data_error = "Current value payload is invalid JSON."
        timestamp = payload.get("timestamp") or row.get("last_update")
        value = {
            "facility_name": row["effective_facility_name"],
            "device_name": row.get("device_name"),
            "data_point_name": row["data_point_name"],
            "value": payload.get("value"),
            "value_available": "value" in payload,
            "value_type": payload.get("type") or row.get("data_point_type"),
            "received_at": self._format_timestamp(timestamp),
            "active": bool(row.get("active")),
        }
        if data_error:
            value["data_error"] = data_error
        return value

    def _format_timestamp(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            timestamp = float(value)
            if timestamp > 1_000_000_000_000:
                timestamp /= 1000
            received = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return None
        return received.astimezone(self.timezone).strftime("%m/%d/%Y %H:%M:%S %Z")


class UnavailableDataPointClient:
    """Data-point client used when database settings are unavailable."""

    def __init__(self, reason: str):
        self.reason = reason

    def get_values(
        self,
        site_id: int,
        facility_name: str | None = None,
        data_point_name: str | None = None,
        device_name: str | None = None,
        facility_names: list[str] | None = None,
    ) -> dict[str, Any]:
        raise DataPointClientError(self.reason)

    def analyze_trend(
        self,
        site_id: int,
        facility_name: str,
        data_point_name: str,
        days: int,
        device_name: str | None = None,
        equipment_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        raise DataPointClientError(self.reason)


def _normalize_name(value: Any) -> str:
    """Normalize human labels and telemetry selectors to underscore tokens."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())).strip("_")


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataPointClientError("Dates must use YYYY-MM-DD format.") from exc


def _percentile(values: list[float], percentile: float) -> float:
    """Return an interpolated percentile for a non-empty numeric list."""
    if not values:
        raise DataPointClientError("Cannot calculate percentile for an empty value list.")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def _trend_pattern(typical_width: float, center: float) -> str:
    """Classify normal behavior separately from first-to-last drift."""
    if center == 0:
        return "flat" if typical_width == 0 else "variable"
    relative_width = abs(typical_width) / abs(center)
    if relative_width <= 0.05:
        return "mostly_flat"
    return "oscillating"


def _variability_level(typical_width: float, center: float) -> str:
    """Classify how wide the normal operating band is relative to its level."""
    if center == 0:
        return "low" if typical_width == 0 else "high"
    relative_width = abs(typical_width) / abs(center)
    if relative_width <= 0.03:
        return "low"
    if relative_width <= 0.12:
        return "moderate"
    return "high"


def _trend_segments(
    samples: list[dict[str, Any]],
    values: list[float],
) -> list[dict[str, Any]]:
    """Summarize beginning/middle/end windows so answers read like a graph review."""
    if not samples:
        return []
    if len(samples) < 3:
        return [
            {
                "label": "period",
                "start_time": samples[0].get("time"),
                "end_time": samples[-1].get("time"),
                "median": median(values),
                "low": min(values),
                "high": max(values),
            }
        ]
    size = max(1, len(samples) // 3)
    ranges = [
        ("beginning", 0, size),
        ("middle", size, size * 2),
        ("end", size * 2, len(samples)),
    ]
    segments = []
    for label, start, end in ranges:
        segment_samples = samples[start:end]
        segment_values = values[start:end]
        if not segment_samples:
            continue
        segments.append(
            {
                "label": label,
                "start_time": segment_samples[0].get("time"),
                "end_time": segment_samples[-1].get("time"),
                "median": median(segment_values),
                "low": min(segment_values),
                "high": max(segment_values),
            }
        )
    return segments


def _significant_changes(
    samples: list[dict[str, Any]],
    values: list[float],
) -> list[dict[str, Any]]:
    """Find the largest point-to-point changes worth mentioning in prose."""
    if len(values) < 2:
        return []
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    non_zero_deltas = [abs(delta) for delta in deltas if abs(delta) > 0]
    if not non_zero_deltas:
        return []
    normal_delta = median(non_zero_deltas)
    typical_low = _percentile(values, 10)
    typical_high = _percentile(values, 90)
    typical_width = max(typical_high - typical_low, 0)
    threshold = max(normal_delta * 4, typical_width * 0.20)
    if threshold <= 0:
        return []

    changes = []
    for index, delta in enumerate(deltas, start=1):
        if abs(delta) < threshold:
            continue
        previous_value = values[index - 1]
        current_value = values[index]
        percent = None if previous_value == 0 else (delta / abs(previous_value)) * 100
        changes.append(
            {
                "direction": "spike" if delta > 0 else "drop",
                "time": samples[index].get("time"),
                "from": previous_value,
                "to": current_value,
                "change": delta,
                "change_percent": percent,
            }
        )

    changes.sort(key=lambda item: abs(float(item["change"])), reverse=True)
    return changes[:3]


def _overall_shape(
    *,
    trend: str,
    pattern: str,
    variability: str,
    segments: list[dict[str, Any]],
    change_percent: float | None,
    significant_changes: list[dict[str, Any]],
) -> str:
    """Classify the visible graph shape from segment medians and major changes."""
    if len(segments) >= 3:
        early = float(segments[0]["median"])
        middle = float(segments[1]["median"])
        late = float(segments[2]["median"])
        scale = max(abs(early), abs(middle), abs(late), 1.0)
        early_to_mid = (middle - early) / scale
        mid_to_late = (late - middle) / scale
        early_to_late = (late - early) / scale
        if early_to_mid <= -0.04 and mid_to_late >= 0.04:
            return "decline_then_recovery"
        if early_to_mid >= 0.04 and mid_to_late <= -0.04:
            return "increase_then_decline"
        if abs(early_to_mid) >= 0.04 and abs(mid_to_late) < 0.025:
            return "step_increase" if early_to_mid > 0 else "step_decrease"
        if abs(mid_to_late) >= 0.04 and abs(early_to_mid) < 0.025:
            return "late_increase" if mid_to_late > 0 else "late_decrease"
        if abs(early_to_late) >= 0.05:
            return "gradual_increase" if early_to_late > 0 else "gradual_decrease"
    if significant_changes and variability == "high":
        return "highly_variable"
    if pattern == "mostly_flat":
        return "mostly_flat"
    if trend == "rising" and (change_percent is None or abs(change_percent) >= 5):
        return "gradual_increase"
    if trend == "falling" and (change_percent is None or abs(change_percent) >= 5):
        return "gradual_decrease"
    if pattern == "oscillating":
        return "oscillating"
    return "mostly_flat"


def _high_outlier_windows(
    samples: list[dict[str, Any]], values: list[float]
) -> list[dict[str, Any]]:
    """Group consecutive high values that sit well above the normal band."""
    if len(values) < 6:
        return []
    center = median(values)
    p10 = _percentile(values, 10)
    p90 = _percentile(values, 90)
    band_width = p90 - p10
    if center <= 0:
        return []
    threshold = p90 + max(band_width * 3, abs(center) * 0.10)
    windows: list[dict[str, Any]] = []
    current: list[tuple[dict[str, Any], float]] = []
    for sample, value in zip(samples, values):
        if value > threshold:
            current.append((sample, value))
            continue
        if current:
            windows.append(_outlier_window(current))
            current = []
    if current:
        windows.append(_outlier_window(current))
    return windows


def _outlier_window(items: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    peak_sample, peak_value = max(items, key=lambda item: item[1])
    return {
        "start_time": items[0][0].get("time"),
        "end_time": items[-1][0].get("time"),
        "peak_time": peak_sample.get("time"),
        "peak_value": peak_value,
    }


def _is_near_reported_high_outlier(
    sample: dict[str, Any], windows: list[dict[str, Any]]
) -> bool:
    sample_time = sample.get("time")
    return any(
        sample_time == window.get("start_time") or sample_time == window.get("end_time")
        for window in windows
    )


def _format_raw_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:,.2f}".rstrip("0").rstrip(".")


def _ranked_matches(
    query: str, names: set[str], *, mode: str, allow_fuzzy: bool = False
) -> list[str]:
    """Return exact matches or the closest safe prefix/suffix matches."""
    normalized = [(name, _normalize_name(name)) for name in names if _normalize_name(name)]
    exact = sorted(name for name, candidate in normalized if candidate == query)
    if exact:
        return exact
    if mode == "exact" and not allow_fuzzy:
        return exact
    if mode == "suffix":
        candidates = [
            (name, len(candidate.split("_")) - len(query.split("_")))
            for name, candidate in normalized
            if candidate.endswith(f"_{query}")
        ]
    else:
        candidates = [
            (name, len(candidate.split("_")) - len(query.split("_")))
            for name, candidate in normalized
            if candidate.startswith(f"{query}_")
        ]
    if not candidates:
        if not allow_fuzzy:
            return []
        scored = [
            (name, SequenceMatcher(None, query, candidate).ratio())
            for name, candidate in normalized
        ]
        if not scored:
            return []
        best_score = max(score for _, score in scored)
        if best_score < 0.78:
            return []
        close = sorted(name for name, score in scored if best_score - score < 0.04)
        return close
    closest = min(score for _, score in candidates)
    return sorted(name for name, score in candidates if score == closest)


def _not_found(selector: str, requested: str) -> dict[str, Any]:
    return {
        "status": "not_found",
        "selector": selector,
        "requested": requested,
        "candidates": [],
    }


def _ambiguous(selector: str, requested: str, candidates: list[str]) -> dict[str, Any]:
    return {
        "status": "ambiguous",
        "selector": selector,
        "requested": requested,
        "candidates": candidates,
    }
