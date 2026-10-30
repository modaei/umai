from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


# Mirrors RodPumpHealthService::saved(), which expires refresh indicators after 15 minutes.
REFRESH_EXPIRY = timedelta(minutes=15)
HEALTH_ORDER = {"problem": 0, "watch": 1, "healthy": 2}
SUMMARY_PREVIEW_CHARS = 240


class RodPumpHealthError(RuntimeError):
    pass


class RodPumpHealthClient:
    """Read the saved rod-pump health analyses that UMetriqs shows on the health page.

    UMetriqs owns refreshing these results (RodPumpHealthService). This client
    never writes and never starts an analysis.
    """

    def __init__(self, engine: Engine, timezone: str = "UTC"):
        self.engine = engine
        self.timezone = ZoneInfo(timezone)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RodPumpHealthClient":
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
        return cls(engine, timezone=settings.timezone)

    def well_health(self, site_id: int, well_name: str) -> dict[str, Any]:
        """Return the saved health review for one monitored rod-pump well."""
        well = self.resolve_well(site_id, well_name)
        row = self._analysis_rows(site_id, [well["id"]]).get(well["id"])
        if row is None:
            return {
                "well": well,
                "available": False,
                "reason": "No saved health analysis exists for this well yet.",
            }
        return {"well": well, "available": True, **self._saved_review(row)}

    def site_health(self, site_id: int) -> dict[str, Any]:
        """Return a compact saved-health row for every monitored rod-pump well."""
        wells = self._monitored_wells(site_id)
        rows = self._analysis_rows(site_id, [well["id"] for well in wells])
        results = []
        for well in wells:
            row = rows.get(well["id"])
            if row is None:
                results.append({"well": well, "available": False})
                continue
            review = self._saved_review(row)
            summary = review.get("summary") or ""
            results.append({
                "well": well,
                "available": True,
                "health_status": review["health_status"],
                "review_status": review["review_status"],
                "summary": _preview(summary),
                "finding_titles": [item["title"] for item in review["findings"] if item.get("title")],
                "surface_load_status": (review.get("surface_load") or {}).get("status"),
                "analyzed_at": review["analyzed_at"],
                "card_captured_at": review["card_captured_at"],
            })
        results.sort(key=_site_order)
        counts: dict[str, int] = {}
        for item in results:
            key = item.get("health_status") or ("not_analyzed" if not item["available"] else "unavailable")
            counts[key] = counts.get(key, 0) + 1
        return {"site_id": site_id, "well_count": len(results), "counts": counts, "wells": results}

    def resolve_well(self, site_id: int, well_name: str) -> dict[str, Any]:
        """Resolve a monitored rod-pump well by exact name, key, or complete field identifier."""
        normalized = re.sub(r"^well\s+", "", well_name.strip(), flags=re.IGNORECASE).strip()
        if not normalized:
            raise RodPumpHealthError("well_name is required.")
        rows = self._rod_wells(site_id)
        requested = normalized.casefold()
        exact = [
            row for row in rows
            if requested in {row["name"].strip().casefold(), row["key"].strip().casefold()}
        ]
        matches = exact or [row for row in rows if requested in _well_identifiers(row)]
        if not matches:
            raise RodPumpHealthError(
                "Rod-pump well not found. Use an exact well name, telemetry key, "
                f"or complete field identifier: {well_name}"
            )
        if len(matches) > 1:
            candidates = ", ".join(row["name"] for row in matches)
            raise RodPumpHealthError(
                f"Rod-pump well identifier is ambiguous: {well_name}. "
                f"Use one of these exact names: {candidates}"
            )
        well = matches[0]
        if well["id"] not in self._monitored_well_ids(site_id):
            raise RodPumpHealthError(
                f"{well['name']} is not configured for rod-pump health monitoring."
            )
        return well

    def _rod_wells(self, site_id: int) -> list[dict[str, Any]]:
        query = text("""
            SELECT id, name, `key`
            FROM wells
            WHERE site_id=:site_id AND LOWER(pump_type)='rod'
            ORDER BY name
        """)
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(query, {"site_id": site_id}).mappings().all()
        except SQLAlchemyError as exc:
            raise RodPumpHealthError(f"Could not read rod-pump wells: {exc}") from exc
        return [
            {"id": int(row["id"]), "name": str(row["name"]), "key": str(row["key"] or "")}
            for row in rows
        ]

    def _monitored_well_ids(self, site_id: int) -> set[int]:
        """Health covers the wells in the site's rod_pump_monitoring configuration, as in UMetriqs."""
        query = text("""
            SELECT related_entities
            FROM report_configurations
            WHERE site_id=:site_id AND function_name='rod_pump_monitoring'
            ORDER BY id
            LIMIT 1
        """)
        try:
            with self.engine.connect() as connection:
                raw = connection.execute(query, {"site_id": site_id}).scalar()
        except SQLAlchemyError as exc:
            raise RodPumpHealthError(f"Could not read rod-pump monitoring configuration: {exc}") from exc
        entities = _decoded(raw, {})
        well_ids = entities.get("well_ids", []) if isinstance(entities, dict) else []
        result = set()
        for well_id in well_ids if isinstance(well_ids, list) else []:
            try:
                result.add(int(well_id))
            except (TypeError, ValueError):
                continue
        return result

    def _monitored_wells(self, site_id: int) -> list[dict[str, Any]]:
        monitored = self._monitored_well_ids(site_id)
        return [well for well in self._rod_wells(site_id) if well["id"] in monitored]

    def _analysis_rows(self, site_id: int, well_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not well_ids:
            return {}
        placeholders = ", ".join(f":well_{index}" for index in range(len(well_ids)))
        query = text(f"""
            SELECT a.well_id, a.status, a.latest_card_at, a.analyzed_at,
                   a.refresh_started_at, a.last_refresh_failed_at, a.last_refresh_error,
                   a.findings
            FROM rod_pump_health_analyses a
            JOIN wells w ON w.id=a.well_id
            WHERE w.site_id=:site_id AND a.well_id IN ({placeholders})
        """)
        params: dict[str, Any] = {"site_id": site_id}
        params.update({f"well_{index}": well_id for index, well_id in enumerate(well_ids)})
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(query, params).mappings().all()
        except SQLAlchemyError as exc:
            raise RodPumpHealthError(f"Could not read saved rod-pump health: {exc}") from exc
        return {int(row["well_id"]): dict(row) for row in rows}

    def _saved_review(self, row: dict[str, Any]) -> dict[str, Any]:
        """Keep what the health page shows; drop raw trend samples and token usage."""
        findings = _decoded(row["findings"], {})
        if not isinstance(findings, dict):
            findings = {}
        review = {
            "record_status": row["status"],
            "review_status": findings.get("review_status"),
            "review_source": findings.get("review_source"),
            "health_status": findings.get("health_status"),
            "diagnosis_status": findings.get("diagnosis_status"),
            "summary": findings.get("summary"),
            "findings": [
                {
                    key: item.get(key)
                    for key in ("title", "side", "kind", "confidence", "reasoning", "actions")
                    if item.get(key) is not None
                }
                for item in findings.get("findings") or []
                if isinstance(item, dict)
            ],
            "limitations": findings.get("limitations") or [],
            "card_captured_at": self._timestamp(row["latest_card_at"]),
            "analyzed_at": self._timestamp(row["analyzed_at"]),
            "refresh": self._refresh_state(row),
        }
        if findings.get("shutdown"):
            review["shutdown"] = findings["shutdown"]
        if isinstance(findings.get("surface_load"), dict):
            review["surface_load"] = findings["surface_load"]
        if isinstance(findings.get("classifier"), dict):
            review["classifier"] = findings["classifier"]
        trends = findings.get("parameter_trends")
        if isinstance(trends, dict) and trends.get("status") not in (None, "disabled"):
            review["parameter_trends"] = {
                key: trends.get(key)
                for key in (
                    "status", "window_start_at", "window_end_at", "analyzed_at",
                    "trends", "missing_parameters", "interpretation",
                )
                if trends.get(key) is not None
            }
        conclusion = findings.get("conclusion")
        if isinstance(conclusion, dict) and conclusion.get("status") not in (None, "disabled"):
            review["conclusion"] = {
                key: conclusion.get(key)
                for key in ("status", "title", "body", "analyzed_at")
                if conclusion.get(key) is not None
            }
        return review

    def _refresh_state(self, row: dict[str, Any]) -> dict[str, Any]:
        started = self._datetime(row["refresh_started_at"])
        state: dict[str, Any] = {
            "in_progress": bool(started and datetime.now(self.timezone) - started < REFRESH_EXPIRY),
        }
        if row["last_refresh_failed_at"]:
            state["last_failed_at"] = self._timestamp(row["last_refresh_failed_at"])
            state["last_error"] = row["last_refresh_error"]
        return state

    def _datetime(self, value: Any) -> datetime | None:
        """UMetriqs stores these columns in its application time zone without an offset."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return None
        if not isinstance(value, datetime):
            return None
        return value.replace(tzinfo=self.timezone) if value.tzinfo is None else value.astimezone(self.timezone)

    def _timestamp(self, value: Any) -> str | None:
        parsed = self._datetime(value)
        return parsed.isoformat() if parsed else None


class UnavailableRodPumpHealthClient:
    def __init__(self, reason: str): self.reason = reason
    def well_health(self, *args, **kwargs): raise RodPumpHealthError(self.reason)
    def site_health(self, *args, **kwargs): raise RodPumpHealthError(self.reason)
    def resolve_well(self, *args, **kwargs): raise RodPumpHealthError(self.reason)


def _decoded(value: Any, default: Any) -> Any:
    if isinstance(value, (bytes, str)):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return default if value is None else value


def _well_identifiers(row: dict[str, Any]) -> set[str]:
    """Return complete terminal identifiers so 823 cannot resolve 5823."""
    identifiers = set()
    for value in (row["name"], row["key"]):
        tokens = [token for token in re.split(r"[\s_]+", str(value).strip()) if token]
        if tokens:
            identifiers.add(tokens[-1].casefold())
    return identifiers


def _preview(summary: str) -> str:
    if len(summary) <= SUMMARY_PREVIEW_CHARS:
        return summary
    return summary[:SUMMARY_PREVIEW_CHARS].rsplit(" ", 1)[0] + "…"


def _site_order(item: dict[str, Any]) -> tuple[int, str]:
    rank = HEALTH_ORDER.get(item.get("health_status") or "", 3 if item["available"] else 4)
    return rank, item["well"]["name"]
