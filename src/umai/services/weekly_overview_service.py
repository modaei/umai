from __future__ import annotations

import json
from calendar import monthrange
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from umai.agents.chat_agent import build_model
from umai.clients.report_client import ReportClient
from umai.clients.shutdown_client import ShutdownClient
from umai.config.settings import Settings
from umai.rag.vector_store import VectorOperationalContextStore


REPORT_METRICS = {
    "oil_production": "Oil production",
    "oil_sale": "Oil sale",
}


@dataclass(frozen=True)
class WeeklyOverviewResult:
    overview: str


class WeeklyOverviewService:
    """Build the UMAI Overview section for scheduled weekly emails."""

    def __init__(
        self,
        report_client: ReportClient,
        shutdown_client: ShutdownClient,
        operational_context_store: VectorOperationalContextStore,
        model: Any,
    ):
        self.report_client = report_client
        self.shutdown_client = shutdown_client
        self.operational_context_store = operational_context_store
        self.model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "WeeklyOverviewService":
        settings.validate()
        return cls(
            report_client=ReportClient(
                api_url=settings.umreports_api_url,
                timeout_seconds=settings.umreports_timeout_seconds,
                max_report_days=settings.max_report_days,
            ),
            shutdown_client=ShutdownClient.from_settings(settings),
            operational_context_store=VectorOperationalContextStore.from_settings(settings),
            model=build_model(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                reasoning_effort="medium",
            ),
        )

    def generate(
        self,
        site_id: int,
        start_date: date,
        end_date: date,
    ) -> WeeklyOverviewResult:
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")

        dataset = self._build_dataset(site_id, start_date, end_date)
        overview = _message_text(
            self.model.invoke(
                [
                    SystemMessage(content=_system_prompt()),
                    HumanMessage(
                        content=(
                            "Write the weekly UMAI Overview section from this JSON. "
                            "Return only the section body, not the heading.\n\n"
                            + json.dumps(dataset, default=str, separators=(",", ":"))
                        )
                    ),
                ]
            ).content
        ).strip()
        if not overview:
            raise ValueError("UMAI overview was empty.")
        return WeeklyOverviewResult(overview=overview)

    def _build_dataset(
        self,
        site_id: int,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        previous_start, previous_end = _previous_period(start_date, end_date)
        weekly = self._compare_reports(
            site_id,
            start_date,
            end_date,
            previous_start,
            previous_end,
        )
        month_comparison = None
        if _crosses_month_boundary(start_date, end_date):
            current_month_start, current_month_end = _previous_month_range(end_date)
            prior_month_start, prior_month_end = _previous_month_range(current_month_start)
            month_comparison = self._compare_reports(
                site_id,
                current_month_start,
                current_month_end,
                prior_month_start,
                prior_month_end,
            )

        shutdowns = self.shutdown_client.get_shutdowns(
            site_id,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        shutdown_causes = self.shutdown_client.summarize_shutdown_causes(
            site_id,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        operational_context = self.operational_context_store.search(
            query=(
                "weekly operational overview production injection flared shutdown "
                "general notes chart notes reading comments work orders alarms "
                "unusual issues reasons"
            ),
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            limit=20,
        )
        compact_shutdowns = _compact_shutdowns(shutdowns, start_date, end_date)
        compact_shutdown_causes = _compact_shutdown_causes(shutdown_causes)
        compact_operational_context = _compact_context(operational_context)
        return {
            "site_id": site_id,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "previous_week_period": {
                "start_date": previous_start.isoformat(),
                "end_date": previous_end.isoformat(),
            },
            "weekly_report_comparisons": weekly,
            "month_comparison": month_comparison,
            "executive_snapshot": _executive_snapshot(
                weekly,
                compact_shutdowns,
                compact_shutdown_causes,
            ),
            "production_sales_gap": _production_sales_gap(weekly),
            "top_operational_drivers": _top_operational_drivers(
                compact_shutdowns,
                compact_shutdown_causes,
            ),
            "exceptions": _exceptions(
                weekly,
                compact_shutdowns,
                compact_operational_context,
            ),
            "shutdowns": compact_shutdowns,
            "shutdown_causes": compact_shutdown_causes,
            "operational_context": compact_operational_context,
            "instructions": {
                "change_analysis_rule": (
                    "Include a reason for a weekly or monthly change only when "
                    "shutdowns or operational_context contain supporting evidence. "
                    "If no reason is supported, report the trend without saying why."
                ),
                "format": (
                    "Write for high-level management. Use one executive summary "
                    "paragraph that includes the oil production trend, oil sale "
                    "trend, material production-sales gap, and shutdown summary. "
                    "The first comparison sentence must start with the current "
                    "period and use separate clauses for oil production and oil "
                    "sale so opposite directions are unambiguous. Use this shape: "
                    "During [current period], oil production was [production] "
                    "barrels, down/up [production delta] barrels ([production "
                    "percent]) compared to the previous week ([previous week "
                    "range]), while oil sale was [sale] barrels, down/up [sale "
                    "delta] barrels ([sale percent]). Month-end comparisons must "
                    "explicitly say they are compared with the previous month and "
                    "include the previous month date range. "
                    "Then include short Operational Focus Areas bullets focused "
                    "on operational information from notes, comments, work orders, "
                    "alarms, and unusual field context. Do not include a Key "
                    "metrics section. Add a month-end note only when "
                    "month_comparison is present. Do not include a heading."
                ),
            },
        }

    def _compare_report(
        self,
        site_id: int,
        report_name: str,
        current_start: date,
        current_end: date,
        previous_start: date,
        previous_end: date,
    ) -> dict[str, Any]:
        current_result = self.report_client.run_report(
            site_id,
            report_name,
            current_start.isoformat(),
            current_end.isoformat(),
        )
        previous_result = self.report_client.run_report(
            site_id,
            report_name,
            previous_start.isoformat(),
            previous_end.isoformat(),
        )
        current_total = _report_total(current_result)
        previous_total = _report_total(previous_result)
        current_days = _inclusive_days(current_start, current_end)
        previous_days = _inclusive_days(previous_start, previous_end)
        return {
            "report_name": report_name,
            "label": REPORT_METRICS[report_name],
            "current_period": {
                "start_date": current_start.isoformat(),
                "end_date": current_end.isoformat(),
                "total": current_total,
                "daily_average": round(current_total / current_days, 2),
            },
            "comparison_period": {
                "start_date": previous_start.isoformat(),
                "end_date": previous_end.isoformat(),
                "total": previous_total,
                "daily_average": round(previous_total / previous_days, 2),
            },
            "change": _change_summary(current_total, previous_total),
        }

    def _compare_reports(
        self,
        site_id: int,
        current_start: date,
        current_end: date,
        previous_start: date,
        previous_end: date,
    ) -> dict[str, dict[str, Any]]:
        """Compare all weekly overview report metrics concurrently."""

        with ThreadPoolExecutor(max_workers=len(REPORT_METRICS)) as executor:
            results = list(
                executor.map(
                    lambda report_name: (
                        report_name,
                        self._compare_report(
                            site_id,
                            report_name,
                            current_start,
                            current_end,
                            previous_start,
                            previous_end,
                        ),
                    ),
                    REPORT_METRICS,
                )
            )
        return dict(results)


def _system_prompt() -> str:
    return (
        "You write a concise executive weekly overview for an oil-field operations email. "
        "Use only the JSON supplied by the user. For weekly and monthly report "
        "comparisons, compare only oil production and oil sale, including "
        "percentage changes when available. Start with one executive summary "
        "paragraph that also includes the shutdown summary. Mention the "
        "production-sales gap when material. For the weekly comparison, the first "
        "sentence must start with the current period and use separate clauses for "
        "oil production and oil sale so opposite directions are unambiguous. Use "
        "this shape: During [current period], oil production was [production] "
        "barrels, down/up [production delta] barrels ([production percent]) "
        "compared to the previous week ([previous week range]), while oil sale "
        "was [sale] barrels, down/up [sale delta] barrels ([sale percent]). For "
        "month-end comparisons, explicitly say they are compared with the previous "
        "month and include the previous month date range. Do not use 'respectively' "
        "for oil production and oil sale comparisons. Do not use ambiguous "
        "comparison wording unless the baseline period is named. "
        "Do not include a 'Key metrics:' section. Then include a short "
        "'Operational Focus Areas:' bullet section "
        "focused on summarized operational information from general notes, chart "
        "notes, reading comments, shutdown comments, work orders, alarms, repeated "
        "themes, and unusual field context. "
        "For weekly or monthly comparisons, include change analysis only when a "
        "reason is directly supported by shutdowns or operational context. If no "
        "reason is supported, state the trend without explaining why. Add a "
        "month-end note only when month_comparison is present. Do not invent "
        "causes. Use MM/DD/YYYY dates when dates are needed. Use barrels for "
        "liquid volumes and $ for costs. Do not mention internal site IDs."
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _previous_period(start_date: date, end_date: date) -> tuple[date, date]:
    day_count = _inclusive_days(start_date, end_date)
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=day_count - 1)
    return previous_start, previous_end


def _inclusive_days(start_date: date, end_date: date) -> int:
    return max((end_date - start_date).days + 1, 1)


def _crosses_month_boundary(start_date: date, end_date: date) -> bool:
    return (start_date.year, start_date.month) != (end_date.year, end_date.month)


def _previous_month_range(reference_date: date) -> tuple[date, date]:
    year = reference_date.year
    month = reference_date.month - 1
    if month == 0:
        year -= 1
        month = 12
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _report_total(result: Any) -> float:
    if isinstance(result, dict):
        for key in ("date_totals", "battery_totals"):
            value = result.get(key)
            if isinstance(value, dict):
                return _sum_numeric(value.values())
            if isinstance(value, list):
                return _sum_record_values(value)
        return _sum_record_values([result])
    if isinstance(result, list):
        return _sum_record_values(result)
    return 0.0


def _sum_record_values(rows: list[Any]) -> float:
    keys = (
        "value",
        "production",
        "gas_production",
        "flared",
        "flared_vented",
        "total",
    )
    total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            if key in row:
                total += _to_float(row.get(key))
                break
    return round(total, 2)


def _sum_numeric(values: Any) -> float:
    return round(sum(_to_float(value) for value in values), 2)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _change_summary(current: float, previous: float) -> dict[str, Any]:
    delta = round(current - previous, 2)
    if previous == 0:
        percent = None
    else:
        percent = round((delta / previous) * 100, 2)
    if abs(delta) < 0.01:
        direction = "flat"
    elif delta > 0:
        direction = "increase"
    else:
        direction = "decrease"
    return {"direction": direction, "delta": delta, "percent": percent}


def _compact_shutdowns(
    result: dict[str, Any],
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    shutdowns = result.get("shutdowns") or []
    new_shutdowns = [
        row for row in shutdowns
        if _date_in_range(_row_date(row.get("start")), start_date, end_date)
    ]
    ended_shutdowns = [
        row for row in shutdowns
        if _date_in_range(_row_date(row.get("end")), start_date, end_date)
    ]
    return {
        "count": result.get("count", 0),
        "total_hours": result.get("total_hours", 0),
        "new_shutdowns": new_shutdowns[:20],
        "ended_shutdowns": ended_shutdowns[:20],
        "common_shutdown_comments": _common_comments(shutdowns),
        "shutdowns": shutdowns[:30],
    }


def _row_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _date_in_range(value: date | None, start_date: date, end_date: date) -> bool:
    return value is not None and start_date <= value <= end_date


def _common_comments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comments = [
        str(row.get("comments")).strip()
        for row in rows
        if str(row.get("comments") or "").strip()
    ]
    return [
        {"comment": comment, "count": count}
        for comment, count in Counter(comments).most_common(5)
    ]


def _compact_shutdown_causes(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_cause": result.get("main_cause"),
        "causes": (result.get("causes") or [])[:10],
    }


def _compact_context(result: dict[str, Any]) -> list[dict[str, Any]]:
    matches = result.get("matches") if isinstance(result, dict) else []
    compact = []
    for match in matches or []:
        compact.append(
            {
                "source_type": match.get("source_type"),
                "event_date": match.get("event_date"),
                "entity_name": match.get("entity_name"),
                "text": match.get("text"),
            }
        )
    return compact[:20]


def _executive_snapshot(
    weekly: dict[str, dict[str, Any]],
    shutdowns: dict[str, Any],
    shutdown_causes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "oil_production": _metric_snapshot(weekly.get("oil_production")),
        "oil_sale": _metric_snapshot(weekly.get("oil_sale")),
        "shutdown_summary": {
            "shutdown_count": shutdowns.get("count", 0),
            "shutdown_hours": shutdowns.get("total_hours", 0),
            "new_shutdown_count": len(shutdowns.get("new_shutdowns") or []),
            "ended_shutdown_count": len(shutdowns.get("ended_shutdowns") or []),
            "main_cause": shutdown_causes.get("main_cause"),
        },
    }


def _metric_snapshot(metric: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    current = metric.get("current_period") or {}
    previous = metric.get("comparison_period") or {}
    return {
        "label": metric.get("label"),
        "current_total": current.get("total"),
        "current_daily_average": current.get("daily_average"),
        "previous_total": previous.get("total"),
        "previous_daily_average": previous.get("daily_average"),
        "change": metric.get("change"),
    }


def _production_sales_gap(weekly: dict[str, dict[str, Any]]) -> dict[str, Any]:
    production = _current_total(weekly.get("oil_production"))
    sale = _current_total(weekly.get("oil_sale"))
    gap = round(production - sale, 2)
    if abs(gap) < 0.01:
        direction = "balanced"
    elif gap > 0:
        direction = "production_above_sales"
    else:
        direction = "sales_above_production"
    return {
        "oil_production_total": production,
        "oil_sale_total": sale,
        "gap": gap,
        "direction": direction,
    }


def _current_total(metric: dict[str, Any] | None) -> float:
    if not metric:
        return 0.0
    current = metric.get("current_period") or {}
    return _to_float(current.get("total"))


def _top_operational_drivers(
    shutdowns: dict[str, Any],
    shutdown_causes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "shutdown_causes": (shutdown_causes.get("causes") or [])[:5],
        "shutdown_comments": (shutdowns.get("common_shutdown_comments") or [])[:5],
    }


def _exceptions(
    weekly: dict[str, dict[str, Any]],
    shutdowns: dict[str, Any],
    operational_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exceptions: list[dict[str, Any]] = []
    for report_name, metric in weekly.items():
        change = metric.get("change") or {}
        percent = change.get("percent")
        if percent is not None and abs(_to_float(percent)) >= 10:
            exceptions.append(
                {
                    "type": "large_report_change",
                    "label": metric.get("label") or report_name,
                    "direction": change.get("direction"),
                    "percent": percent,
                }
            )

    gap = _production_sales_gap(weekly)
    production_total = max(abs(_to_float(gap.get("oil_production_total"))), 1.0)
    gap_percent = round((abs(_to_float(gap.get("gap"))) / production_total) * 100, 2)
    if gap_percent >= 10:
        exceptions.append(
            {
                "type": "production_sales_gap",
                "direction": gap.get("direction"),
                "gap": gap.get("gap"),
                "percent_of_production": gap_percent,
            }
        )

    if _to_float(shutdowns.get("total_hours")) >= 24:
        exceptions.append(
            {
                "type": "high_shutdown_hours",
                "hours": shutdowns.get("total_hours"),
            }
        )
    if shutdowns.get("new_shutdowns"):
        exceptions.append(
            {
                "type": "new_shutdowns",
                "count": len(shutdowns.get("new_shutdowns") or []),
            }
        )
    if shutdowns.get("ended_shutdowns"):
        exceptions.append(
            {
                "type": "ended_shutdowns",
                "count": len(shutdowns.get("ended_shutdowns") or []),
            }
        )
    for theme in _repeated_context_themes(operational_context):
        exceptions.append({"type": "repeated_operational_context", **theme})
    return exceptions[:10]


def _repeated_context_themes(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keywords = (
        "esp",
        "paraffin",
        "pump",
        "chemical",
        "hot oil",
        "hot water",
        "alarm",
        "leak",
        "electrical",
    )
    counts: Counter[str] = Counter()
    for row in context:
        text = str(row.get("text") or "").lower()
        for keyword in keywords:
            if keyword in text:
                counts[keyword] += 1
    return [
        {"theme": theme, "count": count}
        for theme, count in counts.most_common(5)
        if count > 1
    ]


def generate_weekly_overview(
    settings: Settings,
    site_id: int,
    start_date: date,
    end_date: date,
) -> dict[str, str]:
    """Generate the weekly UMAI Overview response payload."""

    result = WeeklyOverviewService.from_settings(settings).generate(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
    )
    return {"overview": result.overview}
