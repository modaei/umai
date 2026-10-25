from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import date, timedelta
from time import perf_counter
from typing import Any, TypedDict

from langchain_core.tools import BaseTool


class ReportComparisonDependency(TypedDict):
    """Structured report comparison supplied to the general agent."""

    context: str
    traces: list[dict[str, Any]]
    stats: dict[str, Any]
    tool_name: str


REPORT_NAME_BY_PHRASE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("oil_production", ("oil production", "oil")),
    ("water_production", ("water production",)),
    ("gas_production", ("gas production",)),
    ("gas_flared", ("gas flared", "flared gas")),
    ("gas_vented", ("gas vented", "vented gas")),
)


def prepare_report_comparison_dependency(
    *,
    tools: list[BaseTool],
    question: str,
    today: date,
) -> ReportComparisonDependency | None:
    """Prefetch high-confidence report comparisons with stable dates."""
    route = _classify_report_comparison(question, today)
    if route is None:
        return None
    tool = next((item for item in tools if item.name == "compare_report_periods"), None)
    if tool is None:
        return None

    started_at = perf_counter()
    tool_started_at = perf_counter()
    try:
        raw_result = tool.invoke(route["arguments"])
        result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except Exception as exc:
        result = {"ok": False, "error": f"Report comparison failed: {exc}"}
    tool_seconds = perf_counter() - tool_started_at
    context = {
        "authoritative_data_type": "report_comparison",
        "tool": "compare_report_periods",
        "arguments": route["arguments"],
        "result": result,
    }
    return {
        "context": json.dumps(context, separators=(",", ":"), default=str),
        "traces": [
            {
                "tool": "compare_report_periods",
                "arguments": route["arguments"],
                "result": result,
            }
        ],
        "stats": {
            "total_seconds": perf_counter() - started_at,
            "model_seconds": 0.0,
            "tool_seconds": tool_seconds,
            "model_calls": 0,
            "tool_calls": [{"tool": "compare_report_periods", "seconds": tool_seconds}],
        },
        "tool_name": "compare_report_periods",
    }


def _classify_report_comparison(question: str, today: date) -> dict[str, Any] | None:
    normalized = " ".join(question.lower().split())
    if not re.search(r"\bcompar(?:e|ing|ison)\b", normalized):
        return None
    include_batteries = bool(re.search(r"\b(each|per|by)\s+batter(?:y|ies)\b", normalized))
    report_name = "battery" if include_batteries else _report_name(normalized)
    if report_name is None:
        return None

    ranges = _comparison_ranges(normalized, today)
    if ranges is None:
        return None
    first_start, first_end, second_start, second_end = ranges
    return {
        "include_batteries": include_batteries,
        "arguments": {
            "report_name": report_name,
            "first_start_date": first_start.isoformat(),
            "first_end_date": first_end.isoformat(),
            "second_start_date": second_start.isoformat(),
            "second_end_date": second_end.isoformat(),
        },
    }


def _report_name(text: str) -> str | None:
    for report_name, phrases in REPORT_NAME_BY_PHRASE:
        if any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases):
            return report_name
    return None


def _comparison_ranges(text: str, today: date) -> tuple[date, date, date, date] | None:
    if "yesterday" in text and re.search(r"\b(day before|previous day)\b", text):
        yesterday = today - timedelta(days=1)
        day_before = today - timedelta(days=2)
        return yesterday, yesterday, day_before, day_before

    if "last month" in text and re.search(r"\b(month before|previous month)\b", text):
        first = _month_range(today.year, today.month, -1)
        second = _month_range(today.year, today.month, -2)
        return (*first, *second)

    month_pair = re.search(
        r"\b(?:in\s+)?"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(?:and|vs|versus|with)\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        text,
    )
    if month_pair:
        first = _named_month_range(month_pair.group(1), today.year)
        second = _named_month_range(month_pair.group(2), today.year)
        return (*first, *second)

    return None


def _month_range(year: int, month: int, offset: int) -> tuple[date, date]:
    zero_based = (year * 12 + month - 1) + offset
    resolved_year = zero_based // 12
    resolved_month = zero_based % 12 + 1
    return (
        date(resolved_year, resolved_month, 1),
        date(resolved_year, resolved_month, monthrange(resolved_year, resolved_month)[1]),
    )


def _named_month_range(month_name: str, year: int) -> tuple[date, date]:
    month = (
        "january february march april may june july august september october november december"
    ).split().index(month_name) + 1
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])

