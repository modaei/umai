from __future__ import annotations

import json
import re
from datetime import date
from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from umai.agents.producing_wells_graph import _resolve_date_range


class WellTestAnalysisState(TypedDict, total=False):
    """State for deterministic well-test analysis and dependency prefetching."""

    question: str
    analysis_mode: Literal[
        "latest_previous", "range_summary", "recent_tests", "range_sequence"
    ]
    group_by: Literal["well", "battery", "site"]
    arguments: dict[str, Any]
    result: dict[str, Any]
    answer: str
    traces: list[dict[str, Any]]
    tool_timings: list[dict[str, Any]]


class WellTestAnalysisDependency(TypedDict):
    """Structured well-test analysis supplied to the general agent."""

    context: str
    traces: list[dict[str, Any]]
    stats: dict[str, Any]
    tool_name: str


def try_answer_well_test_analysis(
    *,
    tools: list[BaseTool],
    question: str,
    history: list[dict[str, str]],
    today: date | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
    """Answer complete, explicit well-test analyses without an LLM call."""
    route = _classify_well_test_analysis(question, history, today or date.today())
    if route is None or route["route_type"] != "terminal":
        return None
    tool = _tool_by_name(tools, "analyze_well_tests")
    if tool is None:
        return None

    state, elapsed = _run_analysis_graph(tool, question, route["arguments"])
    result = state["result"]
    answer = _format_analysis_result(result)
    timings = state.get("tool_timings", [])
    return answer, state.get("traces", []), _stats(elapsed, timings)


def prepare_well_test_analysis_dependency(
    *,
    tools: list[BaseTool],
    question: str,
    history: list[dict[str, str]],
    today: date | None = None,
) -> WellTestAnalysisDependency | None:
    """Prefetch well-test analysis needed by a broader multi-tool answer."""
    route = _classify_well_test_analysis(question, history, today or date.today())
    if route is None or route["route_type"] != "dependency":
        return None
    tool = _tool_by_name(tools, "analyze_well_tests")
    if tool is None:
        return None

    state, elapsed = _run_analysis_graph(tool, question, route["arguments"])
    result = state["result"]
    context = {
        "authoritative_data_type": "well_test_analysis",
        "tool": "analyze_well_tests",
        "arguments": route["arguments"],
        "result": result,
    }
    timings = state.get("tool_timings", [])
    return {
        "context": json.dumps(context, separators=(",", ":"), default=str),
        "traces": state.get("traces", []),
        "stats": _stats(elapsed, timings),
        "tool_name": "analyze_well_tests",
    }


def _run_analysis_graph(
    tool: BaseTool,
    question: str,
    arguments: dict[str, Any],
) -> tuple[WellTestAnalysisState, float]:
    """Execute the fixed analysis node and preserve tool diagnostics."""
    graph = StateGraph(WellTestAnalysisState)

    def analyze(state: WellTestAnalysisState) -> dict[str, Any]:
        started_at = perf_counter()
        try:
            raw_result = tool.invoke(arguments)
            result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except Exception as exc:
            result = {"ok": False, "error": f"Well-test analysis failed: {exc}"}
        seconds = perf_counter() - started_at
        return {
            "result": result,
            "traces": [
                {"tool": "analyze_well_tests", "arguments": arguments, "result": result}
            ],
            "tool_timings": [{"tool": "analyze_well_tests", "seconds": seconds}],
        }

    graph.add_node("analyze_well_tests", analyze)
    graph.add_edge(START, "analyze_well_tests")
    graph.add_edge("analyze_well_tests", END)
    started_at = perf_counter()
    state = graph.compile().invoke(
        {"question": question, "arguments": arguments, "traces": [], "tool_timings": []}
    )
    return state, perf_counter() - started_at


def _classify_well_test_analysis(
    question: str,
    history: list[dict[str, str]],
    today: date,
) -> dict[str, Any] | None:
    """Resolve explicit analysis intent, including short contextual follow-ups."""
    current = " ".join(question.lower().split())
    prior = _latest_well_test_user_question(history)
    has_current_subject = bool(re.search(r"\bwell[ -]?tests?\b", current))
    is_follow_up = bool(
        prior
        and re.search(
            r"\b(them|those|each|previous|previus|prior|before|break(?:\s+it)?\s+down|by battery|whole site)\b",
            current,
        )
    )
    if not has_current_subject and not is_follow_up:
        return None

    contextual = f"{prior} {current}".strip() if is_follow_up else current
    number_words = {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    recent_match = re.search(
        r"\b(?:last|latest|most recent)\s+"
        r"(?P<count>\d+|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:well[ -]?)?tests?\b",
        contextual,
    )
    recent_test_count = None
    if recent_match:
        raw_count = recent_match.group("count")
        recent_test_count = number_words.get(
            raw_count,
            int(raw_count) if raw_count.isdigit() else 0,
        )
        if not 2 <= recent_test_count <= 10:
            return None

    resolved_dates = _resolve_date_range(contextual, today)
    explicitly_previous = bool(
        re.search(r"\b(previous|previus|prior|before)\b", contextual)
    )
    range_sequence = bool(
        recent_test_count is None
        and resolved_dates
        and not explicitly_previous
        and re.search(r"\bcompar(?:e|ing|ison)\b", contextual)
    )
    latest_previous = bool(
        recent_test_count is None
        and not range_sequence
        and (
        re.search(
            r"\b(latest|last|most recent)\b.*\b(previous|previus|prior|before|last test)\b",
            contextual,
        )
        or re.search(r"\bcompare\b.*\b(previous|previus|prior|before)\b", contextual)
        or re.search(r"\btest one before\b", contextual)
        )
    )
    range_summary = bool(
        re.search(
            r"\b(analy[sz]e|summary|summarize|average|avg|sum|total|minimum|maximum|min|max|breakdown|break down)\b",
            contextual,
        )
    )
    if (
        not recent_test_count
        and not range_sequence
        and not latest_previous
        and not range_summary
    ):
        return None

    if recent_test_count:
        analysis_mode = "recent_tests"
    elif range_sequence:
        analysis_mode = "range_sequence"
    elif latest_previous:
        analysis_mode = "latest_previous"
    else:
        analysis_mode = "range_summary"
    if re.search(r"\b(?:by|per)\s+batter(?:y|ies)\b|\bbattery breakdown\b", contextual):
        group_by = "battery"
    elif re.search(r"\b(?:by|per|each)\s+well\b|\bfor each well\b", contextual):
        group_by = "well"
    elif re.search(r"\b(?:whole|entire|selected) site\b|\b(?:by|per) site\b", contextual):
        group_by = "site"
    else:
        group_by = (
            "well"
            if analysis_mode in {"latest_previous", "recent_tests", "range_sequence"}
            else "site"
        )
    if analysis_mode in {"recent_tests", "range_sequence"}:
        group_by = "well"

    arguments: dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "group_by": group_by,
    }
    if resolved_dates:
        arguments["start_date"] = resolved_dates[0].isoformat()
        arguments["end_date"] = resolved_dates[1].isoformat()
    elif analysis_mode in {"range_summary", "range_sequence"}:
        return None
    if recent_test_count:
        arguments["test_count"] = recent_test_count

    battery_match = re.search(r"\bbattery\s+([a-z0-9_-]+)\b", current, re.IGNORECASE)
    if battery_match and battery_match.group(1).lower() not in {"breakdown", "summary"}:
        battery_reference = battery_match.group(1)
        arguments["battery_name"] = (
            f"Battery {battery_reference}"
            if battery_reference.isdigit()
            else battery_reference
        )
    well_name = _extract_well_name_filter(current)
    if well_name:
        arguments["well_name"] = well_name

    composite_terms = re.search(
        r"\b(shutdowns?|alarms?|notes?|work orders?|production|allocation|"
        r"injection|flares?|tanks?|why|causes?|explain|correlat\w*|impact)\b",
        current,
    )
    return {
        "route_type": "dependency" if composite_terms else "terminal",
        "arguments": arguments,
    }


def _latest_well_test_user_question(history: list[dict[str, str]]) -> str:
    """Find the latest user turn that established a well-test subject or period."""
    for item in reversed(history[-10:]):
        if item.get("role") != "user":
            continue
        content = " ".join(str(item.get("content", "")).lower().split())
        if re.search(r"\bwell[ -]?tests?\b", content):
            return content
    return ""


def _extract_well_name_filter(current: str) -> str | None:
    """Extract a specific well reference from common Hartzog shorthand."""
    patterns = (
        r"\bhartzog\s+draw\s+unit\s+([a-z0-9_-]*\d[a-z0-9_-]*)\b",
        r"\bwell\s+(?:hdu[\s_-]*)?([a-z0-9_-]*\d[a-z0-9_-]*)\b",
        r"\bhdu[\s_-]*([a-z0-9_-]*\d[a-z0-9_-]*)\b",
        r"\bfor\s+([a-z]?\d{3,6}[a-z]?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, current, re.IGNORECASE)
        if match:
            return _normalize_well_name_filter(match.group(1))
    return None


def _normalize_well_name_filter(value: str) -> str:
    """Return the DB-friendly fragment used by ReadingClient LIKE matching."""
    normalized = value.strip(" ,.?").replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    hdu_match = re.fullmatch(r"hdu\s+(.+)", normalized, re.IGNORECASE)
    if hdu_match:
        normalized = hdu_match.group(1)
    return normalized


def _format_analysis_result(result: dict[str, Any]) -> str:
    """Format complete analyses consistently without a final model call."""
    if not result.get("ok", True):
        return str(result.get("error") or "Well-test analysis could not be completed.")
    groups = result.get("groups", [])
    if not groups:
        return "No matching well tests were found."
    mode = result.get("analysis_mode")
    group_by = result.get("group_by")
    if mode == "latest_previous":
        lines = [
            f"Compared the latest and previous well tests for {len(groups)} {group_by} group(s)."
        ]
        lines.extend(_format_latest_previous_group(group, group_by) for group in groups)
        return "\n".join(lines)
    if mode in {"recent_tests", "range_sequence"}:
        return _format_sequence_result(result, groups)

    start = _display_date(result.get("start_date"))
    end = _display_date(result.get("end_date"))
    lines = [f"Well-test summary from {start} through {end} by {group_by}:"]
    lines.extend(_format_range_group(group) for group in groups)
    return "\n".join(lines)


def _format_sequence_result(
    result: dict[str, Any], groups: list[dict[str, Any]]
) -> str:
    """Render per-well chronological tests and their adjacent changes."""
    if result.get("analysis_mode") == "recent_tests":
        requested = result.get("requested_test_count")
        lines = [
            f"Compared the latest {requested} well tests for {len(groups)} well(s)."
        ]
    else:
        lines = [
            "Compared well tests from "
            f"{_display_date(result.get('start_date'))} through "
            f"{_display_date(result.get('end_date'))} for {len(groups)} well(s)."
        ]
    if result.get("truncated"):
        lines.append(
            "The result reached the row limit, so later wells or tests may be omitted."
        )
    for group in groups:
        lines.append(f"- {group.get('well_name', group.get('group_name', 'Unknown'))}")
        tests = group.get("tests", [])
        if len(tests) == 1:
            lines.append(
                f"  - {_format_sequence_test(tests[0], baseline=True)}; "
                "no comparison is available because only one test was selected"
            )
            continue
        for index, test in enumerate(tests):
            lines.append(
                f"  - {_format_sequence_test(test, baseline=index == 0)}"
            )
    return "\n".join(lines)


def _format_sequence_test(test: dict[str, Any], *, baseline: bool) -> str:
    """Format one selected test with changes from its immediate predecessor."""
    metrics = []
    for metric, unit in (
        ("oil", " bbl"),
        ("water", " bbl"),
        ("gas", ""),
        ("runtime", " hours"),
    ):
        value = f"{_number(test.get(metric))}{unit}"
        if not baseline and test.get(f"{metric}_change") is not None:
            value += f" ({_signed(test.get(f'{metric}_change'))})"
        metrics.append(f"{metric} {value}")
    suffix = "; baseline (no preceding selected test)" if baseline else ""
    return f"{_display_date(test.get('date'))}: {'; '.join(metrics)}{suffix}"


def _format_latest_previous_group(group: dict[str, Any], group_by: str) -> str:
    name = str(group.get("group_name", "Unknown"))
    if group_by == "well":
        dates = f"{_display_date(group.get('previous_date'))} -> {_display_date(group.get('latest_date'))}"
        return (
            f"- {name} ({dates}): "
            f"oil {_transition(group, 'oil')} bbl; "
            f"water {_transition(group, 'water')} bbl; "
            f"gas {_transition(group, 'gas')}; "
            f"runtime {_transition(group, 'runtime')} hours"
        )
    return (
        f"- {name} ({group.get('well_count', 0)} wells): "
        f"oil {_aggregate_transition(group, 'oil')} bbl; "
        f"water {_aggregate_transition(group, 'water')} bbl; "
        f"gas {_aggregate_transition(group, 'gas')}"
    )


def _format_range_group(group: dict[str, Any]) -> str:
    return (
        f"- {group.get('group_name', 'Unknown')}: {group.get('test_count', 0)} tests, "
        f"{group.get('well_count', 0)} wells; "
        f"oil total {_number(group.get('oil_sum'))} bbl, average {_number(group.get('oil_average'))} bbl; "
        f"water total {_number(group.get('water_sum'))} bbl, average {_number(group.get('water_average'))} bbl; "
        f"gas total {_number(group.get('gas_sum'))}, average {_number(group.get('gas_average'))}"
    )


def _transition(group: dict[str, Any], metric: str) -> str:
    return (
        f"{_number(group.get(f'previous_{metric}'))} -> "
        f"{_number(group.get(f'latest_{metric}'))} "
        f"({_signed(group.get(f'{metric}_change'))})"
    )


def _aggregate_transition(group: dict[str, Any], metric: str) -> str:
    return (
        f"{_number(group.get(f'previous_{metric}_sum'))} -> "
        f"{_number(group.get(f'latest_{metric}_sum'))} "
        f"({_signed(group.get(f'{metric}_sum_change'))})"
    )


def _number(value: Any) -> str:
    if value is None:
        return "not recorded"
    number = float(value)
    return f"{number:,.3f}".rstrip("0").rstrip(".")


def _signed(value: Any) -> str:
    if value is None:
        return "change not available"
    number = float(value)
    return f"{number:+,.3f}".rstrip("0").rstrip(".")


def _display_date(value: Any) -> str:
    if not value:
        return "date unavailable"
    return date.fromisoformat(str(value)[:10]).strftime("%m/%d/%Y")


def _tool_by_name(tools: list[BaseTool], name: str) -> BaseTool | None:
    return next((tool for tool in tools if tool.name == name), None)


def _stats(elapsed: float, timings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_seconds": elapsed,
        "model_seconds": 0.0,
        "tool_seconds": sum(item["seconds"] for item in timings),
        "model_calls": 0,
        "tool_calls": timings,
    }
