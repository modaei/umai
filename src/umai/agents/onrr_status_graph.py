from __future__ import annotations

import json
import re
from datetime import date, timedelta
from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph


OnrrStatus = Literal["active", "inactive", "injection", "producing"]
OutputMode = Literal["count", "list"]


class OnrrStatusState(TypedDict, total=False):
    """State for deterministic ONRR status count/list questions."""

    question: str
    status: OnrrStatus
    as_of_date: str
    output_mode: OutputMode
    result: dict[str, Any]
    answer: str
    traces: list[dict[str, Any]]
    tool_timings: list[dict[str, Any]]


def try_answer_onrr_status_question(
    *,
    tools: list[BaseTool],
    question: str,
    history: list[dict[str, str]] | None = None,
    today: date | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
    """Answer ONRR status count/list requests without LLM interpretation.

    This route exists because the ONRR status tool already returns both counts
    and selected well rows. Formatting those rows directly avoids model
    variability on follow-ups like "list them".
    """
    route = _classify_status_request(question, history or [], today or date.today())
    if route is None:
        return None

    tool_name = (
        "get_producing_wells"
        if route["status"] == "producing"
        else "count_wells_by_onrr_status"
    )
    tool = next((item for item in tools if item.name == tool_name), None)
    if tool is None:
        return None

    graph = StateGraph(OnrrStatusState)

    def fetch_status(state: OnrrStatusState) -> dict[str, Any]:
        if state["status"] == "producing":
            arguments = {"producing_date": state["as_of_date"]}
        else:
            arguments = {
                "as_of_date": state["as_of_date"],
                "status": state["status"],
            }
        started_at = perf_counter()
        try:
            raw_result = tool.invoke(arguments)
            result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except Exception as exc:
            result = {"ok": False, "error": f"ONRR status lookup failed: {exc}"}
        seconds = perf_counter() - started_at
        return {
            "result": result,
            "traces": [{"tool": tool_name, "arguments": arguments, "result": result}],
            "tool_timings": [{"tool": tool_name, "seconds": seconds}],
        }

    def format_answer(state: OnrrStatusState) -> dict[str, str]:
        return {"answer": _format_status_answer(state)}

    graph.add_node("fetch_status", fetch_status)
    graph.add_node("format_answer", format_answer)
    graph.add_edge(START, "fetch_status")
    graph.add_edge("fetch_status", "format_answer")
    graph.add_edge("format_answer", END)

    started_at = perf_counter()
    state = graph.compile().invoke({**route, "question": question})
    elapsed = perf_counter() - started_at
    timings = state.get("tool_timings", [])
    return (
        state["answer"],
        state.get("traces", []),
        {
            "total_seconds": elapsed,
            "model_seconds": 0.0,
            "tool_seconds": sum(item["seconds"] for item in timings),
            "model_calls": 0,
            "tool_calls": timings,
        },
    )


def _classify_status_request(
    question: str,
    history: list[dict[str, str]],
    today: date,
) -> dict[str, Any] | None:
    """Classify direct ONRR status questions and safe list follow-ups."""
    normalized = _normalize_text(question)
    output_mode = _output_mode(normalized)
    direct = _direct_status_route(normalized, today, output_mode)
    if direct is not None:
        return direct

    if not _is_status_list_follow_up(normalized):
        return None
    anchor = _latest_status_anchor(history, today)
    if anchor is None:
        return None
    anchor["output_mode"] = "list"
    return anchor


def _direct_status_route(
    normalized: str,
    today: date,
    output_mode: OutputMode | None,
) -> dict[str, Any] | None:
    if _has_mixed_operational_intent(normalized):
        return None
    status = _status_from_text(normalized)
    if status is None:
        return None
    if output_mode is None:
        return None
    as_of_date = _resolve_single_date(normalized, today)
    if as_of_date is None:
        return None
    return {
        "status": status,
        "as_of_date": as_of_date.isoformat(),
        "output_mode": output_mode,
    }


def _has_mixed_operational_intent(normalized: str) -> bool:
    """Return true when status words are only one filter in a broader request.

    The deterministic ONRR route is intentionally narrow. Questions such as
    "active wells that were shutdown yesterday" require combining the ONRR
    status tool with shutdown tooling, so they must continue to the normal LLM
    tool-selection path instead of being answered as a plain active-well count.
    """
    return bool(
        re.search(
            r"\b("
            r"shutdowns?|shut\s*downs?|shut-?ins?|downtime|down|"
            r"alarms?|events?|tests?|well\s*tests?|readings?|reports?|"
            r"production|produced|allocation|contributed|"
            r"tanks?|flares?|flow\s*meters?|lacts?|treaters?|pumps?|"
            r"notes?|work\s*orders?|timeline|history"
            r")\b",
            normalized,
        )
    )


def _latest_status_anchor(
    history: list[dict[str, str]],
    today: date,
) -> dict[str, Any] | None:
    """Find the last user turn that established a status and date."""
    for item in reversed(history[-10:]):
        if item.get("role") != "user":
            continue
        route = _direct_status_route(_normalize_text(item.get("content", "")), today, "count")
        if route is not None:
            return route
    return None


def _format_status_answer(state: OnrrStatusState) -> str:
    result = state["result"]
    if not result.get("ok", True):
        return result.get("error", "ONRR status lookup failed.")

    status = state["status"]
    display_date = _american_date(state["as_of_date"])
    wells = _status_wells(status, result)
    count = _status_count(status, result, wells)
    label = _status_label(status)

    if state["output_mode"] == "count":
        return f"There were {count} {label} on {display_date}."

    if not wells:
        return f"No {label} were found on {display_date}."
    names = "\n".join(f"- {_well_name(item)}" for item in wells if _well_name(item))
    return f"Here are the {count} {label} on {display_date}:\n\n{names}"


def _status_wells(status: OnrrStatus, result: dict[str, Any]) -> list[dict[str, Any]]:
    if status == "producing":
        return list(result.get("producing_wells") or [])
    return list(result.get("wells") or [])


def _status_count(
    status: OnrrStatus,
    result: dict[str, Any],
    wells: list[dict[str, Any]],
) -> int:
    if status == "producing":
        return int(result.get("producing_count", len(wells)))
    counts = result.get("counts") or {}
    return int(counts.get(status, len(wells)))


def _status_label(status: OnrrStatus) -> str:
    if status == "producing":
        return "producing wells"
    if status == "injection":
        return "injection wells"
    return f"{status} wells"


def _well_name(item: dict[str, Any]) -> str:
    return str(item.get("well") or "").strip()


def _output_mode(normalized: str) -> OutputMode | None:
    if re.search(r"\b(how many|count|number of)\b", normalized):
        return "count"
    if re.search(r"\b(list|show|which|what)\b", normalized):
        return "list"
    return None


def _status_from_text(normalized: str) -> OnrrStatus | None:
    if re.search(r"\binactive wells?\b", normalized):
        return "inactive"
    if re.search(r"\binjection wells?\b", normalized):
        return "injection"
    if re.search(r"\bactive wells?\b", normalized):
        return "active"
    if re.search(r"\bproducing wells?\b", normalized):
        return "producing"
    return None


def _is_status_list_follow_up(normalized: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:please\s+)?(?:list|show)\s+(?:them|those|these|the wells)|"
            r"(?:which|what)\s+(?:ones|wells)\??",
            normalized,
        )
    )


def _resolve_single_date(text: str, today: date) -> date | None:
    if re.search(r"\btoday\b|\bnow\b", text):
        return today
    if re.search(r"\byesterday\b", text):
        return today - timedelta(days=1)
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        return date.fromisoformat(iso_match.group(1))
    month_day_match = re.search(
        r"\b(?:on|as of)?\s*([a-z]+)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b",
        text,
    )
    if month_day_match:
        month = MONTHS.get(month_day_match.group(1))
        if month is None:
            return None
        year = int(month_day_match.group(3) or today.year)
        try:
            return date(year, month, int(month_day_match.group(2)))
        except ValueError:
            return None
    return None


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().rstrip("?.!").split())


def _american_date(value: str) -> str:
    return date.fromisoformat(value).strftime("%m/%d/%Y")


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
