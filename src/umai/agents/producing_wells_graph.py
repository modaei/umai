from __future__ import annotations

import json
import re
from datetime import date, timedelta
from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph


class ProducingWellGraphState(TypedDict, total=False):
    """State shared by deterministic producing-well workflow nodes."""

    question: str
    intent: Literal["count", "well_test_coverage"]
    population_type: Literal["active", "producing"]
    active_date: str
    start_date: str
    end_date: str
    is_single_date: bool
    output_mode: Literal["all", "list", "count"]
    filters: list[dict[str, Any]]
    population_result: dict[str, Any]
    producing_result: dict[str, Any]
    well_test_result: dict[str, Any]
    answer: str
    traces: list[dict[str, Any]]
    tool_timings: list[dict[str, Any]]


class WellPopulationDependency(TypedDict):
    """Authoritative population prepared for a broader agent question."""

    context: str
    traces: list[dict[str, Any]]
    stats: dict[str, Any]
    tool_name: str


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


def try_answer_producing_well_question(
    *,
    tools: list[BaseTool],
    question: str,
    history: list[dict[str, str]] | None = None,
    today: date | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
    """Run clear producing-well questions without an LLM routing call.

    This route protects the domain definition of a producing well. It invokes the
    authoritative tool directly and performs well-test coverage set operations in
    Python. Questions that are not recognized confidently fall through to the
    general chat agent.
    """
    route = _classify_request(question, today or date.today(), history or [])
    if route is None:
        return None

    tool_map = {tool.name: tool for tool in tools}
    required = {
        "get_active_wells"
        if route.get("population_type") == "active"
        else "get_producing_wells"
    }
    if route["intent"] == "well_test_coverage":
        required.add("search_well_tests")
    if not required.issubset(tool_map):
        return None

    started_at = perf_counter()
    graph = _build_graph(tool_map)
    state = graph.invoke({**route, "question": question, "traces": [], "tool_timings": []})
    elapsed = perf_counter() - started_at
    timings = state.get("tool_timings", [])
    stats = {
        "total_seconds": elapsed,
        "model_seconds": 0.0,
        "tool_seconds": sum(item["seconds"] for item in timings),
        "model_calls": 0,
        "tool_calls": timings,
    }
    return state["answer"], state.get("traces", []), stats


def prepare_well_population_dependency(
    *,
    tools: list[BaseTool],
    question: str,
    today: date | None = None,
) -> WellPopulationDependency | None:
    """Prefetch active/producing wells when they are one input to a broader task.

    Clear count and well-test coverage requests are handled completely by
    `try_answer_producing_well_question`. This function handles other explicit
    references such as "alarms for active wells yesterday" or "compare production
    for producing wells last month" and supplies the authoritative population to
    the general agent.
    """
    route = _classify_population_dependency(question, today or date.today())
    if route is None:
        return None
    tool_map = {tool.name: tool for tool in tools}
    tool = tool_map.get(route["tool_name"])
    if tool is None:
        return None

    graph = StateGraph(ProducingWellGraphState)

    def fetch_population(state: ProducingWellGraphState) -> dict[str, Any]:
        result, trace, timing = _invoke_json_tool(tool, route["arguments"])
        return {
            "producing_result": result,
            "traces": [trace],
            "tool_timings": [timing],
        }

    graph.add_node("fetch_population", fetch_population)
    graph.add_edge(START, "fetch_population")
    graph.add_edge("fetch_population", END)
    started_at = perf_counter()
    state = graph.compile().invoke({"question": question})
    elapsed = perf_counter() - started_at
    result = state["producing_result"]
    if not result.get("ok", True):
        return None

    compact = _compact_population_result(route["tool_name"], result)
    timings = state.get("tool_timings", [])
    return {
        "context": json.dumps(compact, separators=(",", ":")),
        "traces": state.get("traces", []),
        "stats": {
            "total_seconds": elapsed,
            "model_seconds": 0.0,
            "tool_seconds": sum(item["seconds"] for item in timings),
            "model_calls": 0,
            "tool_calls": timings,
        },
        "tool_name": route["tool_name"],
    }


def _build_graph(tool_map: dict[str, BaseTool]):
    """Compile the small workflow whose tool choices are fixed by intent."""
    graph = StateGraph(ProducingWellGraphState)

    def fetch_well_population(state: ProducingWellGraphState) -> dict[str, Any]:
        if state.get("population_type") == "active":
            tool_name = "get_active_wells"
            arguments = {"active_date": state["active_date"]}
        else:
            tool_name = "get_producing_wells"
            arguments = (
                {"producing_date": state["start_date"]}
                if state["is_single_date"]
                else {
                    "start_date": state["start_date"],
                    "end_date": state["end_date"],
                    "range_mode": "any_day",
                }
            )
            if state.get("filters"):
                arguments["filters"] = state["filters"]
        result, trace, timing = _invoke_json_tool(tool_map[tool_name], arguments)
        return {
            "population_result": result,
            "traces": [*state.get("traces", []), trace],
            "tool_timings": [*state.get("tool_timings", []), timing],
        }

    def fetch_well_tests(state: ProducingWellGraphState) -> dict[str, Any]:
        arguments = {
            "start_date": state["start_date"],
            "end_date": state["end_date"],
        }
        result, trace, timing = _invoke_json_tool(tool_map["search_well_tests"], arguments)
        return {
            "well_test_result": result,
            "traces": [*state.get("traces", []), trace],
            "tool_timings": [*state.get("tool_timings", []), timing],
        }

    def format_answer(state: ProducingWellGraphState) -> dict[str, str]:
        population = state["population_result"]
        population_type = state.get("population_type", "producing")
        if not population.get("ok", True):
            fallback = (
                "Active-well lookup failed."
                if population_type == "active"
                else "Producing-well lookup failed."
            )
            return {"answer": population.get("error", fallback)}
        start = _american_date(state["start_date"])
        end = _american_date(state["end_date"])
        period = start if start == end else f"{start} through {end}"

        if state["intent"] == "count":
            count_key = (
                "active_count" if population_type == "active" else "producing_count"
            )
            count = int(population.get(count_key, 0))
            qualifier = "on" if state["is_single_date"] else "during"
            scope = _filter_scope(state.get("filters", []))
            return {
                "answer": (
                    f"There were {count} {population_type} wells{scope} "
                    f"{qualifier} {period}."
                )
            }

        tests = state.get("well_test_result", {})
        if not tests.get("ok", True):
            return {"answer": tests.get("error", "Well-test lookup failed.")}
        wells_key = "active_wells" if population_type == "active" else "producing_wells"
        count_key = "active_count" if population_type == "active" else "producing_count"
        population_names = {
            _normalize_well_name(item.get("well", ""))
            for item in population.get(wells_key, [])
            if item.get("well")
        }
        tested_names = {
            _normalize_well_name(_well_test_display_name(item))
            for item in tests.get("well_tests", [])
            if _well_test_display_name(item)
        }
        missing = sorted(population_names - tested_names)
        tested_population_count = len(population_names & tested_names)
        population_count = int(population.get(count_key, len(population_names)))
        if population_type == "active":
            active_date = _american_date(state["active_date"])
            definition = f"\n\nActive means ONRR-active as of {active_date}."
        else:
            definition = ""
        if not missing:
            if population_type == "active":
                answer = (
                    f"All {population_count} active wells as of {active_date} had "
                    f"a well test from {period}."
                )
                return {"answer": answer + definition}
            return {
                "answer": (
                    f"All {len(population_names)} wells that produced on at least one "
                    f"day from {period} had a well test in that period."
                )
            }
        if population_type == "active":
            summary = (
                f"As of {active_date}, {len(missing)} of {population_count} "
                f"active wells had no well test from {period}. "
                f"{tested_population_count} active wells had at least one test."
                f"{definition}"
            )
        else:
            summary = (
                f"For {period}, {len(missing)} of {len(population_names)} wells that "
                f"produced on at least one day had no well test. "
                f"{tested_population_count} producing wells had at least one test."
            )
        if state.get("output_mode") == "count":
            return {"answer": summary}
        names = "\n".join(f"- {name}" for name in missing)
        return {"answer": f"{summary}\n\n{names}"}

    graph.add_node("fetch_well_population", fetch_well_population)
    graph.add_node("fetch_well_tests", fetch_well_tests)
    graph.add_node("format_answer", format_answer)
    graph.add_edge(START, "fetch_well_population")
    graph.add_conditional_edges(
        "fetch_well_population",
        lambda state: (
            "format_answer"
            if not state["population_result"].get("ok", True)
            else (
                "fetch_well_tests"
                if state["intent"] == "well_test_coverage"
                else "format_answer"
            )
        ),
    )
    graph.add_edge("fetch_well_tests", "format_answer")
    graph.add_edge("format_answer", END)
    return graph.compile()


def _invoke_json_tool(
    tool: BaseTool, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Invoke one structured tool and produce existing chat trace/stat shapes."""
    started_at = perf_counter()
    raw_result = tool.invoke(arguments)
    seconds = perf_counter() - started_at
    result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    return (
        result,
        {"tool": tool.name, "arguments": arguments, "result": result},
        {"tool": tool.name, "seconds": seconds},
    )


def _classify_request(
    question: str,
    today: date,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Recognize explicit producing/active count and test-coverage requests."""
    normalized = " ".join(question.lower().split())
    has_producing_terms = bool(
        re.search(r"\b(producing wells?|oil producers?|active producers?)\b", normalized)
        or re.search(r"\bactive oil producing wells?\b", normalized)
    )
    has_active_terms = bool(re.search(r"\bactive wells?\b", normalized))
    if not has_producing_terms and not has_active_terms:
        return _classify_coverage_follow_up(normalized, today, history or [])
    has_well_test = bool(
        re.search(r"\bwell[ -]?tests?\b", normalized)
        or re.search(r"\btests?\b", normalized)
    )
    coverage_terms = bool(
        re.search(
            r"\b(missing|without|did not|does not|do not|don't|dont|didn't|didnt|"
            r"have not|haven't|havent|no |coverage|at least one|all)\b",
            normalized,
        )
    )
    intent: Literal["count", "well_test_coverage"]
    if has_well_test and coverage_terms:
        intent = "well_test_coverage"
    elif re.search(r"\b(how many|count|number of)\b", normalized):
        intent = "count"
    else:
        return None

    date_range = _resolve_date_range(normalized, today)
    if date_range is None:
        return None
    start_date, end_date = date_range
    population_type: Literal["active", "producing"] = (
        "producing" if has_producing_terms else "active"
    )
    active_date = _resolve_active_population_date(normalized, today) or today
    if population_type == "active" and intent == "count" and start_date != end_date:
        return None
    filters = _extract_well_filters(normalized)
    return {
        "intent": intent,
        "population_type": population_type,
        "active_date": active_date.isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "is_single_date": start_date == end_date,
        "output_mode": _coverage_output_mode(normalized),
        "filters": filters,
    }


def is_producing_well_test_coverage_request(
    question: str,
    history: list[dict[str, str]] | None = None,
    today: date | None = None,
) -> bool:
    """Return whether a turn belongs to deterministic well-test coverage."""
    route = _classify_request(question, today or date.today(), history or [])
    return bool(route and route.get("intent") == "well_test_coverage")


def _classify_coverage_follow_up(
    normalized: str,
    today: date,
    history: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Apply a safe date/output-only follow-up to the latest coverage request."""
    if not _is_safe_coverage_follow_up(normalized):
        return None
    anchor = _latest_coverage_anchor(history, today)
    if anchor is None:
        return None
    replacement_range = _resolve_date_range(normalized, today)
    if replacement_range is None:
        bare_days = re.search(r"\b(\d{1,3})\s+days?\b", normalized)
        if bare_days:
            day_count = int(bare_days.group(1))
            replacement_range = (today - timedelta(days=day_count - 1), today)
    if replacement_range is not None:
        anchor["start_date"] = replacement_range[0].isoformat()
        anchor["end_date"] = replacement_range[1].isoformat()
        anchor["is_single_date"] = replacement_range[0] == replacement_range[1]
    anchor["output_mode"] = _coverage_output_mode(normalized, anchor["output_mode"])
    return anchor


def _latest_coverage_anchor(
    history: list[dict[str, str]], today: date
) -> dict[str, Any] | None:
    """Find the latest complete user request establishing coverage semantics."""
    for item in reversed(history[-10:]):
        if item.get("role") != "user":
            continue
        route = _classify_request(str(item.get("content", "")), today, [])
        if route and route.get("intent") == "well_test_coverage":
            return dict(route)
    return None


def _is_safe_coverage_follow_up(normalized: str) -> bool:
    """Allow only bounded date replacements and count/list refinements."""
    if re.search(
        r"\b(shutdowns?|alarms?|production|allocation|reports?|readings?|tanks?|flares?|injection)\b",
        normalized,
    ):
        return False
    return bool(
        re.match(r"^(?:how|what)\s+about\b", normalized)
        or re.match(r"^and\b", normalized)
        or re.match(r"^use\b", normalized)
        or re.match(r"^(?:show|list|count)\b", normalized)
    )


def _coverage_output_mode(normalized: str, default: str = "all") -> str:
    if re.search(r"\b(count only|how many|number of)\b", normalized):
        return "count"
    if re.search(r"\b(list|show)\b", normalized):
        return "list"
    return default


def _classify_population_dependency(
    question: str, today: date
) -> dict[str, Any] | None:
    """Recognize explicit population dependencies that require further analysis."""
    normalized = " ".join(question.lower().split())
    if _classify_request(question, today) is not None:
        return None

    producing = bool(
        re.search(r"\b(producing wells?|oil producers?|active producers?)\b", normalized)
    )
    active = bool(
        re.search(r"\b(active|online|available) wells?\b", normalized)
    )
    if not producing and not active:
        return None
    resolved = _resolve_date_range(normalized, today)
    if resolved is None:
        return None
    start_date, end_date = resolved

    if producing:
        arguments = (
            {"producing_date": start_date.isoformat()}
            if start_date == end_date
            else {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "range_mode": "any_day",
            }
        )
        filters = _extract_well_filters(normalized)
        if filters:
            arguments["filters"] = filters
        return {"tool_name": "get_producing_wells", "arguments": arguments}
    if start_date != end_date:
        # Active-well range semantics have not been defined. Leave these requests
        # to the normal agent instead of silently applying producing-range rules.
        return None
    return {
        "tool_name": "get_active_wells",
        "arguments": {"active_date": start_date.isoformat()},
    }


def _compact_population_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep the authoritative names and rules while dropping verbose status rows."""
    if tool_name == "get_producing_wells":
        wells_key = "producing_wells"
        count_key = "producing_count"
        population_type = "producing_wells"
    else:
        wells_key = "active_wells"
        count_key = "active_count"
        population_type = "active_wells"
    return {
        "authoritative_population": population_type,
        "date": result.get("date"),
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "range_mode": result.get("range_mode"),
        "count": result.get(count_key, 0),
        "well_names": [
            _normalize_well_name(item.get("well", ""))
            for item in result.get(wells_key, [])
            if item.get("well")
        ],
        "partial_shutdown_well_names": [
            _normalize_well_name(name)
            for name in result.get("partial_shutdown_well_names", [])
        ],
        "filters": result.get("filters", []),
        "rules": result.get("rules") or result.get("rule"),
    }


def _extract_well_filters(text: str) -> list[dict[str, Any]]:
    """Extract small, high-confidence well filters from producing-well questions."""
    filters: list[dict[str, Any]] = []

    for field, value in re.findall(
        r"\b(battery|pump[_\s-]?type|onrr[_\s-]?code)\s*=\s*([a-z0-9 _-]+?)(?=\s+(?:on|in|during|for|yesterday|today|last|this|from|to)\b|[?.!,]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        normalized_field = field.lower().replace(" ", "_").replace("-", "_")
        if normalized_field == "pump_type":
            filters.append({"field": "pump_type", "value": value.strip()})
        elif normalized_field == "onrr_code":
            filters.append({"field": "onrr_code", "value": value.strip()})
        else:
            filters.append({"field": "battery", "value": _battery_filter_value(value)})

    battery_match = re.search(
        r"\bbattery\s+([a-z0-9_-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if battery_match:
        filters.append(
            {"field": "battery", "value": _battery_filter_value(battery_match.group(1))}
        )

    pump_match = re.search(
        r"\b(rod|esp|jet|flowing(?:\s+well(?:\s+no\s+lift)?)?)\s+"
        r"(?:oil\s+)?(?:producing\s+)?(?:wells?|producers?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if pump_match:
        filters.append({"field": "pump_type", "value": pump_match.group(1)})

    deduped: list[dict[str, Any]] = []
    seen = set()
    for item in filters:
        key = (item["field"], str(item["value"]).casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _battery_filter_value(value: str) -> str:
    value = " ".join(str(value).strip().split())
    return f"Battery {value}" if value.isdigit() else value


def _filter_scope(filters: list[dict[str, Any]]) -> str:
    if not filters:
        return ""
    parts = []
    for item in filters:
        field = str(item.get("field", ""))
        value = str(item.get("value", "")).strip()
        if field == "battery" and value:
            parts.append(f"in {value}")
        elif field == "pump_type" and value:
            parts.append(f"with pump type {value.upper()}")
        elif field and value:
            parts.append(f"with {field.replace('_', ' ')} {value}")
    return " " + " and ".join(parts) if parts else ""


def _resolve_active_population_date(text: str, today: date) -> date | None:
    """Resolve explicit dates that qualify the active-well population."""
    relative = re.search(r"\b(?:active wells?|online wells?|available wells?)\s+(today|now|yesterday)\b", text)
    if relative:
        value = relative.group(1)
        return today - timedelta(days=1) if value == "yesterday" else today

    iso_match = re.search(
        r"\b(?:active wells?|online wells?|available wells?)\s+(?:on|as of)\s+(\d{4}-\d{2}-\d{2})\b",
        text,
    )
    if iso_match:
        return date.fromisoformat(iso_match.group(1))

    month_day_match = re.search(
        rf"\b(?:active wells?|online wells?|available wells?)\s+(?:on|as of)\s+({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b",
        text,
    )
    if month_day_match:
        month = MONTHS[month_day_match.group(1)]
        day = int(month_day_match.group(2))
        year = int(month_day_match.group(3) or today.year)
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def _resolve_date_range(text: str, today: date) -> tuple[date, date] | None:
    """Resolve common explicit dates without paying for a classifier model call."""
    trailing_days = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", text)
    if trailing_days:
        day_count = int(trailing_days.group(1))
        if day_count <= 0:
            return None
        return today - timedelta(days=day_count - 1), today
    if re.search(r"\btoday\b|\bnow\b", text):
        return today, today
    if re.search(r"\byesterday\b", text):
        value = today - timedelta(days=1)
        return value, value
    if "this month" in text:
        return date(today.year, today.month, 1), today
    if "last month" in text:
        end = date(today.year, today.month, 1) - timedelta(days=1)
        return date(end.year, end.month, 1), end

    iso_dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_dates:
        values = [date.fromisoformat(value) for value in iso_dates[:2]]
        return (values[0], values[-1])

    month_match = re.search(
        rf"\b({'|'.join(MONTHS)})\b(?:\s+(\d{{4}}))?", text
    )
    if month_match:
        month = MONTHS[month_match.group(1)]
        year = int(month_match.group(2) or today.year)
        start = date(year, month, 1)
        next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        return start, next_month - timedelta(days=1)
    return None


def _normalize_well_name(value: Any) -> str:
    """Normalize display prefixes so independent tool results can be joined."""
    name = str(value).strip()
    return re.sub(r"^well\s*-\s*", "", name, flags=re.IGNORECASE)


def _well_test_display_name(item: dict[str, Any]) -> Any:
    """Read the canonical well-test display field with legacy fallbacks."""
    return (
        item.get("entity_display_name")
        or item.get("entity")
        or item.get("entity_name")
        or ""
    )


def _american_date(value: str) -> str:
    return date.fromisoformat(value).strftime("%m/%d/%Y")
