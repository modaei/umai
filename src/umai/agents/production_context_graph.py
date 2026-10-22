from __future__ import annotations

import json
import re
from datetime import date
from time import perf_counter
from typing import Any, TypedDict

from langchain_core.tools import BaseTool

from umai.clients.reading_client import ReadingClientError
from umai.clients.well_filter_client import WellFilterClientError


class ProductionContextDependency(TypedDict):
    """Production trend plus operational context supplied to the general agent."""

    context: str
    traces: list[dict[str, Any]]
    stats: dict[str, Any]
    tool_name: str


def prepare_production_context_dependency(
    *,
    tools: list[BaseTool],
    question: str,
    history: list[dict[str, str]] | None = None,
    site_id: int,
    today: date | None = None,
    reading_client: Any | None = None,
    well_filter_client: Any | None = None,
    max_related_entities: int = 25,
) -> ProductionContextDependency | None:
    """Prefetch report trend and notes for production variation/reason questions."""
    route = _classify_production_context(question, today=today)
    if route is None:
        route = _classify_production_context_follow_up(
            question,
            history or [],
            today=today,
        )
    if route is None:
        return None
    tool_map = {tool.name: tool for tool in tools}
    report_tool = tool_map.get("summarize_report_by_month")
    context_tool = tool_map.get("search_operational_context")
    if report_tool is None or context_tool is None:
        return None

    started_at = perf_counter()
    traces = []
    timings = []
    related_entities = _resolve_related_battery_entities(
        site_id=site_id,
        battery_name=route["battery_name"],
        reading_client=reading_client,
        well_filter_client=well_filter_client,
        max_related_entities=max_related_entities,
    )

    report_result, report_trace, report_timing = _invoke_json_tool(
        report_tool,
        route["report_arguments"],
    )
    traces.append(report_trace)
    timings.append(report_timing)

    context_results = []
    for arguments in _context_search_arguments(route, related_entities):
        context_result, context_trace, context_timing = _invoke_json_tool(
            context_tool,
            arguments,
        )
        context_result = _filter_context_result_for_battery_scope(
            context_result,
            related_entities,
        )
        context_trace = context_trace | {"result": context_result}
        traces.append(context_trace)
        timings.append(context_timing)
        context_results.append(
            {
                "searched_entity_name": arguments.get("entity_name"),
                "arguments": arguments,
                "result": context_result,
            }
        )

    context = {
        "authoritative_data_type": "production_variation_context",
        "instruction": (
            "Use the production report to describe the variation. Use only the "
            "filtered operational-context results for causal explanations. The "
            "filtered results are scoped to the requested battery and verified "
            "related wells/equipment. Do not infer causes from site-level notes "
            "or unrelated well/tract notes. If context has no supporting records, "
            "say no supporting battery-scoped operational notes were found."
        ),
        "related_entities": related_entities,
        "focus_periods": route.get("focus_periods", []),
        "report": {
            "tool": "summarize_report_by_month",
            "arguments": route["report_arguments"],
            "result": report_result,
        },
        "operational_context": context_results[0] if context_results else None,
        "operational_context_by_entity": {
            "tool": "search_operational_context",
            "results": context_results,
        },
    }
    return {
        "context": json.dumps(context, separators=(",", ":"), default=str),
        "traces": traces,
        "stats": {
            "total_seconds": perf_counter() - started_at,
            "model_seconds": 0.0,
            "tool_seconds": sum(item["seconds"] for item in timings),
            "model_calls": 0,
            "tool_calls": timings,
        },
        "tool_name": "production_context_dependency",
    }


def _classify_production_context(
    question: str,
    today: date | None = None,
) -> dict[str, Any] | None:
    normalized = " ".join(question.lower().split())
    if not re.search(r"\b(why|reason|caus(?:e|ed|es)|explain)\b", normalized):
        return None
    if not re.search(
        r"\b(variation|varied|vary|drop(?:ped)?|dip|increase|decrease|change|"
        r"lower|higher|less|more|below|above)\b",
        normalized,
    ):
        return None
    if not re.search(r"\b(oil production|production|oil)\b", normalized):
        return None
    battery_match = re.search(r"\bbattery\s+([a-z0-9_-]+)\b", normalized)
    if battery_match is None:
        return None
    year_match = re.search(r"\b(20\d{2})\b", normalized)
    month_pair = _month_pair(normalized)
    if year_match is None and month_pair is None:
        return None

    battery_name = _battery_name(battery_match.group(1))
    year = int(year_match.group(1)) if year_match else (today or date.today()).year
    focus_periods = (
        [f"{year}-{month_pair[0]:02d}", f"{year}-{month_pair[1]:02d}"]
        if month_pair
        else []
    )
    comparison_terms = _comparison_terms(normalized)
    query = " ".join(
        item
        for item in [
            battery_name,
            "oil production",
            comparison_terms or "variation",
            _month_name(month_pair[0]) if month_pair else "",
            _month_name(month_pair[1]) if month_pair else "",
            "reason",
            str(year),
        ]
        if item
    )
    return {
        "battery_name": battery_name,
        "year": year,
        "focus_periods": focus_periods,
        "report_arguments": {
            "report_name": "oil_production",
            "year": year,
            "battery_name": battery_name,
            "value_key": None,
        },
        "context_arguments": {
            "query": query,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "entity_name": battery_name,
            "limit": 10,
        },
    }


def _classify_production_context_follow_up(
    question: str,
    history: list[dict[str, str]],
    today: date | None = None,
) -> dict[str, Any] | None:
    """Resolve short reason follow-ups from recent production comparison turns."""
    normalized_question = " ".join(question.lower().split())
    if not _is_reason_follow_up(normalized_question):
        return None

    for item in reversed(history[-8:]):
        content = str(item.get("content") or "")
        expanded_question = _production_context_question_from_history(
            content,
            today=today,
        )
        if expanded_question:
            return _classify_production_context(expanded_question, today=today)
    return None


def _is_reason_follow_up(text: str) -> bool:
    """Return whether a short message is asking for the cause of prior context."""
    return bool(
        re.fullmatch(
            r"(?:why|why\?|what caused (?:it|that|this)\??|"
            r"what was the reason\??|explain(?: that| it| this)?\??|"
            r"reason\??|cause\??)",
            text,
        )
    )


def _production_context_question_from_history(
    content: str,
    today: date | None = None,
) -> str | None:
    """Build a self-contained reason question from a prior comparison message."""
    normalized = " ".join(content.lower().split())
    if not re.search(r"\bbattery\s+([a-z0-9_-]+)\b", normalized):
        return None
    if not re.search(r"\b(oil production|produced|production)\b", normalized):
        return None

    battery_match = re.search(r"\bbattery\s+([a-z0-9_-]+)\b", normalized)
    month_pair = _month_pair_from_history(normalized)
    if battery_match is None or month_pair is None:
        return None

    year = _year_from_history(normalized, today=today)
    battery_name = _battery_name(battery_match.group(1))
    return (
        f"Why did {battery_name} oil production change in "
        f"{_month_name(month_pair[0])} than in {_month_name(month_pair[1])} {year}?"
    )


def _month_pair_from_history(text: str) -> tuple[int, int] | None:
    """Extract two compared months from user or assistant history text."""
    direct_pair = _month_pair(text)
    if direct_pair is not None:
        return direct_pair

    month_names = []
    for match in re.finditer(r"\b(" + "|".join(MONTHS) + r")\b", text):
        month = MONTHS[match.group(1)]
        if month not in month_names:
            month_names.append(month)
        if len(month_names) == 2:
            return month_names[0], month_names[1]
    return None


def _year_from_history(text: str, today: date | None = None) -> int:
    """Resolve the comparison year from history, falling back to current year."""
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        return int(year_match.group(1))
    return (today or date.today()).year


def _battery_name(value: str) -> str:
    value = " ".join(str(value).strip().split())
    return f"Battery {value}" if value.isdigit() else value


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


def _month_pair(text: str) -> tuple[int, int] | None:
    month_pattern = "|".join(MONTHS)
    match = re.search(
        rf"\b({month_pattern})\b\s+"
        rf"(?:than|vs|versus|compared\s+to|compared\s+with|and|with)\s+"
        rf"(?:in\s+)?"
        rf"\b({month_pattern})\b",
        text,
    )
    if match is None:
        return None
    return MONTHS[match.group(1)], MONTHS[match.group(2)]


def _month_name(month: int) -> str:
    for name, value in MONTHS.items():
        if value == month:
            return name.title()
    return ""


def _comparison_terms(text: str) -> str | None:
    terms = [
        term
        for term in (
            "lower",
            "higher",
            "less",
            "more",
            "below",
            "above",
            "drop",
            "dropped",
            "dip",
            "increase",
            "decrease",
            "change",
            "variation",
        )
        if re.search(rf"\b{re.escape(term)}\b", text)
    ]
    return " ".join(terms) if terms else None


def _resolve_related_battery_entities(
    *,
    site_id: int,
    battery_name: str,
    reading_client: Any | None,
    well_filter_client: Any | None,
    max_related_entities: int,
) -> dict[str, Any]:
    related: dict[str, Any] = {
        "battery_name": battery_name,
        "wells": [],
        "equipment": {},
        "errors": [],
    }
    if well_filter_client is not None:
        try:
            wells = well_filter_client.find_wells(
                site_id=site_id,
                filters=[{"field": "battery", "value": battery_name}],
            )
        except TypeError:
            try:
                wells = well_filter_client.find_wells(
                    site_id,
                    filters=[{"field": "battery", "value": battery_name}],
                )
            except (WellFilterClientError, RuntimeError, ValueError) as exc:
                related["errors"].append(f"well resolution failed: {exc}")
            else:
                related["wells"] = _entity_names(wells.get("wells", []), "name")
        except (WellFilterClientError, RuntimeError, ValueError) as exc:
            related["errors"].append(f"well resolution failed: {exc}")
        else:
            related["wells"] = _entity_names(wells.get("wells", []), "name")

    if reading_client is not None:
        for equipment_type in _equipment_types():
            try:
                result = reading_client.list_equipment(
                    site_id,
                    equipment_type,
                    battery_name=battery_name,
                )
            except TypeError:
                try:
                    result = reading_client.list_equipment(
                        site_id=site_id,
                        equipment_type=equipment_type,
                        battery_name=battery_name,
                    )
                except (ReadingClientError, RuntimeError, ValueError) as exc:
                    related["errors"].append(f"{equipment_type} resolution failed: {exc}")
                    continue
            except (ReadingClientError, RuntimeError, ValueError) as exc:
                related["errors"].append(f"{equipment_type} resolution failed: {exc}")
                continue
            names = _entity_names(result.get("entities", []), "entity_name")
            if names:
                related["equipment"][equipment_type] = names

    related["search_entity_names"] = _bounded_related_entity_names(
        battery_name,
        related,
        max_related_entities,
    )
    return related


def _equipment_types() -> tuple[str, ...]:
    return (
        "tank",
        "lact",
        "flare",
        "flow_meter",
        "treater",
        "knock_out",
        "water_plant",
        "pump",
    )


def _entity_names(rows: list[dict[str, Any]], key: str) -> list[str]:
    names = []
    seen = set()
    for row in rows:
        name = str(row.get(key) or "").strip()
        if not name:
            continue
        dedupe_key = name.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        names.append(name)
    return names


def _bounded_related_entity_names(
    battery_name: str,
    related: dict[str, Any],
    max_related_entities: int,
) -> list[str]:
    names = [battery_name]
    names.extend(related.get("wells", []))
    for equipment_names in related.get("equipment", {}).values():
        names.extend(equipment_names)
    deduped = []
    seen = set()
    for name in names:
        key = str(name).casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped[: max(1, max_related_entities)]


def _context_search_arguments(
    route: dict[str, Any],
    related_entities: dict[str, Any],
) -> list[dict[str, Any]]:
    arguments = []
    for entity_name in related_entities.get("search_entity_names", []) or [
        route["battery_name"]
    ]:
        arguments.append(
            {
                **route["context_arguments"],
                "entity_name": entity_name,
            }
        )
    return arguments


def _filter_context_result_for_battery_scope(
    result: dict[str, Any],
    related_entities: dict[str, Any],
) -> dict[str, Any]:
    """Keep only matches that explicitly reference the battery scope.

    Broad general notes can mention unrelated wells/tracts while ranking well for
    a battery query. The production-cause dependency is stricter than generic RAG:
    a note must be tied to the requested battery, one of its verified wells, or
    one of its verified pieces of equipment before the LLM can use it as causal
    evidence.
    """
    matches = result.get("matches")
    if not isinstance(matches, list):
        return result

    allowed_names = _battery_scope_allowed_names(related_entities)
    kept = [
        match
        for match in matches
        if isinstance(match, dict)
        and _match_references_allowed_entity(match, allowed_names)
    ]
    return {
        **result,
        "count": len(kept),
        "matches": kept,
        "original_count": len(matches),
        "filtered_count": len(kept),
        "excluded_count": len(matches) - len(kept),
    }


def _battery_scope_allowed_names(related_entities: dict[str, Any]) -> list[str]:
    names = [str(related_entities.get("battery_name") or "").strip()]
    names.extend(str(name).strip() for name in related_entities.get("wells", []))
    for equipment_names in related_entities.get("equipment", {}).values():
        names.extend(str(name).strip() for name in equipment_names)

    aliases = []
    for name in names:
        if not name:
            continue
        aliases.append(name)
        aliases.extend(_entity_aliases(name))
    return _dedupe_names(aliases)


def _entity_aliases(name: str) -> list[str]:
    normalized = " ".join(name.split())
    well_match = re.search(r"\bhartzog\s+draw\s+unit\s+([a-z0-9-]+)\b", normalized, re.I)
    if well_match:
        return [well_match.group(1)]
    return []


def _match_references_allowed_entity(
    match: dict[str, Any],
    allowed_names: list[str],
) -> bool:
    entity_name = str(match.get("entity_name") or "")
    text = str(match.get("text") or "")
    return any(
        _contains_entity_reference(entity_name, allowed_name)
        or _contains_entity_reference(text, allowed_name)
        for allowed_name in allowed_names
    )


def _contains_entity_reference(text: str, entity_name: str) -> bool:
    normalized_text = _normalize_entity_text(text)
    normalized_entity = _normalize_entity_text(entity_name)
    if not normalized_text or not normalized_entity:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_entity)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _normalize_entity_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _dedupe_names(names: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for name in names:
        key = _normalize_entity_text(name)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _invoke_json_tool(
    tool: BaseTool, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started_at = perf_counter()
    try:
        raw_result = tool.invoke(arguments)
        result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except Exception as exc:
        result = {"ok": False, "error": f"{tool.name} failed: {exc}"}
    seconds = perf_counter() - started_at
    return (
        result,
        {"tool": tool.name, "arguments": arguments, "result": result},
        {"tool": tool.name, "seconds": seconds},
    )
