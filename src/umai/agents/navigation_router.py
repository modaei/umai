from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import date, timedelta
from time import perf_counter
from typing import Any

from langchain_core.tools import BaseTool

from umai.agents.producing_wells_graph import MONTHS, _resolve_date_range
from umai.clients.shutdown_client import DOWNTIME_CODES


DeterministicAnswer = tuple[str, list[dict[str, Any]], dict[str, Any]]


VIEW_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("production_allocation", ("production allocation",)),
    ("injection_allocation", ("injection allocation",)),
    ("oil_production", ("oil production",)),
    ("oil_sale", ("oil sales", "oil sale")),
    ("water_production", ("water production",)),
    ("water_transfer", ("water transfer",)),
    ("water_injection", ("water injection",)),
    ("flared_vent", ("gas flared", "gas vented", "flared gas", "vented gas")),
    ("fuel_gas", ("fuel gas",)),
    ("battery", ("battery report",)),
    ("flow_meter_readings", ("flow meter reading", "flowmeter reading")),
    ("water_plant_readings", ("water plant reading",)),
    ("tank_readings", ("tank reading",)),
    ("flare_readings", ("flare reading",)),
    ("pump_readings", ("pump reading",)),
    ("treater_readings", ("treater reading",)),
    ("knockout_readings", ("knockout reading", "knock out reading")),
    ("lact_readings", ("lact reading",)),
    ("well_tests", ("well test",)),
    ("well_fluids", ("well fluid level", "well fluid", "fluid level")),
    ("well_injections", ("well injection", "injection reading")),
    ("shutdowns", ("shutdown",)),
    ("general_notes", ("general note",)),
    ("work_orders", ("work order",)),
    ("run_tickets", ("run ticket",)),
    ("water_draws", ("water draw",)),
)


ENTRY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("flow_meter_reading", ("flow meter reading", "flowmeter reading")),
    ("water_plant_reading", ("water plant reading",)),
    ("knockout_reading", ("knockout reading", "knock out reading")),
    ("flare_reading", ("flare reading",)),
    ("pump_reading", ("pump reading",)),
    ("treater_reading", ("treater reading",)),
    ("lact_reading", ("lact reading",)),
    ("tank_reading", ("tank reading",)),
    ("well_test", ("well test",)),
    ("well_fluid", ("well fluid level", "fluid level")),
    ("well_injection", ("well injection", "injection reading")),
    ("well_shutdown", ("well shutdown", "shutdown")),
    ("water_draw", ("water draw",)),
    ("run_ticket", ("run ticket",)),
    ("general_note", ("general note",)),
    ("work_order", ("work order",)),
)


ENTRY_FIELDS: dict[str, tuple[str, ...]] = {
    "flare_reading": ("pressure", "volume"),
    "flow_meter_reading": ("total", "flow", "odometer"),
    "pump_reading": ("suction_pressure", "discharge_pressure"),
    "treater_reading": ("oil_intake", "pressure", "temperature"),
    "knockout_reading": ("inlet", "oil_off"),
    "water_plant_reading": ("flow_rate", "suction_pressure", "discharge_pressure"),
    "lact_reading": ("reading", "temperature", "bs_w"),
    "tank_reading": (
        "top_level_feet",
        "top_level_inches",
        "water_level_feet",
        "water_level_inches",
        "initial_feet",
        "initial_inches",
        "final_feet",
        "final_inches",
        "level",
        "feet",
        "inches",
        "temperature",
    ),
    "well_test": (
        "oil",
        "water",
        "gas",
        "pip",
        "m_temp",
        "amps",
        "tbgp",
        "csgp",
        "fluid_level",
        "runtime",
    ),
    "well_fluid": ("level",),
    "well_injection": ("flow_rate", "total", "tbg", "csg"),
    "well_shutdown": ("hours",),
    "water_draw": ("initial_feet", "initial_inches", "final_feet", "final_inches"),
    "run_ticket": (
        "number",
        "obs_grav",
        "obs_temp",
        "b_s_w",
        "corr_grav",
        "gov",
        "nsv",
        "initial_feet",
        "initial_inches",
        "initial_qtr",
        "final_feet",
        "final_inches",
        "final_qtr",
        "initial_volume",
        "final_volume",
        "initial_meter_reading",
        "final_meter_reading",
    ),
    "work_order": ("cost_estimate", "final_cost", "priority"),
}


READING_ENTITY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("flow_meter_reading", ("flow meter", "flowmeter")),
    ("water_plant_reading", ("water plant",)),
    ("knockout_reading", ("knockout", "knock out")),
    ("flare_reading", ("flare",)),
    ("pump_reading", ("pump",)),
    ("treater_reading", ("treater",)),
    ("lact_reading", ("lact",)),
    ("tank_reading", ("tank",)),
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "bs_w": ("bsw", "bs w", "b s w"),
    "b_s_w": ("bsw", "bs w", "b s w"),
    "top_level_feet": ("top level feet", "top level foot", "top level fit"),
    "top_level_inches": ("top level inches", "top level inch"),
    "water_level_feet": ("water level feet", "water level foot", "water level fit"),
    "water_level_inches": ("water level inches", "water level inch"),
    "initial_feet": ("initial feet", "initial foot", "initial fit"),
    "initial_inches": ("initial inches", "initial inch"),
    "final_feet": ("final feet", "final foot", "final fit"),
    "final_inches": ("final inches", "final inch"),
    "runtime": ("runtime", "run time"),
    "feet": ("feet", "foot", "fit"),
    "inches": ("inches", "inch"),
}

SPOKEN_NUMBERS: dict[str, float] = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def try_answer_navigation_request(
    *,
    tools: list[BaseTool],
    question: str,
    today: date | None = None,
) -> DeterministicAnswer | None:
    """Route unambiguous view and form-navigation requests without an LLM call."""
    current_date = today or date.today()
    route = _parse_view_request(question, current_date)
    tool_name = "prepare_data_view"
    if route is None:
        route = _parse_capability_help_request(question)
        tool_name = "search_ometrics_capabilities"
    if route is None:
        route = _parse_data_entry_request(question, current_date)
        tool_name = "prepare_data_entry"
    if route is None:
        return None

    tool = next((item for item in tools if item.name == tool_name), None)
    if tool is None:
        return None
    started_at = perf_counter()
    tool_started_at = perf_counter()
    try:
        raw_result = tool.invoke(route)
        result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except Exception as exc:
        result = {"status": "invalid", "message": f"Navigation failed: {exc}"}
        raw_result = json.dumps(result, separators=(",", ":"))
    tool_seconds = perf_counter() - tool_started_at
    trace = {"tool": tool_name, "arguments": route, "result": raw_result}
    return (
        _format_navigation_result(result, tool_name),
        [trace],
        {
            "total_seconds": perf_counter() - started_at,
            "model_seconds": 0.0,
            "tool_seconds": tool_seconds,
            "model_calls": 0,
            "tool_calls": [{"tool": tool_name, "seconds": tool_seconds}],
        },
    )


def _parse_view_request(question: str, today: date) -> dict[str, Any] | None:
    """Parse explicit show-me navigation while rejecting uncertain destinations."""
    normalized = _normalize(question)
    if not normalized.startswith("show me "):
        return None
    matches = _alias_matches(normalized, VIEW_ALIASES)
    if len(matches) != 1:
        return None
    arguments: dict[str, Any] = {"view_type": matches[0]}
    date_range = _safe_date_range(normalized, today)
    if _contains_date_marker(normalized) and date_range is None:
        return None
    if date_range:
        arguments["start_date"] = date_range[0].isoformat()
        arguments["end_date"] = date_range[1].isoformat()
    search_text = _view_search_text(normalized, today)
    if search_text:
        arguments["search_text"] = search_text
    if matches[0] == "work_orders":
        status_match = re.search(r"\b(open|closed|all)\s+work orders?\b", normalized)
        if status_match:
            arguments["status"] = status_match.group(1)
    return arguments


def _parse_capability_help_request(question: str) -> dict[str, Any] | None:
    """Parse explicit Ometrics workflow-help questions before form actions."""
    normalized = _normalize(question)
    help_prefix = re.match(
        r"^(?:"
        r"how\s+(?:do|can|should)\s+i|"
        r"where\s+(?:do|can|should)\s+i|"
        r"explain\s+how\s+(?:to|i)|"
        r"show\s+me\s+how\s+(?:to|i)|"
        r"what\s+is\s+the\s+process\s+(?:to|for)"
        r")\b",
        normalized,
    )
    if not help_prefix:
        return None
    if not _looks_like_ometrics_workflow_help(normalized):
        return None
    return {"query": question.strip(), "limit": 3}


def _looks_like_ometrics_workflow_help(normalized: str) -> bool:
    workflow_verbs = (
        "register",
        "create",
        "add",
        "enter",
        "record",
        "open",
        "find",
        "view",
        "use",
        "run",
    )
    workflow_nouns = (
        "reading",
        "well test",
        "fluid level",
        "injection",
        "shutdown",
        "water draw",
        "run ticket",
        "general note",
        "work order",
        "report",
        "allocation",
        "lact",
        "tank",
        "flare",
        "flow meter",
        "water plant",
        "ometrics",
    )
    return any(re.search(rf"\b{re.escape(verb)}\b", normalized) for verb in workflow_verbs) and any(
        re.search(rf"\b{re.escape(noun)}s?\b", normalized) for noun in workflow_nouns
    )


def _parse_data_entry_request(question: str, today: date) -> dict[str, Any] | None:
    """Parse one explicit form request, falling through when values may be lost."""
    normalized = _normalize(question)
    if not re.search(r"\b(create|add|enter|record|register)\b", normalized):
        return None
    equipment_terms = re.findall(
        r"\b(lact|flare|pump|treater|tank|flow meter|flowmeter|water plant|"
        r"knockout|knock out|well)\b",
        normalized,
    )
    if " and " in normalized and len(set(equipment_terms)) > 1:
        return None
    matches = _alias_matches(normalized, ENTRY_ALIASES)
    if len(matches) > 1:
        return None
    if not matches:
        if not re.search(r"\b(?:create|add|enter|record|register)\b.*\breading\b", normalized):
            return None
        entry_type = "generic_reading"
    else:
        entry_type = matches[0]
    if entry_type == "well_shutdown":
        return _parse_shutdown_entry_request(normalized, today)

    date_range = _safe_date_range(normalized, today)
    if _contains_date_marker(normalized) and date_range is None:
        return None
    entity_name = _entry_entity_name(normalized, entry_type, today)
    if entry_type == "generic_reading" and entity_name:
        inferred_types = _reading_types_from_entity(entity_name)
        if not inferred_types:
            inferred_types = _reading_types_from_fields(normalized)
        if len(inferred_types) > 1:
            return None
        if inferred_types:
            entry_type = inferred_types[0]
            entity_name = _strip_entry_field_tail(entity_name, entry_type)

    # Field extraction must happen after generic requests have been resolved to a
    # concrete reading type. Otherwise type-specific values remain unexplained.
    values, consumed_numeric_spans = _entry_values(normalized, entry_type, today)
    entity_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", entity_name or ""))
    # Unknown numbers often represent form values. Let the model interpret them
    # rather than silently opening a form with incomplete prefilled data.
    for numeric_match in re.finditer(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", normalized):
        if _span_is_consumed(numeric_match.span(), consumed_numeric_spans):
            continue
        if numeric_match.group() in entity_numbers:
            continue
        if _number_is_part_of_date(normalized, numeric_match.span()):
            continue
        return None

    arguments: dict[str, Any] = {"entry_type": entry_type, "values": values}
    if entity_name:
        arguments["entity_name"] = entity_name
    return arguments


def _parse_shutdown_entry_request(text: str, today: date) -> dict[str, Any] | None:
    """Parse explicit shutdown form requests into the current interval fields."""
    entity_name = _shutdown_entity_name(text, today)
    values: dict[str, Any] = {}
    consumed: list[tuple[int, int]] = []

    start_value = _shutdown_datetime_value(
        text,
        today,
        ("start", "starting", "starts", "from"),
    )
    if start_value:
        values["start"] = start_value[0]
        consumed.append(start_value[1])

    end_value = _shutdown_datetime_value(
        text,
        today,
        ("end", "ending", "ends", "until", "to"),
    )
    if end_value:
        values["end"] = end_value[0]
        consumed.append(end_value[1])

    reason_value = _shutdown_reason_value(text)
    if reason_value:
        code, span = reason_value
        if code:
            values["downtime_code"] = code
        consumed.append(span)

    entity_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", entity_name or ""))
    for numeric_match in re.finditer(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", text):
        if _span_is_consumed(numeric_match.span(), consumed):
            continue
        if numeric_match.group() in entity_numbers:
            continue
        if _number_is_part_of_date(text, numeric_match.span()):
            continue
        return None

    arguments: dict[str, Any] = {"entry_type": "well_shutdown", "values": values}
    if entity_name:
        arguments["entity_name"] = entity_name
    return arguments


def _shutdown_entity_name(text: str, today: date) -> str | None:
    """Extract the well name from a shutdown command without swallowing clauses."""
    match = re.search(r"\bfor\s+(.+)$", text)
    if not match:
        return None
    candidate = match.group(1).strip(" .?")
    candidate = re.split(
        r"\s+(?:start|starting|starts|from|end|ending|ends|until|to|"
        r"downtime\s+reason|reason|because|due\s+to|comments?|with)\b",
        candidate,
        maxsplit=1,
    )[0].strip(" .?")
    return None if _safe_date_range(candidate, today) else candidate or None


def _shutdown_datetime_value(
    text: str,
    today: date,
    keywords: tuple[str, ...],
) -> tuple[str, tuple[int, int]] | None:
    """Resolve a labelled shutdown date/time clause into Ometrics form format."""
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    date_pattern = _shutdown_date_pattern()
    time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    match = re.search(
        rf"\b(?:{keyword_pattern})\s+({date_pattern})"
        rf"(?:\s+(?:at\s+)?{time_pattern})?",
        text,
    )
    if not match:
        return None

    resolved_date = _parse_shutdown_date_token(match.group(1), today)
    if resolved_date is None:
        return None

    hour = 0
    minute = 0
    if match.group(2) is not None:
        parsed_time = _parse_shutdown_time_parts(
            match.group(2),
            match.group(3),
            match.group(4),
        )
        if parsed_time is None:
            return None
        hour, minute = parsed_time

    return (
        f"{resolved_date.isoformat()} {hour:02d}:{minute:02d}",
        match.span(),
    )


def _shutdown_date_pattern() -> str:
    """Date tokens supported by the deterministic shutdown create route."""
    months = "|".join(MONTHS)
    return (
        r"today|yesterday|"
        r"\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}/\d{1,2}/\d{4}|"
        rf"(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    )


def _parse_shutdown_date_token(value: str, today: date) -> date | None:
    """Parse one deterministic shutdown date token."""
    normalized = value.strip(" ,.").lower()
    if normalized == "today":
        return today
    if normalized == "yesterday":
        return today - timedelta(days=1)
    return _parse_date_endpoint(normalized, today, "start")


def _parse_shutdown_time_parts(
    hour_text: str,
    minute_text: str | None,
    meridiem: str | None,
) -> tuple[int, int] | None:
    """Parse a deterministic time token with optional am/pm."""
    hour = int(hour_text)
    minute = int(minute_text or "0")
    if minute > 59:
        return None
    if meridiem:
        if hour < 1 or hour > 12:
            return None
        hour = hour % 12
        if meridiem == "pm":
            hour += 12
    elif hour > 23:
        return None
    return hour, minute


def _shutdown_reason_value(text: str) -> tuple[str | None, tuple[int, int]] | None:
    """Resolve an explicit shutdown reason to a known downtime code if exact."""
    match = re.search(
        r"\b(?:downtime\s+reason|reason|because|due\s+to)\s+(.+?)"
        r"(?=\s+(?:start|starting|starts|from|end|ending|ends|until|to|"
        r"comments?|with)\b|$)",
        text,
    )
    if not match:
        return None
    reason = match.group(1).strip(" .?")
    return _downtime_code_for_reason(reason), match.span()


def _downtime_code_for_reason(reason: str) -> str | None:
    """Map exact downtime codes or exact reason labels to a form code."""
    normalized = _normalize_reason(reason)
    if not normalized:
        return None
    upper_reason = reason.strip().upper()
    if upper_reason in DOWNTIME_CODES:
        return upper_reason
    for code, label in DOWNTIME_CODES.items():
        if normalized == _normalize_reason(label):
            return code
    return None


def _normalize_reason(value: str) -> str:
    """Normalize a downtime reason for exact deterministic matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _reading_types_from_entity(entity_name: str) -> list[str]:
    """Infer reading types from explicit equipment words in an entity phrase."""
    matches = []
    for entry_type, markers in READING_ENTITY_MARKERS:
        if any(re.search(rf"\b{re.escape(marker)}\b", entity_name) for marker in markers):
            matches.append(entry_type)
    return list(dict.fromkeys(matches))


def _reading_types_from_fields(text: str) -> list[str]:
    """Infer one reading form from distinctive field labels when no marker exists."""
    field_hits = {
        entry_type: {
            field for field in fields
            if _has_field_label(text, field)
        }
        for entry_type, fields in ENTRY_FIELDS.items()
    }
    matches = []
    if field_hits["tank_reading"] & {
        "top_level_feet", "top_level_inches", "water_level_feet",
        "water_level_inches", "initial_feet", "initial_inches",
        "final_feet", "final_inches",
    } or field_hits["tank_reading"] >= {"feet", "inches"}:
        matches.append("tank_reading")
    if field_hits["flow_meter_reading"] & {"flow", "total", "odometer"}:
        matches.append("flow_meter_reading")
    if (
        "flow_rate" in field_hits["water_plant_reading"]
        and field_hits["water_plant_reading"]
        & {"suction_pressure", "discharge_pressure"}
    ):
        matches.append("water_plant_reading")
    if field_hits["pump_reading"] >= {"suction_pressure", "discharge_pressure"}:
        matches.append("pump_reading")
    if field_hits["flare_reading"] >= {"pressure", "volume"}:
        matches.append("flare_reading")
    if field_hits["treater_reading"] & {"oil_intake"}:
        matches.append("treater_reading")
    if field_hits["knockout_reading"] & {"inlet", "oil_off"}:
        matches.append("knockout_reading")
    if field_hits["lact_reading"] & {"bs_w"}:
        matches.append("lact_reading")
    if field_hits["well_injection"] & {"tbg", "csg"}:
        matches.append("well_injection")
    if field_hits["well_fluid"] and not matches:
        matches.append("well_fluid")
    if "water_plant_reading" in matches and "pump_reading" in matches:
        matches.remove("pump_reading")
    if "well_injection" in matches and "flow_meter_reading" in matches:
        matches.remove("flow_meter_reading")
    return list(dict.fromkeys(matches))


def _has_field_label(text: str, field: str) -> bool:
    return any(
        re.search(rf"\b{label}\s*(?:is|=|:)?\s*({_number_value_pattern()})\b", text)
        for label in _field_labels(field)
    )


def _alias_matches(
    text: str,
    aliases: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[str]:
    """Return unique canonical types whose complete aliases occur in the text."""
    found = []
    for canonical, variants in aliases:
        if any(re.search(rf"\b{re.escape(alias)}s?\b", text) for alias in variants):
            found.append(canonical)
    if "well_test" in found and "well_fluid" in found:
        found.remove("well_fluid")
    return list(dict.fromkeys(found))


def _safe_date_range(text: str, today: date) -> tuple[date, date] | None:
    """Resolve supported relative, ISO, and US-formatted dates without guessing."""
    explicit_range = re.search(
        r"\bfrom\s+(.+?)\s+(?:until|to|through)\s+(.+?)"
        r"(?:\s+for\s+.+)?\s*$",
        text,
    )
    if not explicit_range:
        explicit_range = re.search(
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+for\s+.+)?\s*$",
            text,
        )
    if explicit_range:
        start = _parse_date_endpoint(explicit_range.group(1), today, "start")
        end = _parse_date_endpoint(explicit_range.group(2), today, "end")
        return (start, end) if start and end else None

    named_dates = re.findall(
        rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        r"(?:,?\s+(\d{4}))?\b",
        text,
    )
    if named_dates:
        try:
            values = [
                date(int(year or today.year), MONTHS[month], int(day))
                for month, day, year in named_dates[:2]
            ]
        except ValueError:
            return None
        return values[0], values[-1]
    us_dates = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if us_dates:
        try:
            values = [
                date(int(year), int(month), int(day))
                for month, day, year in us_dates[:2]
            ]
        except ValueError:
            return None
        return values[0], values[-1]
    try:
        return _resolve_date_range(text, today)
    except ValueError:
        return None


def _parse_date_endpoint(
    value: str,
    today: date,
    boundary: str,
) -> date | None:
    """Parse one boundary in an explicit from/between date expression."""
    normalized = value.strip(" ,.?\t\n").lower()
    if normalized in {"today", "now"}:
        return today
    if normalized == "yesterday":
        return date.fromordinal(today.toordinal() - 1)

    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", normalized)
    us_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", normalized)
    named_match = re.fullmatch(
        rf"({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        r"(?:,?\s+(\d{4}))?",
        normalized,
    )
    month_match = re.fullmatch(
        rf"({'|'.join(MONTHS)})(?:\s+(\d{{4}}))?",
        normalized,
    )
    try:
        if iso_match:
            return date(*(int(part) for part in iso_match.groups()))
        if us_match:
            month, day, year = (int(part) for part in us_match.groups())
            return date(year, month, day)
        if named_match:
            month_name, day, year = named_match.groups()
            return date(int(year or today.year), MONTHS[month_name], int(day))
        if month_match:
            month_name, year = month_match.groups()
            resolved_year = int(year or today.year)
            month = MONTHS[month_name]
            day = 1 if boundary == "start" else monthrange(resolved_year, month)[1]
            return date(resolved_year, month, day)
    except ValueError:
        return None
    return None


def _contains_date_marker(text: str) -> bool:
    """Identify date-like text so malformed dates do not silently lose filters."""
    return bool(
        re.search(
            r"\b(today|yesterday|this month|last month|last \d+ days?|"
            r"past \d+ days?|january|february|march|april|may|june|july|"
            r"august|september|october|november|december)\b",
            text,
        )
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b", text)
    )


def _view_search_text(text: str, today: date) -> str | None:
    """Extract a plain index query after `for`, excluding a trailing date phrase."""
    match = re.search(r"\bfor\s+(.+)$", text)
    if not match:
        return None
    candidate = match.group(1).strip(" .?")
    candidate = re.split(
        r"\s+(?:on|in|during|from|between)\s+(?=(?:today|yesterday|this|last|past|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}))",
        candidate,
        maxsplit=1,
    )[0]
    if _safe_date_range(candidate, today):
        return None
    return candidate.strip() or None


def _entry_values(
    text: str,
    entry_type: str,
    today: date,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Extract explicitly labelled scalar values and a single entry date."""
    values: dict[str, Any] = {}
    consumed = []
    date_range = _safe_date_range(text, today)
    if date_range and date_range[0] == date_range[1]:
        values["time" if entry_type == "run_ticket" else "date"] = date_range[0].isoformat()
    for field in ENTRY_FIELDS.get(entry_type, ()):
        for label in _field_labels(field):
            match = re.search(
                rf"\b{label}\s*(?:is|=|:)?\s*({_number_value_pattern()})\b",
                text,
            )
            if match:
                if _span_is_consumed(match.span(1), consumed):
                    continue
                values[field] = _parse_number_value(match.group(1))
                consumed.append(match.span(1))
                break
    comments = re.search(r"\bcomments?\s*(?:is|=|:)?\s*[\"']([^\"']+)[\"']", text)
    if not comments:
        comments = re.search(r"\bcomments?\s*(?:is|=|:)?\s+(.+)$", text)
    if comments:
        values["comments"] = comments.group(1).strip()
        consumed.extend(_numeric_spans_in_group(comments, 1))
    subject = re.search(r"\bsubject\s*(?:is|=|:)?\s*[\"']([^\"']+)[\"']", text)
    if subject and entry_type == "work_order":
        values["subject"] = subject.group(1).strip()
        consumed.extend(_numeric_spans_in_group(subject, 1))
    return values, consumed


def _numeric_spans_in_group(
    match: re.Match[str],
    group: int,
) -> list[tuple[int, int]]:
    """Return absolute spans for numbers inside an explicitly captured text value."""
    offset = match.start(group)
    return [
        (offset + item.start(), offset + item.end())
        for item in re.finditer(r"[-+]?\d+(?:\.\d+)?", match.group(group))
    ]


def _entry_entity_name(text: str, entry_type: str, today: date) -> str | None:
    """Extract only an entity introduced by an explicit `for` preposition."""
    if entry_type in {"general_note", "work_order"}:
        return None
    match = re.search(r"\bfor\s+(.+)$", text)
    if not match:
        return None
    candidate = match.group(1).strip(" .?")
    candidate = re.split(r"\s+with\s+", candidate, maxsplit=1)[0]
    if entry_type in ENTRY_FIELDS:
        candidate = _strip_entry_field_tail(candidate, entry_type)
    candidate = re.split(
        r"\s+(?:on|at)\s+(?=(?:today|yesterday|\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}/\d{1,2}/\d{4}))",
        candidate,
        maxsplit=1,
    )[0]
    return None if _safe_date_range(candidate, today) else candidate.strip() or None


def _span_is_consumed(
    span: tuple[int, int], consumed: list[tuple[int, int]]
) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in consumed)


def _number_is_part_of_date(text: str, span: tuple[int, int]) -> bool:
    context = text[max(0, span[0] - 20) : min(len(text), span[1] + 20)]
    return bool(
        re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", context)
        or re.search(r"\b(?:last|past)\s+\d+\s+days?\b", context)
        or re.search(
            rf"\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}}(?:st|nd|rd|th)?"
            r"(?:,?\s+\d{4})?\b",
            context,
        )
    )


def _field_labels(field: str) -> tuple[str, ...]:
    """Return regex labels for a form field, including voice-dictation aliases."""
    aliases = FIELD_ALIASES.get(field, (field.replace("_", " "),))
    return tuple(re.escape(alias).replace(r"\ ", r"[ _-]+") for alias in aliases)


def _number_value_pattern() -> str:
    """Match either a digit number or the small spoken numbers operators dictate."""
    words = "|".join(sorted((re.escape(item) for item in SPOKEN_NUMBERS), key=len, reverse=True))
    return rf"[-+]?\d+(?:\.\d+)?|{words}"


def _parse_number_value(value: str) -> float:
    """Convert a numeric token into the float shape already used by this router."""
    normalized = value.strip().lower()
    if normalized in SPOKEN_NUMBERS:
        return float(SPOKEN_NUMBERS[normalized])
    return float(normalized)


def _strip_entry_field_tail(candidate: str, entry_type: str) -> str:
    """Remove measurement text accidentally captured as part of an entity name."""
    labels = [
        label
        for field in ENTRY_FIELDS.get(entry_type, ())
        for label in _field_labels(field)
    ]
    if not labels:
        return candidate
    match = re.search(
        rf"\b(?:{'|'.join(labels)})\s*(?:is|=|:)?\s*({_number_value_pattern()})\b",
        candidate,
    )
    return candidate[:match.start()].strip(" .?") if match else candidate


def _format_navigation_result(result: dict[str, Any], tool_name: str) -> str:
    """Convert validated client statuses into concise user-facing responses."""
    if tool_name == "search_ometrics_capabilities":
        return _format_capability_result(result)
    status = result.get("status")
    if status == "ready":
        if tool_name == "prepare_data_entry":
            return "Opening the prefilled data-entry form."
        return "Opening the requested page."
    message = str(result.get("message") or "The request could not be prepared.")
    candidates = result.get("candidates") or []
    if candidates:
        labels = []
        for item in candidates:
            label = (
                f"{item.get('type')} - {item.get('name')}"
                if item.get("type")
                else str(item.get("name"))
            )
            labels.append(label)
        return f"{message} Candidates: {', '.join(labels)}"
    return message


def _format_capability_result(result: dict[str, Any]) -> str:
    if not result.get("ok", True):
        return str(result.get("error") or "I could not search Ometrics capability guidance.")
    matches = result.get("matches") or []
    if not matches:
        return "I could not find matching Ometrics capability guidance for that workflow."
    first = matches[0]
    capability = str(first.get("capability") or "Ometrics workflow")
    summary = str(first.get("summary") or "").strip()
    query = str(result.get("query") or "")
    workflow_answer = _workflow_answer_from_capability_summary(summary, query)
    if workflow_answer:
        return workflow_answer
    if not summary:
        return capability
    return f"{capability}: {summary}"


def _workflow_answer_from_capability_summary(summary: str, query: str) -> str | None:
    """Return curated workflow wording embedded in capability guidance."""
    quoted_actions = re.findall(r'"([^"]+)"', summary)
    normalized_query = _normalize(query)
    query_terms = [
        term for term in _workflow_entity_terms()
        if re.search(rf"\b{re.escape(term)}s?\b", normalized_query)
    ]
    if query_terms:
        for action in quoted_actions:
            normalized_action = _normalize(action)
            if any(term in normalized_action for term in query_terms):
                return action.strip()
    for action in quoted_actions:
        if re.search(r"\bcreate\b|\buse\b", action, re.IGNORECASE):
            return action.strip()
    return None


def _workflow_entity_terms() -> tuple[str, ...]:
    return (
        "lact",
        "flare",
        "tank",
        "flow meter",
        "pump",
        "treater",
        "knock out",
        "knockout",
        "water plant",
        "well test",
        "well fluid",
        "fluid level",
        "well injection",
        "run ticket",
        "water draw",
        "general note",
        "work order",
        "shutdown",
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().rstrip("?.").split())
