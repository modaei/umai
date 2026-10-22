from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

from umai.config.settings import Settings


@dataclass(frozen=True)
class EntryDefinition:
    """Describe one supported form, its owning entity, and duplicate key."""

    entity_type: str | None
    entity_table: str | None
    entity_column: str | None
    record_table: str
    date_column: str | None
    fields: frozenset[str]


ENTRY_DEFINITIONS: dict[str, EntryDefinition] = {
    "flare_reading": EntryDefinition("flare", "flares", "flare_id", "flare_readings", "time", frozenset({"date", "pressure", "volume", "comments"})),
    "flow_meter_reading": EntryDefinition("flow meter", "flow_meters", "flow_meter_id", "flow_meter_readings", "time", frozenset({"date", "total", "flow", "odometer", "comments"})),
    "pump_reading": EntryDefinition("pump", "pumps", "pump_id", "pump_readings", "time", frozenset({"date", "suction_pressure", "discharge_pressure", "comments"})),
    "treater_reading": EntryDefinition("treater", "treaters", "treater_id", "treater_readings", "time", frozenset({"date", "oil_intake", "pressure", "temperature", "comments"})),
    "knockout_reading": EntryDefinition("knockout", "knock_outs", "knock_out_id", "knock_out_readings", "time", frozenset({"date", "inlet", "oil_off", "comments"})),
    "water_plant_reading": EntryDefinition("water plant", "water_plants", "water_plant_id", "water_plant_readings", "time", frozenset({"date", "flow_rate", "suction_pressure", "discharge_pressure", "comments"})),
    "lact_reading": EntryDefinition("LACT", "lacts", "lact_id", "lact_readings", "time", frozenset({"date", "reading", "temperature", "bs_w", "comments"})),
    "tank_reading": EntryDefinition("tank", "tanks", "tank_id", "tank_readings", "time", frozenset({"date", "level", "feet", "inches", "temperature", "top_level_feet", "top_level_inches", "water_level_feet", "water_level_inches", "initial_feet", "initial_inches", "final_feet", "final_inches", "comments"})),
    "well_test": EntryDefinition("well", "wells", "well_id", "well_tests", "time", frozenset({"date", "oil", "water", "gas", "pip", "m_temp", "amps", "tbgp", "csgp", "fluid_level", "runtime", "comments"})),
    "well_fluid": EntryDefinition("well", "wells", "well_id", "well_fluids", "time", frozenset({"date", "level", "comments"})),
    "well_injection": EntryDefinition("well", "wells", "well_id", "well_injections", "time", frozenset({"date", "flow_rate", "total", "tbg", "csg", "type", "comments"})),
    "well_shutdown": EntryDefinition("well", "wells", "well_id", "well_shutdowns", "start", frozenset({"start", "end", "downtime_code", "comments"})),
    "water_draw": EntryDefinition("tank", "tanks", "tank_id", "water_draws", "time", frozenset({"date", "initial_feet", "initial_inches", "final_feet", "final_inches", "comments"})),
    "run_ticket": EntryDefinition("tank", "tanks", "tank_id", "run_tickets", "time", frozenset({"time", "number", "measurement_method", "obs_grav", "obs_temp", "b_s_w", "corr_grav", "gov", "nsv", "initial_feet", "initial_inches", "initial_qtr", "final_feet", "final_inches", "final_qtr", "initial_volume", "final_volume", "initial_meter_reading", "final_meter_reading", "comments"})),
    "general_note": EntryDefinition(None, None, None, "general_notes", "date", frozenset({"date", "comments"})),
    "work_order": EntryDefinition(None, None, None, "work_orders", None, frozenset({"date", "subject", "cost_estimate", "final_cost", "priority", "status", "assigned_user", "vendor", "comments"})),
}

GENERIC_READING_TYPES = {
    "flare_reading", "flow_meter_reading", "pump_reading", "treater_reading",
    "knockout_reading", "water_plant_reading", "lact_reading", "tank_reading",
}

SPOKEN_NUMBERS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


class DataEntryClient:
    """Resolve and validate form-navigation requests without writing records."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataEntryClient":
        """Build a bounded, process-safe connection pool from application settings."""
        settings.validate_database()
        return cls(create_engine(
            URL.create(
                "mysql+pymysql", username=settings.db_user,
                password=settings.db_password, host=settings.db_host,
                port=settings.db_port, database=settings.db_name,
            ),
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        ))

    def prepare(self, site_id: int, entry_type: str, entity_name: str | None,
                values: dict[str, Any], current_date: str) -> dict[str, Any]:
        """Return a ready intent or a structured reason navigation must stop."""
        if entry_type == "generic_reading":
            return self._prepare_generic_reading(
                site_id, entity_name, values, current_date
            )
        definition = ENTRY_DEFINITIONS.get(entry_type)
        if definition is None:
            return {"status": "unsupported", "message": f"Unsupported data-entry type: {entry_type}."}
        unknown = sorted(set(values) - definition.fields)
        if unknown:
            return {"status": "invalid", "message": f"Unsupported fields for {entry_type}: {', '.join(unknown)}."}
        values = {key: value for key, value in values.items() if value is not None}
        if entry_type == "work_order" and values.get("assigned_user"):
            user_result = self._resolve_user(site_id, str(values.pop("assigned_user")))
            if user_result.get("status") != "resolved":
                return user_result
            values["assigned_user_id"] = user_result["entity"]["id"]
        date_key = "time" if entry_type == "run_ticket" else "date"
        if date_key in definition.fields and not values.get(date_key):
            values[date_key] = current_date

        entity = None
        if definition.entity_table:
            if not entity_name:
                return {"status": "needs_clarification", "message": f"Which {definition.entity_type} should this use?", "candidates": []}
            entity_result = self._resolve_entity(site_id, definition, entity_name)
            if entity_result.get("status") != "resolved":
                return entity_result
            entity = entity_result["entity"]

        duplicate = self._find_duplicate(site_id, entry_type, definition, entity, values)
        if duplicate:
            return {
                "status": "duplicate",
                "message": duplicate,
                "entry_type": entry_type,
            }
        return {
            "status": "ready",
            "entry_type": entry_type,
            "entity_type": definition.entity_type,
            "entity_id": entity["id"] if entity else None,
            "entity_name": entity["name"] if entity else None,
            "values": values,
        }

    def _prepare_generic_reading(self, site_id: int, entity_name: str | None,
                                 values: dict[str, Any], current_date: str) -> dict[str, Any]:
        """Resolve an untyped reading name across every reading-enabled entity type."""
        if not entity_name:
            return {"status": "needs_clarification", "message": "Which equipment should the reading use?", "candidates": []}
        needle = _normalize(entity_name)
        catalog = []
        for entry_type in GENERIC_READING_TYPES:
            definition = ENTRY_DEFINITIONS[entry_type]
            extra = ", type AS tank_type" if definition.entity_table == "tanks" else ""
            with self.engine.connect() as connection:
                rows = [dict(row) for row in connection.execute(
                    text(f"SELECT id, name{extra} FROM {definition.entity_table} WHERE site_id=:site_id"),
                    {"site_id": site_id},
                ).mappings()]
            catalog.extend((entry_type, definition, row) for row in rows)
        matches = [
            item for item in catalog
            if _entity_match_key(needle, item[1].entity_type) == _normalize(item[2]["name"])
        ]
        if not matches:
            matches = [
                item for item in catalog
                if _token_sequence_match(
                    _entity_match_key(needle, item[1].entity_type),
                    _normalize(item[2]["name"]),
                )
            ]
        if not matches:
            matches = [
                item for item in catalog
                if _fuzzy_match(needle, _normalize(item[2]["name"]))
            ]
        if len(matches) == 1:
            entry_type, _, entity = matches[0]
            return self.prepare(site_id, entry_type, entity["name"], values, current_date)
        candidates = [
            {"type": definition.entity_type, "name": entity["name"]}
            for _, definition, entity in matches[:10]
        ]
        message = (f"Multiple entities matched '{entity_name}'. Which one did you mean?"
                   if matches else f"No reading-enabled entity matched '{entity_name}'.")
        return {"status": "needs_clarification", "message": message, "candidates": candidates}

    def _resolve_entity(self, site_id: int, definition: EntryDefinition, name: str) -> dict[str, Any]:
        """Prefer an exact site-scoped name, then accept only an unambiguous fuzzy match."""
        extra = ", type AS tank_type" if definition.entity_table == "tanks" else ""
        query = text(f"SELECT id, name{extra} FROM {definition.entity_table} WHERE site_id = :site_id")
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(query, {"site_id": site_id}).mappings()]
        needle = _normalize(name)
        match_key = _entity_match_key(needle, definition.entity_type)
        exact = [row for row in rows if _normalize(row["name"]) == match_key]
        token_exact = exact or [
            row for row in rows
            if _token_sequence_match(match_key, _normalize(row["name"]))
        ]
        matches = token_exact or [
            row for row in rows if _fuzzy_match(needle, _normalize(row["name"]))
        ]
        if len(matches) == 1:
            return {"status": "resolved", "entity": matches[0]}
        candidates = [{"type": definition.entity_type, "name": row["name"]} for row in matches[:10]]
        if not matches:
            return {"status": "needs_clarification", "message": f"No {definition.entity_type} matched '{name}'.", "candidates": []}
        return {"status": "needs_clarification", "message": f"Multiple {definition.entity_type} records matched '{name}'. Which one did you mean?", "candidates": candidates}

    def _resolve_user(self, site_id: int, name: str) -> dict[str, Any]:
        """Resolve a work-order assignee from users attached to the selected site."""
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(text(
                "SELECT users.id, users.name FROM users JOIN site_user ON site_user.user_id=users.id WHERE site_user.site_id=:site_id"
            ), {"site_id": site_id}).mappings()]
        needle = _normalize(name)
        exact = [row for row in rows if _normalize(row["name"]) == needle]
        matches = exact or [row for row in rows if _fuzzy_match(needle, _normalize(row["name"]))]
        if len(matches) == 1:
            return {"status": "resolved", "entity": matches[0]}
        return {
            "status": "needs_clarification",
            "message": f"Which assigned user did you mean by '{name}'?",
            "candidates": [{"type": "user", "name": row["name"]} for row in matches[:10]],
        }

    def _find_duplicate(self, site_id: int, entry_type: str, definition: EntryDefinition,
                        entity: dict[str, Any] | None, values: dict[str, Any]) -> str | None:
        """Apply the destination form's uniqueness rule before proposing navigation."""
        if entry_type == "work_order" and values.get("subject"):
            sql = "SELECT 1 FROM work_orders WHERE site_id=:site_id AND subject=:subject LIMIT 1"
            params = {"site_id": site_id, "subject": values["subject"]}
        elif entry_type == "general_note":
            sql = "SELECT 1 FROM general_notes WHERE site_id=:site_id AND date=:date LIMIT 1"
            params = {"site_id": site_id, "date": values.get("date")}
        elif definition.entity_column and definition.date_column and entity:
            value_key = "time" if entry_type == "run_ticket" else definition.date_column
            if value_key not in values:
                value_key = "date"
            date_value = values.get(value_key)
            if not date_value:
                return None
            record_table = definition.record_table
            date_column = definition.date_column
            if entry_type == "tank_reading":
                record_table = {
                    "mixed-water-oil": "mixed_tank_readings",
                    "non-linear-volume": "non_linear_tank_readings",
                }.get(entity.get("tank_type"), "linear_tank_readings")
            sql = f"SELECT 1 FROM {record_table} WHERE {definition.entity_column}=:entity_id AND DATE({date_column})=DATE(:entry_date) LIMIT 1"
            params = {"entity_id": entity["id"], "entry_date": date_value}
        else:
            return None
        with self.engine.connect() as connection:
            exists = connection.execute(text(sql), params).first() is not None
        return "An entry already exists for the intended record and date. No form was opened." if exists else None


def _normalize(value: str) -> str:
    """Normalize display names without exposing database-specific matching rules."""
    return " ".join(
        SPOKEN_NUMBERS.get(token, token)
        for token in re.findall(r"[a-z0-9]+", value.lower())
    )


def _fuzzy_match(needle: str, candidate: str) -> bool:
    """Recognize strong partial/token matches while leaving ambiguity to the user."""
    if not needle or not candidate:
        return False
    needle_tokens, candidate_tokens = set(needle.split()), set(candidate.split())
    return (needle in candidate or candidate in needle or needle_tokens <= candidate_tokens
            or SequenceMatcher(None, needle, candidate).ratio() >= 0.80)


def _token_sequence_match(needle: str, candidate: str) -> bool:
    """Match adjacent full tokens so tank 5-2 does not match tank 15-2."""
    needle_tokens = needle.split()
    candidate_tokens = candidate.split()
    if not needle_tokens or len(needle_tokens) > len(candidate_tokens):
        return False
    return any(
        candidate_tokens[index:index + len(needle_tokens)] == needle_tokens
        for index in range(len(candidate_tokens) - len(needle_tokens) + 1)
    )


def _entity_match_key(needle: str, entity_type: str | None) -> str:
    """Drop a spoken entity-type prefix when database names omit it."""
    if not entity_type:
        return needle
    entity_tokens = entity_type.lower().split()
    needle_tokens = needle.split()
    if needle_tokens[:len(entity_tokens)] == entity_tokens:
        return " ".join(needle_tokens[len(entity_tokens):])
    return needle
