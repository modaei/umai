from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.rag.vector_store import (
    OperationalContextStoreError,
    VectorOperationalContextStore,
)


logger = logging.getLogger(__name__)


class SearchOperationalContextInput(BaseModel):
    """Schema exposed to the LLM for operational-context retrieval.

    The current selected site is injected server-side and is not accepted from
    the model. That prevents cross-site search from a malformed or hallucinated
    tool call.
    """

    query: str = Field(description="Question about operational notes, comments, alarms, or history.")
    start_date: str | None = Field(
        default=None,
        description="Optional start date in YYYY-MM-DD format.",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional end date in YYYY-MM-DD format.",
    )
    entity_name: str | None = Field(
        default=None,
        description="Optional well or equipment name mentioned by the user.",
    )
    source_types: list[str] | None = Field(
        default=None,
        description=(
            "Optional source types. Only set this when the user explicitly asks "
            "for a specific record type. Use work_order for work order records "
            "(subject, status, vendor, comments). Use work_order_note only for "
            "follow-up notes attached to work orders. Other examples: "
            "general_note, chart_note, well_shutdown, alarm_log."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _parse_tool_date(value: str | None) -> date | None:
    if not value:
        return None

    stripped = value.strip()
    if len(stripped) == 8 and stripped.isdigit():
        return date(
            int(stripped[0:4]),
            int(stripped[4:6]),
            int(stripped[6:8]),
        )
    return date.fromisoformat(stripped)


def _normalize_source_types(
    query: str,
    source_types: list[str] | None,
) -> list[str] | None:
    """Keep source filters only for explicit record-type requests.

    The LLM sometimes picks a narrow source list even when the user asked a
    broad operational-context question. That can hide valid records, such as
    chart notes for paraffin/hot-water questions. Source filtering is therefore
    treated as an optimization only when the query text explicitly names a
    record type.
    """
    if not source_types:
        return source_types

    # Keep the model's order, but remove duplicates before applying safeguards.
    normalized = list(dict.fromkeys(source_types))
    if not _is_explicit_source_type_query(query):
        return None

    if "work_order_note" not in normalized or "work_order" in normalized:
        return normalized
    if not _is_general_work_order_query(query):
        return normalized

    # General work-order requests need work_order records. work_order_note only
    # covers follow-up notes and can produce false "nothing found" answers.
    return [*normalized, "work_order"]


def _is_explicit_source_type_query(query: str) -> bool:
    """Return True when the query names the kind of operational record to search."""
    normalized = " ".join(query.lower().replace("-", " ").replace("_", " ").split())
    source_type_phrases = {
        "general note",
        "general notes",
        "chart note",
        "chart notes",
        "work order",
        "work orders",
        "work order note",
        "work order notes",
        "shutdown",
        "shutdowns",
        "well shutdown",
        "well shutdowns",
        "shutdown",
        "shutdowns",
        "alarm log",
        "alarm logs",
        "alarm event",
        "alarm events",
        "well history",
        "history records",
        "well test comment",
        "well test comments",
        "reading comment",
        "reading comments",
    }
    return any(phrase in normalized for phrase in source_type_phrases)


def _is_general_work_order_query(query: str) -> bool:
    normalized = " ".join(query.lower().replace("-", " ").split())
    if not ("work order" in normalized or "work orders" in normalized):
        return False

    note_phrases = {
        "work order note",
        "work order notes",
        "notes on work order",
        "notes on work orders",
        "follow up note",
        "follow up notes",
        "work order update",
        "work order updates",
    }
    return not any(phrase in normalized for phrase in note_phrases)


def build_operational_context_tools(
    store: VectorOperationalContextStore,
    site_id: int,
) -> list[StructuredTool]:
    """Build the LangChain tool around the vector operational-context store."""

    def search_operational_context(
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        entity_name: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 8,
    ) -> str:
        """Search indexed Ometrics notes, comments, alarms, work, shutdowns, and history."""
        logger.info(
            "Searching operational context site_id=%s query=%s start=%s end=%s entity=%s",
            site_id,
            query,
            start_date,
            end_date,
            entity_name,
        )
        try:
            # Convert date strings here so the store receives typed date values.
            # Bad dates become structured tool errors instead of uncaught crashes.
            return _json_result(
                {
                    "ok": True,
                    **store.search(
                        query=query,
                        site_id=site_id,
                        start_date=_parse_tool_date(start_date),
                        end_date=_parse_tool_date(end_date),
                        entity_name=entity_name,
                        source_types=_normalize_source_types(query, source_types),
                        limit=limit,
                    ),
                }
            )
        except (OperationalContextStoreError, ValueError) as exc:
            # Tools return JSON errors instead of raising so the LLM can explain
            # that operational context is unavailable or the date input is invalid.
            logger.warning("Operational context search failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    return [
        StructuredTool.from_function(
            func=search_operational_context,
            name="search_operational_context",
            description=(
                "Search indexed Ometrics operational text from the vector DB: general notes, "
                "chart notes, work orders and notes, shutdown comments, downtime "
                "codes, well-test comments, reading comments, alarm logs, and well "
                "history. Use for questions asking what happened, why something "
                "may have happened, summaries of notes/comments, work history, "
                "alarm context, or records mentioning a condition. Summarize only "
                "returned records."
            ),
            args_schema=SearchOperationalContextInput,
        )
    ]
