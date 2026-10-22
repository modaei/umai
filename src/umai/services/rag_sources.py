"""Extract concise RAG-source audit records from operational-context tool calls."""

from __future__ import annotations

import json
from typing import Any


MAX_RAG_SOURCES = 10
RAG_TOOL_NAME = "search_operational_context"
SOURCE_PREVIEW_LENGTH = 240


def extract_rag_sources(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deduplicated operational-context sources for internal message audit."""
    sources: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for call in tool_calls:
        if call.get("tool") != RAG_TOOL_NAME:
            continue
        payload = _parse_result(call.get("result"))
        if not payload.get("ok"):
            continue
        matches = payload.get("matches")
        if not isinstance(matches, list):
            continue
        for match in matches:
            if not isinstance(match, dict):
                continue
            chunk_id = str(match.get("chunk_id") or "")
            dedupe_key = chunk_id or json.dumps(match, sort_keys=True, default=str)
            if dedupe_key in seen_chunk_ids:
                continue
            seen_chunk_ids.add(dedupe_key)
            sources.append(_source_from_match(match))
            if len(sources) >= MAX_RAG_SOURCES:
                return sources

    return sources


def format_rag_source(source: dict[str, Any]) -> str:
    """Format one source for internal diagnostics or future presentation clients."""
    parts = [
        str(source.get("source_type") or "operational_context"),
        str(source.get("event_date") or "no date"),
    ]
    entity_name = source.get("entity_name")
    if entity_name:
        parts.append(str(entity_name))
    header = " - ".join(parts)
    text = source.get("text")
    if text:
        return f"**{header}**\n\n{text}"
    return f"**{header}**"


def _parse_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return {}
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_from_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": match.get("chunk_id"),
        "source_type": match.get("source_type"),
        "event_date": match.get("event_date"),
        "entity_name": match.get("entity_name"),
        "text": _preview(str(match.get("text") or "")),
    }


def _preview(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= SOURCE_PREVIEW_LENGTH:
        return text
    return text[: SOURCE_PREVIEW_LENGTH - 3].rstrip() + "..."
