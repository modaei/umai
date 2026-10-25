from __future__ import annotations

import logging
from datetime import date
from typing import Any, Protocol

from umai.rag.vector_store import OperationalContextStoreError


logger = logging.getLogger(__name__)


class OperationalContextSearchStore(Protocol):
    """Small interface shared by tools that can add operational RAG context."""

    def search(
        self,
        query: str,
        site_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
        entity_name: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        ...


def add_operational_context(
    result: dict[str, Any],
    store: OperationalContextSearchStore | None,
    *,
    site_id: int,
    query: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    entity_name: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Attach best-effort RAG context to an existing structured tool result.

    Structured tools remain authoritative for numeric totals, readings, and
    exact records. RAG context is supplementary and is omitted if the vector
    store is unavailable or if the date inputs cannot be parsed.
    """
    if store is None:
        return result

    try:
        context = store.search(
            query=query,
            site_id=site_id,
            start_date=_parse_optional_date(start_date),
            end_date=_parse_optional_date(end_date),
            entity_name=entity_name,
            source_types=None,
            limit=limit,
        )
    except (OperationalContextStoreError, ValueError) as exc:
        logger.warning("Operational context enrichment failed: %s", exc)
        return result

    return {
        **result,
        "operational_context": context,
    }


def _parse_optional_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)
