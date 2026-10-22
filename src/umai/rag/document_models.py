from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class RagDocument:
    """Normalized operational-text record before chunking and embedding.

    Source tables in Ometrics have very different shapes: notes have comments,
    shutdowns have downtime fields, alarms have event strings, work orders have
    subjects and notes, and readings attach comments to different equipment
    tables. The RAG pipeline should not make the vector store know about all of
    those table-specific schemas, so every extractor converts rows into this
    common representation first.

    MySQL remains the source of truth. This object is only an indexing payload
    used to create vector DB records; no RAG metadata is written back to MySQL.
    """

    source_type: str
    source_id: str
    site_id: int
    text: str
    event_date: date | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    entity_name: str | None = None
    source_updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_id(self) -> str:
        # This ID must be deterministic so re-indexing the same source row upserts
        # the same vector DB record instead of creating duplicates.
        return f"{self.source_type}:{self.source_id}"


@dataclass(frozen=True)
class RagChunk:
    """A searchable piece of a RagDocument.

    Most Ometrics comments are short and become one chunk. Longer records, such
    as work-order notes or history descriptions, may split into multiple chunks.
    Each chunk carries the parent document metadata so the vector DB can filter
    by site, date, source type, and entity before/while running semantic search.
    """

    chunk_id: str
    document: RagDocument
    chunk_index: int
    text: str

    @property
    def metadata(self) -> dict[str, str | int | float | bool]:
        # Vector metadata values should stay scalar JSON-like values. Keep rich
        # Python objects out of metadata and serialize dates/datetimes explicitly.
        metadata: dict[str, str | int | float | bool] = {
            "source_type": self.document.source_type,
            "source_id": self.document.source_id,
            "site_id": self.document.site_id,
            "chunk_index": self.chunk_index,
        }
        if self.document.event_date is not None:
            # Keep both date forms:
            # - event_date is readable and is returned to the assistant/user.
            # - event_date_key is numeric YYYYMMDD for databases that prefer
            #   integer date-range filtering.
            metadata["event_date"] = self.document.event_date.isoformat()
            metadata["event_date_key"] = int(self.document.event_date.strftime("%Y%m%d"))
        if self.document.entity_type:
            metadata["entity_type"] = self.document.entity_type
        if self.document.entity_id:
            metadata["entity_id"] = self.document.entity_id
        if self.document.entity_name:
            metadata["entity_name"] = self.document.entity_name
        if self.document.source_updated_at is not None:
            metadata["source_updated_at"] = self.document.source_updated_at.isoformat()
        for key, value in self.document.metadata.items():
            # Only pass through metadata types that serialize reliably.
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata


def chunk_document(
    document: RagDocument,
    max_chars: int = 3_200,
    overlap_chars: int = 300,
) -> list[RagChunk]:
    """Split a document into deterministic, overlapping text chunks.

    Character-based chunking is intentional here. Operational text records are
    usually short comments, so adding a tokenization dependency would add more
    complexity than value. The overlap preserves context when a long note is cut
    near the phrase a user later searches for.
    """

    # Normalize whitespace so semantically identical text produces stable chunk
    # boundaries across indexing runs.
    text = " ".join(document.text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        # Preserve a chunk suffix even for single-chunk documents. That keeps the
        # Chunk ID format consistent: source_type:source_id:chunk:n.
        return [
            RagChunk(
                chunk_id=f"{document.stable_id}:chunk:0",
                document=document,
                chunk_index=0,
                text=text,
            )
        ]

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer ending at a word boundary. Avoid tiny chunks by accepting the
            # boundary only if it is reasonably far into the requested window.
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                RagChunk(
                    chunk_id=f"{document.stable_id}:chunk:{chunk_index}",
                    document=document,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1
        if end >= len(text):
            break
        # Move back by overlap_chars so adjacent chunks share context.
        start = max(0, end - overlap_chars)
    return chunks
