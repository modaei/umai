from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from umai.rag.vector_store import VectorOperationalContextStore
from umai.rag.extractors.operational_text import OperationalTextExtractor


@dataclass(frozen=True)
class IndexResult:
    """Small command/reporting DTO returned after one indexing run."""

    documents: int
    chunks: int
    deleted_chunks: int = 0


class OperationalContextIndexer:
    """Coordinates extraction from MySQL and upsert into the vector DB.

    Keeping this orchestration separate from both the extractor and store makes
    the boundary explicit: MySQL is read, the vector DB is written. There is no
    MySQL ledger or RAG table in this implementation.
    """

    def __init__(
        self,
        extractor: OperationalTextExtractor,
        store: VectorOperationalContextStore,
    ):
        self.extractor = extractor
        self.store = store

    def index(
        self,
        site_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
        source_types: set[str] | None = None,
    ) -> IndexResult:
        # Extract first so source-specific SQL stays isolated from vector-store
        # code. This also makes tests easy with an in-memory SQL database.
        documents = self.extractor.extract(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            source_types=source_types,
        )
        # Vector-store upsert handles chunking and embedding. Stable chunk IDs make
        # repeated indexing idempotent for the same source rows.
        chunks = self.store.upsert_documents(documents)
        return IndexResult(documents=len(documents), chunks=chunks)

    def index_source(
        self,
        site_id: int,
        source_type: str,
        source_id: str,
    ) -> IndexResult:
        deleted_chunks = self.store.delete_source(site_id, source_type, source_id)
        document = self.extractor.extract_one(site_id, source_type, source_id)
        documents = [document] if document is not None else []
        chunks = self.store.upsert_documents(documents)
        return IndexResult(
            documents=len(documents),
            chunks=chunks,
            deleted_chunks=deleted_chunks,
        )

    def delete_source(
        self,
        site_id: int,
        source_type: str,
        source_id: str,
    ) -> IndexResult:
        deleted_chunks = self.store.delete_source(site_id, source_type, source_id)
        return IndexResult(documents=0, chunks=0, deleted_chunks=deleted_chunks)
