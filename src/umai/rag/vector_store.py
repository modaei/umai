from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain_openai import OpenAIEmbeddings
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    Computed,
    Date,
    DateTime,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    bindparam,
    create_engine,
    delete,
    func,
    literal,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, insert
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings
from umai.rag.document_models import RagDocument, chunk_document


DEFAULT_UPSERT_BATCH_SIZE = 200
HYBRID_CANDIDATE_MINIMUM = 50
HYBRID_CANDIDATE_MULTIPLIER = 5
RRF_K = 60


class OperationalContextStoreError(RuntimeError):
    """Raised when vector operational context cannot be indexed or searched."""


class VectorOperationalContextStore:
    """Postgres/pgvector-backed vector index for operational text.

    The public naming is intentionally provider-neutral because the rest of UMAI
    only needs "a vector DB". The first implementation uses Postgres + pgvector
    behind that interface. MySQL/MariaDB remains the source of truth; this store
    contains only derived chunks and embeddings that can be rebuilt.
    """

    def __init__(
        self,
        engine: Engine,
        embeddings: OpenAIEmbeddings,
        embedding_dimensions: int,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ):
        self.engine = engine
        self.embeddings = embeddings
        self.embedding_dimensions = embedding_dimensions
        self.upsert_batch_size = upsert_batch_size
        self.metadata = MetaData()
        self.chunks = _rag_chunks_table(self.metadata, embedding_dimensions)

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorOperationalContextStore":
        embeddings = OpenAIEmbeddings(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.rag_embedding_model,
        )
        return cls(
            engine=_engine_from_settings(settings),
            embeddings=embeddings,
            embedding_dimensions=settings.rag_embedding_dimensions,
        )

    def migrate(self) -> None:
        """Create table and indexes if pgvector is already installed.

        Creating PostgreSQL extensions usually requires superuser or elevated
        database privileges, so application migrations deliberately do not run
        CREATE EXTENSION. A DBA should run `CREATE EXTENSION vector` once for the
        configured vector database before this migration is executed.
        """

        try:
            with self.engine.begin() as connection:
                extension_status = _pgvector_extension_status(connection)
                if not extension_status["installed"]:
                    raise OperationalContextStoreError(
                        "pgvector extension is not installed in the connected "
                        "database. Run `CREATE EXTENSION vector` once with a "
                        "privileged Postgres user in this exact database, then "
                        f"rerun this migration. Connection: database="
                        f"{extension_status['database']}, user="
                        f"{extension_status['user']}, server="
                        f"{extension_status['server_addr']}:"
                        f"{extension_status['server_port']}."
                    )
                if not extension_status["type_visible"]:
                    raise OperationalContextStoreError(
                        "pgvector extension is installed, but the `vector` type "
                        "is not visible on the app user's search_path. Install "
                        "the extension in a schema on search_path, or update the "
                        f"database/user search_path. Extension schema: "
                        f"{extension_status['extension_schema']}. Connection: "
                        f"database={extension_status['database']}, user="
                        f"{extension_status['user']}."
                    )
                self.metadata.create_all(connection)
                connection.execute(
                    text(
                        """
                        ALTER TABLE rag_chunks
                        ADD COLUMN IF NOT EXISTS text_search_vector tsvector
                        GENERATED ALWAYS AS (
                            to_tsvector('english'::regconfig, COALESCE(text, ''))
                        ) STORED
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
                        ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS idx_rag_chunks_text_search
                        ON rag_chunks USING gin (text_search_vector)
                        """
                    )
                )
        except SQLAlchemyError as exc:
            raise OperationalContextStoreError(
                f"Could not migrate vector DB schema: {exc}"
            ) from exc

    def upsert_documents(self, documents: list[RagDocument]) -> int:
        """Embed and upsert normalized operational documents into pgvector."""

        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document)
            if chunk.text.strip()
        ]
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        rows = [
            {
                **_chunk_row(chunk.metadata),
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": _extra_metadata(chunk.metadata),
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            with self.engine.begin() as connection:
                for batch in _batched(rows, self.upsert_batch_size):
                    connection.execute(self._upsert_statement(batch))
        except SQLAlchemyError as exc:
            raise OperationalContextStoreError(
                f"Could not upsert operational context: {exc}"
            ) from exc
        return len(rows)

    def _upsert_statement(self, rows: list[dict[str, Any]]):
        statement = insert(self.chunks).values(rows)
        update_columns = {
            column.name: getattr(statement.excluded, column.name)
            for column in self.chunks.columns
            if column.name not in {"id", "text_search_vector", "created_at", "updated_at"}
        }
        return statement.on_conflict_do_update(
            index_elements=[self.chunks.c.chunk_id],
            set_=update_columns | {"updated_at": func.now()},
        )

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
        """Run metadata-filtered hybrid search using vector and keyword ranking."""

        if not query.strip():
            raise OperationalContextStoreError("Query cannot be empty.")
        if limit <= 0:
            raise OperationalContextStoreError("limit must be positive.")

        query_vector = self.embeddings.embed_query(query)
        distance = self.chunks.c.embedding.cosine_distance(
            bindparam("query_embedding", value=query_vector)
        )
        conditions = [self.chunks.c.site_id == site_id]
        if start_date is not None:
            conditions.append(self.chunks.c.event_date >= start_date)
        if end_date is not None:
            conditions.append(self.chunks.c.event_date <= end_date)
        if source_types:
            conditions.append(self.chunks.c.source_type.in_(source_types))
        if entity_name:
            conditions.append(_entity_name_condition(self.chunks, entity_name))

        candidate_limit = max(limit * HYBRID_CANDIDATE_MULTIPLIER, HYBRID_CANDIDATE_MINIMUM)

        semantic_statement = (
            select(
                self.chunks.c.chunk_id,
                self.chunks.c.source_type,
                self.chunks.c.source_id,
                self.chunks.c.site_id,
                self.chunks.c.event_date,
                self.chunks.c.entity_type,
                self.chunks.c.entity_id,
                self.chunks.c.entity_name,
                self.chunks.c.text,
                distance.label("distance"),
                literal(None).label("keyword_score"),
            )
            .where(and_(*conditions))
            .order_by(distance)
            .limit(candidate_limit)
        )

        keyword_query = func.websearch_to_tsquery(
            text("'english'"),
            bindparam("keyword_query", value=query),
        )
        keyword_score = func.ts_rank_cd(
            self.chunks.c.text_search_vector,
            keyword_query,
        )
        keyword_statement = (
            select(
                self.chunks.c.chunk_id,
                self.chunks.c.source_type,
                self.chunks.c.source_id,
                self.chunks.c.site_id,
                self.chunks.c.event_date,
                self.chunks.c.entity_type,
                self.chunks.c.entity_id,
                self.chunks.c.entity_name,
                self.chunks.c.text,
                literal(None).label("distance"),
                keyword_score.label("keyword_score"),
            )
            .where(and_(*conditions, self.chunks.c.text_search_vector.op("@@")(keyword_query)))
            .order_by(keyword_score.desc())
            .limit(candidate_limit)
        )
        try:
            with self.engine.connect() as connection:
                semantic_rows = connection.execute(semantic_statement).mappings().all()
                keyword_rows = connection.execute(keyword_statement).mappings().all()
        except SQLAlchemyError as exc:
            raise OperationalContextStoreError(
                f"Could not search operational context: {exc}"
            ) from exc

        rows = _merge_hybrid_rows(semantic_rows, keyword_rows, limit)
        matches = [_search_row_to_match(row) for row in rows]
        return {
            "query": query,
            "count": len(matches),
            "matches": matches,
        }

    def delete_site(
        self,
        site_id: int,
        source_types: set[str] | None = None,
    ) -> int:
        """Delete indexed chunks for a site before rebuilding that slice."""

        conditions = [self.chunks.c.site_id == site_id]
        if source_types:
            conditions.append(self.chunks.c.source_type.in_(source_types))
        statement = delete(self.chunks).where(and_(*conditions))
        try:
            with self.engine.begin() as connection:
                result = connection.execute(statement)
                return int(result.rowcount or 0)
        except SQLAlchemyError as exc:
            raise OperationalContextStoreError(
                f"Could not delete operational context: {exc}"
            ) from exc

    def delete_source(
        self,
        site_id: int,
        source_type: str,
        source_id: str,
    ) -> int:
        statement = delete(self.chunks).where(
            and_(
                self.chunks.c.site_id == site_id,
                self.chunks.c.source_type == source_type,
                self.chunks.c.source_id == str(source_id),
            )
        )
        try:
            with self.engine.begin() as connection:
                result = connection.execute(statement)
                return int(result.rowcount or 0)
        except SQLAlchemyError as exc:
            raise OperationalContextStoreError(
                f"Could not delete operational context source: {exc}"
            ) from exc

    def delete_date_range(
        self,
        site_id: int,
        start_date: date,
        end_date: date,
        source_types: set[str] | None = None,
    ) -> int:
        conditions = [
            self.chunks.c.site_id == site_id,
            self.chunks.c.event_date >= start_date,
            self.chunks.c.event_date <= end_date,
        ]
        if source_types:
            conditions.append(self.chunks.c.source_type.in_(source_types))
        statement = delete(self.chunks).where(and_(*conditions))
        try:
            with self.engine.begin() as connection:
                result = connection.execute(statement)
                return int(result.rowcount or 0)
        except SQLAlchemyError as exc:
            raise OperationalContextStoreError(
                f"Could not delete operational context date range: {exc}"
            ) from exc


class UnavailableOperationalContextStore:
    """Drop-in store used when vector DB cannot be initialized."""

    def __init__(self, reason: str):
        self.reason = reason

    def search(self, *args, **kwargs) -> dict[str, Any]:
        raise OperationalContextStoreError(self.reason)

    def upsert_documents(self, *args, **kwargs) -> int:
        raise OperationalContextStoreError(self.reason)

    def delete_site(self, *args, **kwargs) -> int:
        raise OperationalContextStoreError(self.reason)

    def delete_source(self, *args, **kwargs) -> int:
        raise OperationalContextStoreError(self.reason)

    def delete_date_range(self, *args, **kwargs) -> int:
        raise OperationalContextStoreError(self.reason)


def _engine_from_settings(settings: Settings) -> Engine:
    url = URL.create(
        drivername="postgresql+psycopg",
        username=settings.vector_db_user,
        password=settings.vector_db_password,
        host=settings.vector_db_host,
        port=settings.vector_db_port,
        database=settings.vector_db_name,
    )
    return create_engine(
        url,
        pool_size=settings.vector_db_pool_size,
        max_overflow=settings.vector_db_max_overflow,
        pool_timeout=settings.vector_db_pool_timeout,
        pool_recycle=settings.vector_db_pool_recycle,
        pool_pre_ping=True,
    )


def _batched(rows: list[dict[str, Any]], batch_size: int):
    if batch_size <= 0:
        raise OperationalContextStoreError("upsert_batch_size must be positive.")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _pgvector_extension_exists(connection) -> bool:
    return bool(_pgvector_extension_status(connection)["installed"])


def _pgvector_extension_status(connection) -> dict[str, Any]:
    row = connection.execute(
        text(
            """
            SELECT
                current_database() AS database,
                current_user AS "user",
                inet_server_addr()::text AS server_addr,
                inet_server_port() AS server_port,
                EXISTS (
                    SELECT 1
                    FROM pg_extension
                    WHERE extname = 'vector'
                ) AS installed,
                (
                    SELECT n.nspname
                    FROM pg_extension e
                    JOIN pg_namespace n ON n.oid = e.extnamespace
                    WHERE e.extname = 'vector'
                ) AS extension_schema,
                to_regtype('vector') IS NOT NULL AS type_visible
            """
        )
    ).mappings().one()
    return dict(row)


def _rag_chunks_table(metadata: MetaData, embedding_dimensions: int) -> Table:
    table = Table(
        "rag_chunks",
        metadata,
        Column("id", BigInteger, primary_key=True),
        Column("chunk_id", Text, nullable=False, unique=True),
        Column("source_type", Text, nullable=False),
        Column("source_id", Text, nullable=False),
        Column("site_id", BigInteger, nullable=False),
        Column("event_date", Date),
        Column("event_date_key", Integer),
        Column("entity_type", Text),
        Column("entity_id", Text),
        Column("entity_name", Text),
        Column("text", Text, nullable=False),
        Column("source_updated_at", DateTime),
        Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        Column("embedding", Vector(embedding_dimensions), nullable=False),
        Column(
            "text_search_vector",
            TSVECTOR,
            Computed(
                "to_tsvector('english'::regconfig, COALESCE(text, ''))",
                persisted=True,
            ),
        ),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
        Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    )
    Index("idx_rag_chunks_site_date", table.c.site_id, table.c.event_date)
    Index("idx_rag_chunks_source", table.c.site_id, table.c.source_type)
    Index("idx_rag_chunks_entity", table.c.site_id, table.c.entity_name)
    Index("idx_rag_chunks_metadata", table.c.metadata, postgresql_using="gin")
    Index(
        "idx_rag_chunks_text_search",
        table.c.text_search_vector,
        postgresql_using="gin",
    )
    return table


def _chunk_row(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": str(metadata["source_type"]),
        "source_id": str(metadata["source_id"]),
        "site_id": int(metadata["site_id"]),
        "event_date": _parse_date(metadata.get("event_date")),
        "event_date_key": metadata.get("event_date_key"),
        "entity_type": metadata.get("entity_type"),
        "entity_id": metadata.get("entity_id"),
        "entity_name": metadata.get("entity_name"),
        "source_updated_at": _parse_datetime(metadata.get("source_updated_at")),
    }


def _extra_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    promoted = {
        "source_type",
        "source_id",
        "site_id",
        "chunk_index",
        "event_date",
        "event_date_key",
        "entity_type",
        "entity_id",
        "entity_name",
        "source_updated_at",
    }
    return {key: value for key, value in metadata.items() if key not in promoted}


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _parse_datetime(value: Any):
    if value is None:
        return None
    from datetime import datetime

    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _entity_name_condition(chunks: Table, entity_name: str):
    entity_pattern = f"%{entity_name}%"
    return chunks.c.entity_name.ilike(entity_pattern) | chunks.c.text.ilike(
        entity_pattern
    )


def _merge_hybrid_rows(
    semantic_rows: list[Any],
    keyword_rows: list[Any],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for rank, row in enumerate(semantic_rows, start=1):
        row_data = dict(row)
        chunk_id = str(row_data["chunk_id"])
        merged[chunk_id] = row_data | {
            "semantic_rank": rank,
            "keyword_rank": None,
            "keyword_score": row_data.get("keyword_score"),
        }

    for rank, row in enumerate(keyword_rows, start=1):
        row_data = dict(row)
        chunk_id = str(row_data["chunk_id"])
        if chunk_id not in merged:
            merged[chunk_id] = row_data | {
                "semantic_rank": None,
                "keyword_rank": rank,
            }
        else:
            merged[chunk_id]["keyword_rank"] = rank
            merged[chunk_id]["keyword_score"] = row_data.get("keyword_score")

    for row in merged.values():
        semantic_rank = row.get("semantic_rank")
        keyword_rank = row.get("keyword_rank")
        row["hybrid_score"] = (
            (1 / (RRF_K + semantic_rank) if semantic_rank is not None else 0)
            + (1 / (RRF_K + keyword_rank) if keyword_rank is not None else 0)
        )

    return sorted(
        merged.values(),
        key=lambda row: (
            -row["hybrid_score"],
            row.get("distance") is None,
            row.get("distance") if row.get("distance") is not None else float("inf"),
        ),
    )[:limit]


def _search_row_to_match(row: Any) -> dict[str, Any]:
    event_date = row["event_date"]
    match = {
        "chunk_id": row["chunk_id"],
        "source_type": row["source_type"],
        "source_id": row["source_id"],
        "site_id": row["site_id"],
        "event_date": event_date.isoformat() if event_date else None,
        "entity_type": row["entity_type"],
        "entity_id": row["entity_id"],
        "entity_name": row["entity_name"],
        "distance": row["distance"],
        "text": row["text"],
    }
    for key in ("hybrid_score", "semantic_rank", "keyword_rank", "keyword_score"):
        if key in row and row[key] is not None:
            match[key] = row[key]
    return match
