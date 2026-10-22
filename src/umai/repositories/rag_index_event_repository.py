from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


BATCH_SIZE = 50
MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 300
STALE_PROCESSING_SECONDS = 900
PROCESSOR_LOCK_NAME = "umai_rag_index_events_processor"


class RagIndexEventRepositoryError(RuntimeError):
    """Raised when UMAI cannot read or update the RAG outbox table."""


@dataclass(frozen=True)
class RagIndexEvent:
    id: int
    site_id: int
    source_type: str
    source_id: str
    operation: str
    status: str
    attempts: int


class RagIndexEventRepository:
    """Read and update the MySQL outbox written by Ometrics model events."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._lock_connection = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagIndexEventRepository":
        settings.validate_database()
        url = URL.create(
            drivername="mysql+pymysql",
            username=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
        )
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        )
        return cls(engine)

    def claim_batch(self) -> list[RagIndexEvent]:
        """Claim a small batch of due events for this processor run.

        UMAI runs this command periodically in the intended deployment, so a
        normal SELECT ... FOR UPDATE is enough and stays compatible with common
        MySQL/MariaDB versions. Rows are deleted only after successful indexing.
        """

        try:
            with self.engine.begin() as connection:
                rows = (
                    connection.execute(
                        text(
                            """
                            SELECT
                                id,
                                site_id,
                                source_type,
                                source_id,
                                operation,
                                status,
                                attempts
                            FROM ai_rag_index_events
                            WHERE attempts < :max_attempts
                                AND (
                                    status = 'pending'
                                    OR (
                                        status = 'failed'
                                        AND (
                                            available_at IS NULL
                                            OR available_at <= CURRENT_TIMESTAMP
                                        )
                                    )
                                )
                            ORDER BY id
                            LIMIT :batch_size
                            FOR UPDATE
                            """
                        ),
                        {
                            "max_attempts": MAX_ATTEMPTS,
                            "batch_size": BATCH_SIZE,
                        },
                    )
                    .mappings()
                    .all()
                )
                if not rows:
                    return []

                ids = [int(row["id"]) for row in rows]
                connection.execute(
                    text(
                        """
                        UPDATE ai_rag_index_events
                        SET status = 'processing',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id IN :ids
                        """
                    ).bindparams(bindparam("ids", expanding=True)),
                    {"ids": ids},
                )
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not claim RAG index events: {exc}"
            ) from exc

        return [
            RagIndexEvent(
                id=int(row["id"]),
                site_id=int(row["site_id"]),
                source_type=str(row["source_type"]),
                source_id=str(row["source_id"]),
                operation=str(row["operation"]),
                status=str(row["status"]),
                attempts=int(row["attempts"] or 0),
            )
            for row in rows
        ]

    def acquire_processor_lock(self) -> bool:
        """Acquire the process-wide MySQL lock used by the periodic command."""

        if self._lock_connection is not None:
            return True
        try:
            connection = self.engine.connect()
            acquired = connection.execute(
                text("SELECT GET_LOCK(:lock_name, 0)"),
                {"lock_name": PROCESSOR_LOCK_NAME},
            ).scalar_one()
            if int(acquired or 0) != 1:
                connection.close()
                return False
            self._lock_connection = connection
            return True
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not acquire RAG index processor lock: {exc}"
            ) from exc

    def release_processor_lock(self) -> None:
        """Release the MySQL named lock if this process owns it."""

        connection = self._lock_connection
        self._lock_connection = None
        if connection is None:
            return
        try:
            connection.execute(
                text("SELECT RELEASE_LOCK(:lock_name)"),
                {"lock_name": PROCESSOR_LOCK_NAME},
            )
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not release RAG index processor lock: {exc}"
            ) from exc
        finally:
            connection.close()

    def recover_stale_processing_events(self) -> int:
        """Make old in-flight rows retryable after a crashed processor run."""

        try:
            with self.engine.begin() as connection:
                result = connection.execute(
                    text(
                        f"""
                        UPDATE ai_rag_index_events
                        SET status = 'failed',
                            attempts = LEAST(attempts + 1, :max_attempts),
                            available_at = CURRENT_TIMESTAMP,
                            last_error = 'Recovered stale processing event.',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE status = 'processing'
                            AND updated_at < DATE_SUB(
                                CURRENT_TIMESTAMP,
                                INTERVAL {STALE_PROCESSING_SECONDS} SECOND
                            )
                        """
                    ),
                    {"max_attempts": MAX_ATTEMPTS},
                )
                return int(result.rowcount or 0)
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not recover stale RAG index events: {exc}"
            ) from exc

    def database_name(self) -> str:
        try:
            with self.engine.connect() as connection:
                return str(connection.execute(text("SELECT DATABASE()")).scalar_one())
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not read RAG index event database name: {exc}"
            ) from exc

    def delete_completed(self, event_id: int) -> None:
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM ai_rag_index_events WHERE id = :event_id"),
                    {"event_id": event_id},
                )
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not delete completed RAG index event: {exc}"
            ) from exc

    def mark_failed(self, event: RagIndexEvent, error: str) -> None:
        attempts = event.attempts + 1
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE ai_rag_index_events
                        SET status = 'failed',
                            attempts = :attempts,
                            available_at = CASE
                                WHEN :attempts < :max_attempts
                                    THEN DATE_ADD(
                                        CURRENT_TIMESTAMP,
                                        INTERVAL :retry_backoff_seconds SECOND
                                    )
                                ELSE NULL
                            END,
                            last_error = :last_error,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :event_id
                        """
                    ),
                    {
                        "event_id": event.id,
                        "attempts": attempts,
                        "max_attempts": MAX_ATTEMPTS,
                        "retry_backoff_seconds": RETRY_BACKOFF_SECONDS,
                        "last_error": error[:65535],
                    },
                )
        except SQLAlchemyError as exc:
            raise RagIndexEventRepositoryError(
                f"Could not mark RAG index event as failed: {exc}"
            ) from exc
