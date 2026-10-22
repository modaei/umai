from __future__ import annotations

import logging
from typing import Protocol

from umai.config.logging import configure_logging
from umai.config.settings import Settings
from umai.rag.extractors.operational_text import (
    OperationalTextExtractor,
    supported_source_types,
)
from umai.rag.indexer import OperationalContextIndexer
from umai.rag.vector_store import VectorOperationalContextStore
from umai.repositories.rag_index_event_repository import (
    RagIndexEvent,
    RagIndexEventRepository,
)


logger = logging.getLogger(__name__)


class Indexer(Protocol):
    def index_source(
        self,
        site_id: int,
        source_type: str,
        source_id: str,
    ):
        ...

    def delete_source(
        self,
        site_id: int,
        source_type: str,
        source_id: str,
    ):
        ...


class RagIndexEventProcessor:
    def __init__(
        self,
        repository: RagIndexEventRepository,
        indexer: Indexer,
        worker_logger: logging.Logger | None = None,
    ):
        self.repository = repository
        self.indexer = indexer
        self.logger = worker_logger or logger

    def run(self) -> int:
        self.logger.info("Starting UMAI RAG index event processor.")
        try:
            self.logger.info(
                "UMAI RAG index event processor connected to MySQL database %s.",
                self.repository.database_name(),
            )
        except Exception:
            self.logger.exception("UMAI RAG index event processor database check failed.")
            raise

        if not self.repository.acquire_processor_lock():
            self.logger.info("Another UMAI RAG index event processor is already running.")
            return 0

        processed = 0
        try:
            recovered = self.repository.recover_stale_processing_events()
            if recovered:
                self.logger.warning(
                    "Recovered %s stale RAG index processing event(s).",
                    recovered,
                )
            while True:
                batch_count = self.run_once()
                if batch_count == 0:
                    break
                processed += batch_count
            return processed
        finally:
            self.repository.release_processor_lock()

    def run_once(self) -> int:
        events = self.repository.claim_batch()
        for event in events:
            self._process_event(event)
        return len(events)

    def _process_event(self, event: RagIndexEvent) -> None:
        try:
            self._validate_event(event)
            if event.operation == "deleted":
                self.indexer.delete_source(
                    event.site_id,
                    event.source_type,
                    event.source_id,
                )
            else:
                self.indexer.index_source(
                    event.site_id,
                    event.source_type,
                    event.source_id,
                )
            self.repository.delete_completed(event.id)
            if event.attempts > 0:
                self.logger.info(
                    "Previously failed RAG index event succeeded.",
                    extra=_event_log_context(event),
                )
        except Exception as exc:
            self.logger.exception(
                "RAG index event vectorization failed.",
                extra=_event_log_context(event) | {"next_attempt": event.attempts + 1},
            )
            self.repository.mark_failed(event, str(exc))

    def _validate_event(self, event: RagIndexEvent) -> None:
        if event.operation not in {"created", "updated", "deleted"}:
            raise ValueError(f"Unsupported RAG index operation: {event.operation}")
        if event.source_type not in supported_source_types():
            raise ValueError(f"Unsupported RAG source type: {event.source_type}")


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    settings.validate()
    settings.validate_database()

    indexer = OperationalContextIndexer(
        extractor=OperationalTextExtractor.from_settings(settings),
        store=VectorOperationalContextStore.from_settings(settings),
    )
    indexer.store.migrate()

    processor = RagIndexEventProcessor(
        repository=RagIndexEventRepository.from_settings(settings),
        indexer=indexer,
    )
    processor.run()


def _event_log_context(event: RagIndexEvent) -> dict[str, object]:
    return {
        "event_id": event.id,
        "site_id": event.site_id,
        "source_type": event.source_type,
        "source_id": event.source_id,
        "operation": event.operation,
        "attempts": event.attempts,
    }


if __name__ == "__main__":
    main()
