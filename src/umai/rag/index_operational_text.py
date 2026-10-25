from __future__ import annotations

import argparse
from datetime import date

from umai.config.logging import configure_logging
from umai.config.settings import Settings
from umai.rag.extractors.operational_text import OperationalTextExtractor
from umai.rag.indexer import OperationalContextIndexer
from umai.rag.vector_store import VectorOperationalContextStore


def main() -> None:
    # This command is intentionally manual/simple for the first RAG slice. It can
    # later be wrapped by cron, a queue worker, or an Ometrics admin action.
    parser = argparse.ArgumentParser(
        description="Index Ometrics text records into the vector DB."
    )
    parser.add_argument("--site-id", type=int, required=True)
    parser.add_argument("--from-date", dest="start_date")
    parser.add_argument("--to-date", dest="end_date")
    parser.add_argument("--source-type", action="append", dest="source_types")
    parser.add_argument(
        "--reset-site",
        action="store_true",
        help="Delete existing indexed chunks for the selected site/source types before indexing.",
    )
    parser.add_argument(
        "--reset-date-range",
        action="store_true",
        help="Delete existing indexed chunks for the selected site/source/date range before indexing.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    # validate() checks LLM/embedding settings, while validate_database() checks
    # the source MySQL connection used by OperationalTextExtractor.
    settings.validate()
    settings.validate_database()

    indexer = OperationalContextIndexer(
        extractor=OperationalTextExtractor.from_settings(settings),
        store=VectorOperationalContextStore.from_settings(settings),
    )
    indexer.store.migrate()
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    source_types = set(args.source_types) if args.source_types else None
    if args.reset_site:
        deleted = indexer.store.delete_site(
            args.site_id,
            source_types,
        )
        print(f"Deleted {deleted} existing indexed chunks.")
    if args.reset_date_range:
        if start_date is None or end_date is None:
            raise SystemExit("--reset-date-range requires --from-date and --to-date.")
        deleted = indexer.store.delete_date_range(
            args.site_id,
            start_date,
            end_date,
            source_types,
        )
        print(f"Deleted {deleted} existing indexed chunks for date range.")
    result = indexer.index(
        site_id=args.site_id,
        start_date=start_date,
        end_date=end_date,
        source_types=source_types,
    )
    # The command reports vector DB chunks, not MySQL writes. No RAG state is
    # stored in the Ometrics database.
    print(f"Indexed {result.documents} documents into {result.chunks} vector chunks.")


if __name__ == "__main__":
    main()
