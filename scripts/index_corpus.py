"""Embed and index the corpus into Postgres/pgvector.

Run with --dry-run first: it reports the plan (documents to insert/update/
skip, chunks and tokens to embed, estimated cost, and any documents in the
database no longer in the manifest) without making any API call or write.

Usage:
  uv run python scripts/index_corpus.py --dry-run
  uv run python scripts/index_corpus.py --yes
  uv run python scripts/index_corpus.py --yes --source-type policy_pdf
  uv run python scripts/index_corpus.py --yes --doc-id policy_imaging
  uv run python scripts/index_corpus.py --yes --prune
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from caredesk.config import get_settings
from caredesk.ingestion.embedder import EmbedderError
from caredesk.ingestion.indexer import IndexerError, IndexReport, index_corpus
from caredesk.ingestion.loader import SourceType


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive cost confirmation prompt.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete documents present in the database but absent from the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the plan and estimated cost only; no API calls, no writes.",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        dest="source_types",
        choices=[member.value for member in SourceType],
        help="Restrict indexing to this source_type. Repeatable.",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Restrict indexing to this doc_id. Repeatable.",
    )
    return parser.parse_args(argv)


def _print_report(report: IndexReport, *, dry_run: bool) -> None:
    label = "DRY RUN -- nothing was written" if dry_run else "Index report"
    rows: list[tuple[str, str]] = [
        ("Documents inserted", str(report.documents_inserted)),
        ("Documents updated", str(report.documents_updated)),
        ("Documents skipped (unchanged)", str(report.documents_skipped)),
        ("Documents pruned", str(report.documents_pruned)),
        ("Orphaned doc_ids (in DB, not in manifest)", str(len(report.orphaned_doc_ids))),
        ("Chunks written", str(report.chunks_written)),
        ("Tokens embedded", str(report.tokens_embedded)),
        ("API calls", str(report.api_calls)),
        ("Estimated cost (USD)", f"${report.estimated_cost_usd:.4f}"),
        ("Elapsed (s)", f"{report.elapsed_seconds:.2f}"),
    ]
    name_width = max(len(name) for name, _ in rows)

    print(f"\n{label}")
    print("-" * (name_width + 15))
    for name, value in rows:
        print(f"{name:<{name_width}}  {value}")

    if report.orphaned_doc_ids:
        print("\nOrphaned doc_ids:")
        for doc_id in report.orphaned_doc_ids:
            print(f"  {doc_id}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()

    source_types = [SourceType(value) for value in args.source_types] if args.source_types else None

    try:
        report = asyncio.run(
            index_corpus(
                settings,
                prune=args.prune,
                confirm=args.yes,
                dry_run=args.dry_run,
                source_types=source_types,
                doc_ids=args.doc_ids,
            )
        )
    except (IndexerError, EmbedderError) as exc:
        print(f"Indexing aborted: {exc}", file=sys.stderr)
        return 1

    _print_report(report, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
