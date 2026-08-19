"""Print summary statistics for the CareDesk document corpus.

Loads every document via `caredesk.ingestion.loader` and reports the count,
mean/max character length, and longest/shortest document per source type,
plus any files on disk not referenced by the manifest. Run this after
editing `data/corpus/manifest.json` to sanity-check the corpus.

Usage: uv run python scripts/corpus_stats.py
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict

from caredesk.config import get_settings
from caredesk.ingestion.loader import (
    LoadedDocument,
    find_unmanifested_files,
    load_document,
    load_manifest,
)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = get_settings()
    corpus_root = settings.corpus_root

    entries = load_manifest(corpus_root)
    docs: list[LoadedDocument] = [load_document(entry, corpus_root) for entry in entries]

    by_type: dict[str, list[LoadedDocument]] = defaultdict(list)
    for doc in docs:
        by_type[doc.source_type].append(doc)

    print(f"Total documents: {len(docs)}\n")

    print("Per source_type:")
    for source_type in sorted(by_type):
        group = by_type[source_type]
        char_counts = [doc.char_count for doc in group]
        print(
            f"  {source_type}: {len(group)} docs, "
            f"mean {statistics.mean(char_counts):.0f} chars, "
            f"max {max(char_counts)} chars"
        )

    longest = max(docs, key=lambda doc: doc.char_count)
    shortest = min(docs, key=lambda doc: doc.char_count)
    print(f"\nLongest document:  {longest.doc_id} ({longest.char_count} chars)")
    print(f"Shortest document: {shortest.doc_id} ({shortest.char_count} chars)")

    unmanifested = find_unmanifested_files(corpus_root, entries)
    print(f"\nUnmanifested files on disk: {len(unmanifested)}")
    for filename in unmanifested:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
