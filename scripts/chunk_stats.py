"""Print summary statistics for the fixed-size chunker's output.

Loads the whole corpus, chunks every document with the naive fixed-size
chunker, and reports chunk counts, token-count statistics, and the
shortest/longest chunks by token count. This is the Week 1 baseline used
to evaluate the section-aware chunker planned for Week 2.

Usage: uv run python scripts/chunk_stats.py
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from caredesk.config import get_settings
from caredesk.ingestion.chunker import Chunk, chunk_corpus
from caredesk.ingestion.loader import load_corpus


def main() -> None:
    settings = get_settings()
    chunks: list[Chunk] = list(chunk_corpus(load_corpus(settings.corpus_root), settings))

    print(f"Strategy: fixed_{settings.chunk_size}_{settings.chunk_overlap}")
    print(f"Total chunks: {len(chunks)}\n")

    by_type: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_type[chunk.source_type].append(chunk)

    print("Chunks per source_type:")
    for source_type in sorted(by_type):
        print(f"  {source_type}: {len(by_type[source_type])}")

    token_counts = [chunk.token_count for chunk in chunks]
    print(f"\nMean token count:   {statistics.mean(token_counts):.1f}")
    print(f"Median token count: {statistics.median(token_counts):.1f}")

    by_tokens = sorted(chunks, key=lambda chunk: chunk.token_count)

    print("\n5 shortest chunks:")
    for chunk in by_tokens[:5]:
        print(f"  {chunk.chunk_id} ({chunk.token_count} tokens) - doc_id={chunk.doc_id}")

    print("\n5 longest chunks:")
    for chunk in reversed(by_tokens[-5:]):
        print(f"  {chunk.chunk_id} ({chunk.token_count} tokens) - doc_id={chunk.doc_id}")


if __name__ == "__main__":
    main()
