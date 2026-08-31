"""CLI for the vector retrieval baseline.

Usage:
  uv run python scripts/search.py "how do I book an appointment" --persona patient
  uv run python scripts/search.py "prior auth appeal process" --persona staff --k 10
  uv run python scripts/search.py "coverage" --persona patient --source-type policy_pdf --json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from caredesk.config import get_settings
from caredesk.ingestion.embedder import EmbedderError
from caredesk.ingestion.loader import SourceType
from caredesk.retrieval.types import RetrievalResponse
from caredesk.retrieval.vector import RetrievalError, retrieve


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", help="The search query text.")
    parser.add_argument(
        "--persona",
        required=True,
        choices=("patient", "staff"),
        help="Caller persona; filters which chunks are visible.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of results (default: Settings.vector_retrieval_k).",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        dest="source_types",
        choices=[member.value for member in SourceType],
        help="Restrict results to this source_type. Repeatable.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print raw JSON instead of a formatted table."
    )
    return parser.parse_args(argv)


def _print_human(response: RetrievalResponse) -> None:
    print(f'Query: "{response.query}"   persona={response.persona}   strategy={response.strategy}')
    print(
        f"{len(response.results)} result(s)   "
        f"query_tokens={response.query_embedding_tokens}   "
        f"latency={response.latency_ms:.1f}ms\n"
    )

    if not response.results:
        print("(no results)")
        return

    for result in response.results:
        preview = result.text[:200].replace("\n", " ")
        if len(result.text) > 200:
            preview += "..."
        print(
            f"[{result.rank}] score={result.score:.4f}  doc_id={result.doc_id}  "
            f"({result.source_type}, {result.persona_visibility})"
        )
        print(f"    {result.title}")
        print(f"    {preview}")
        print()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()

    source_types = [SourceType(value) for value in args.source_types] if args.source_types else None

    try:
        response = asyncio.run(
            retrieve(args.query, args.persona, settings, k=args.k, source_types=source_types)
        )
    except (RetrievalError, EmbedderError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        _print_human(response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
