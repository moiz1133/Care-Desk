"""CLI: ask a question through the full retrieval -> generation pipeline.

Usage:
  uv run python scripts/ask.py "how do I book an appointment" --persona patient
  uv run python scripts/ask.py "prior auth appeal" --persona staff --show-context
  uv run python scripts/ask.py "coverage" --persona patient --json

--show-context prints the chunks actually supplied to the generator, so a
refusal can be traced to the retriever (nothing relevant came back) versus
the generator (relevant context came back but wasn't usable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from caredesk.config import Settings, get_settings
from caredesk.generation.generator import GeneratorError, generate_answer
from caredesk.generation.types import GeneratedAnswer
from caredesk.ingestion.embedder import EmbedderError
from caredesk.retrieval.types import RetrievalResponse
from caredesk.retrieval.vector import RetrievalError, retrieve


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument(
        "--persona",
        required=True,
        choices=("patient", "staff"),
        help="Caller persona; filters which chunks are visible.",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Number of chunks to retrieve for context."
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved chunks actually supplied to the generator.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print raw JSON instead of formatted output."
    )
    return parser.parse_args(argv)


async def _run(
    query: str, persona: str, settings: Settings, k: int | None
) -> tuple[RetrievalResponse, GeneratedAnswer]:
    retrieval = await retrieve(query, persona, settings, k=k)
    answer = await generate_answer(query, persona, retrieval, settings)
    return retrieval, answer


def _print_context(retrieval: RetrievalResponse) -> None:
    print(f"--- Context: {len(retrieval.results)} chunk(s) retrieved ---")
    for result in retrieval.results:
        preview = result.text[:200].replace("\n", " ")
        if len(result.text) > 200:
            preview += "..."
        print(
            f"[{result.rank}] {result.chunk_id}  score={result.score:.4f}  "
            f"({result.source_type}, {result.persona_visibility})"
        )
        print(f"    {result.title}")
        print(f"    {preview}")
        print()


def _print_answer(answer: GeneratedAnswer) -> None:
    print(f"model={answer.model}  prompt_version={answer.prompt_version}")
    print(
        f"input_tokens={answer.input_tokens}  output_tokens={answer.output_tokens}  "
        f"cost=${answer.cost_usd:.4f}  latency={answer.latency_ms:.1f}ms\n"
    )

    if answer.refused:
        print(f"REFUSED  (reason: {answer.refusal_reason})")
        return

    print(answer.answer_text)
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(f"  [{citation.chunk_id}] {citation.title} ({citation.filename})")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()

    try:
        retrieval, answer = asyncio.run(_run(args.query, args.persona, settings, args.k))
    except (RetrievalError, EmbedderError, GeneratorError) as exc:
        print(f"Ask failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload: dict[str, object] = {"answer": json.loads(answer.model_dump_json())}
        if args.show_context:
            payload["retrieval"] = json.loads(retrieval.model_dump_json())
        print(json.dumps(payload, indent=2))
        return 0

    if args.show_context:
        _print_context(retrieval)
    _print_answer(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
