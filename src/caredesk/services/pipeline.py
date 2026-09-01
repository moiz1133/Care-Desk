"""Query pipeline orchestration: retrieval into generation.

This is plumbing, not a decision layer: it calls `retrieve()` then
`generate_answer()` and returns what they produce. No branching on what
the answer says, no routing, no escalation -- that is the Week 4 decision
engine's job. Keeping this module free of that logic now means Week 4 can
replace what happens *after* a `GeneratedAnswer` exists without touching
this module or the route that calls it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from caredesk.config import Settings
from caredesk.generation.generator import generate_answer
from caredesk.generation.types import GeneratedAnswer
from caredesk.retrieval.types import RetrievalResponse
from caredesk.retrieval.vector import retrieve


@dataclass(frozen=True)
class PipelineResult:
    """Everything a caller needs to render a response: both stage outputs, unmodified."""

    retrieval: RetrievalResponse
    generation: GeneratedAnswer
    k_requested: int
    total_latency_ms: float


async def run_query_pipeline(
    query: str,
    persona: str,
    settings: Settings,
    *,
    k: int | None = None,
) -> PipelineResult:
    """Retrieve context for `query`, then generate a grounded answer from it.

    `k_requested` is resolved here (rather than left for the caller to
    re-derive from `settings.vector_retrieval_k`) so the route doesn't need
    to know the retriever's default-k rule to report it back accurately.
    """
    start = time.monotonic()
    k_requested = k if k is not None else settings.vector_retrieval_k

    retrieval = await retrieve(query, persona, settings, k=k_requested)
    generation = await generate_answer(query, persona, retrieval, settings)

    total_latency_ms = (time.monotonic() - start) * 1000
    return PipelineResult(
        retrieval=retrieval,
        generation=generation,
        k_requested=k_requested,
        total_latency_ms=total_latency_ms,
    )
