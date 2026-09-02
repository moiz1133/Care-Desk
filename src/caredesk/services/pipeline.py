"""Query pipeline orchestration: retrieval into generation.

This is plumbing, not a decision layer: it calls `retrieve()` then
`generate_answer()` and returns what they produce. No branching on what
the answer says, no routing, no escalation -- that is the Week 4 decision
engine's job. Keeping this module free of that logic now means Week 4 can
replace what happens *after* a `GeneratedAnswer` exists without touching
this module or the route that calls it.

This is also where the one `caredesk.query` trace per request is opened
(see `observability.tracing`) -- request_id and conversation_id live in
the API layer, and `retrieve()`/`generate_answer()` are deliberately kept
unaware of tracing arguments, so this orchestration seam is where the two
meet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from caredesk.config import Settings
from caredesk.generation.generator import generate_answer
from caredesk.generation.types import GeneratedAnswer
from caredesk.observability.tracing import start_query_trace
from caredesk.retrieval.types import RetrievalResponse
from caredesk.retrieval.vector import retrieve


@dataclass(frozen=True)
class PipelineResult:
    """Everything a caller needs to render a response: both stage outputs, unmodified."""

    retrieval: RetrievalResponse
    generation: GeneratedAnswer
    k_requested: int
    total_latency_ms: float
    trace_id: str


async def run_query_pipeline(
    query: str,
    persona: str,
    settings: Settings,
    *,
    request_id: str,
    conversation_id: str,
    k: int | None = None,
) -> PipelineResult:
    """Retrieve context for `query`, then generate a grounded answer from it.

    `k_requested` is resolved here (rather than left for the caller to
    re-derive from `settings.vector_retrieval_k`) so the route doesn't need
    to know the retriever's default-k rule to report it back accurately.
    """
    start = time.monotonic()
    k_requested = k if k is not None else settings.vector_retrieval_k

    with start_query_trace(
        settings,
        request_id=request_id,
        conversation_id=conversation_id,
        query=query,
        persona=persona,
    ) as trace:
        retrieval = await retrieve(query, persona, settings, k=k_requested)
        generation = await generate_answer(query, persona, retrieval, settings)

        total_latency_ms = (time.monotonic() - start) * 1000

        trace.finalize(
            output=generation.answer_text if not generation.refused else generation.refusal_reason,
            tags=[persona, retrieval.strategy, generation.prompt_version],
            metadata={
                "persona": persona,
                "k": k_requested,
                "answered": not generation.refused,
                "refused": generation.refused,
                "refusal_reason": generation.refusal_reason,
                "retrieval_strategy": retrieval.strategy,
                "prompt_version": generation.prompt_version,
                "citation_count": len(generation.citations),
                "total_latency_ms": total_latency_ms,
                "environment": settings.environment,
            },
        )

    return PipelineResult(
        retrieval=retrieval,
        generation=generation,
        k_requested=k_requested,
        total_latency_ms=total_latency_ms,
        trace_id=trace.trace_id,
    )
