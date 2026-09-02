"""Query pipeline orchestration: retrieval into generation.

This is plumbing, not a decision layer: it calls `retrieve()` then
`generate_answer()` and returns what they produce. No branching on what
the answer says, no routing, no escalation -- that is the Week 4 decision
engine's job. Keeping this module free of that logic now means Week 4 can
replace what happens *after* a `GeneratedAnswer` exists without touching
this module or the route that calls it.

This is also where the one `caredesk.query` trace per request is opened
(see `observability.tracing`). Unlike commit 8, this module no longer
threads request_id/conversation_id through as parameters:
`start_query_trace` reads them from the ambient `RequestContext` that
`api.middleware` establishes and the route enriches, which is what keeps
identity sourced from exactly one place instead of being passed down
redundantly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from caredesk.config import Settings
from caredesk.generation.generator import generate_answer
from caredesk.generation.types import GeneratedAnswer
from caredesk.observability.tracing import start_query_trace
from caredesk.observability.vocabulary import TraceMetadataKey
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
    k: int | None = None,
) -> PipelineResult:
    """Retrieve context for `query`, then generate a grounded answer from it.

    `k_requested` is resolved here (rather than left for the caller to
    re-derive from `settings.vector_retrieval_k`) so the route doesn't need
    to know the retriever's default-k rule to report it back accurately.
    """
    start = time.monotonic()
    k_requested = k if k is not None else settings.vector_retrieval_k

    with start_query_trace(settings, query=query) as trace:
        retrieval = await retrieve(query, persona, settings, k=k_requested)
        generation = await generate_answer(query, persona, retrieval, settings)

        total_latency_ms = (time.monotonic() - start) * 1000

        trace.finalize(
            output=generation.answer_text if not generation.refused else generation.refusal_reason,
            metadata={
                TraceMetadataKey.K: k_requested,
                TraceMetadataKey.ANSWERED: not generation.refused,
                TraceMetadataKey.REFUSED: generation.refused,
                TraceMetadataKey.REFUSAL_REASON: generation.refusal_reason,
                TraceMetadataKey.RETRIEVAL_STRATEGY: retrieval.strategy,
                TraceMetadataKey.RESULTS_RETURNED: len(retrieval.results),
                TraceMetadataKey.PROMPT_VERSION: generation.prompt_version,
                TraceMetadataKey.CITATION_COUNT: len(generation.citations),
                TraceMetadataKey.TOTAL_LATENCY_MS: total_latency_ms,
            },
        )

    return PipelineResult(
        retrieval=retrieval,
        generation=generation,
        k_requested=k_requested,
        total_latency_ms=total_latency_ms,
        trace_id=trace.trace_id,
    )
