"""POST /query -- the retrieval + generation pipeline endpoint.

The route validates input, calls `services.pipeline.run_query_pipeline`,
and maps the result onto `QueryResponse`. It does nothing else: no
decision logic, no branching on what the answer says. Error-to-status-code
mapping lives in `caredesk.api.main`'s exception handlers, not here, so
this function has no error-handling branches of its own.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request

from caredesk.api.dependencies import get_settings_dependency
from caredesk.api.middleware import set_query_identity
from caredesk.api.schemas import (
    CitationOut,
    ContextChunkOut,
    GenerationMetaOut,
    QueryRequest,
    QueryResponse,
    RetrievalMetaOut,
)
from caredesk.config import Settings
from caredesk.services.pipeline import run_query_pipeline

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    include_context: bool = Query(
        default=False,
        description="Include raw retrieved chunk text in the response. Debugging only.",
    ),
    settings: Settings = Depends(get_settings_dependency),
) -> QueryResponse:
    conversation_id = body.conversation_id or str(uuid4())
    # Enriches the RequestContext api.middleware already bound for this
    # request -- persona/conversation_id aren't known until the body
    # validates, which is why this can't happen in the middleware itself.
    set_query_identity(persona=body.persona, conversation_id=conversation_id)

    result = await asyncio.wait_for(
        run_query_pipeline(body.query, body.persona, settings, k=body.k),
        timeout=settings.api_request_timeout_seconds,
    )
    retrieval = result.retrieval
    generation = result.generation

    context = None
    if include_context:
        context = [
            ContextChunkOut(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                title=r.title,
                text=r.text,
                score=r.score,
                rank=r.rank,
            )
            for r in retrieval.results
        ]

    return QueryResponse(
        request_id=request.state.request_id,
        conversation_id=conversation_id,
        query=body.query,
        persona=body.persona,
        answered=not generation.refused,
        answer=generation.answer_text,
        refused=generation.refused,
        refusal_reason=generation.refusal_reason,
        citations=[CitationOut(**citation.model_dump()) for citation in generation.citations],
        retrieval=RetrievalMetaOut(
            strategy=retrieval.strategy,
            k=result.k_requested,
            results_returned=len(retrieval.results),
            top_score=retrieval.results[0].score if retrieval.results else None,
            latency_ms=retrieval.latency_ms,
        ),
        generation=GenerationMetaOut(
            model=generation.model,
            prompt_version=generation.prompt_version,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cost_usd=generation.cost_usd,
            latency_ms=generation.latency_ms,
        ),
        total_latency_ms=result.total_latency_ms,
        context=context,
    )
