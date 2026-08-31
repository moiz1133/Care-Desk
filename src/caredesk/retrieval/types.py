"""Retrieval result types.

Shared data model for the retrieval layer. `RetrievalResponse.strategy`
exists so later ablations (hybrid, reranked, query-rewritten, ...) can be
distinguished from this pure vector-cosine baseline ("vector_only") when
comparing results.
"""

from __future__ import annotations

from pydantic import BaseModel

from caredesk.ingestion.loader import PersonaVisibility


class RetrievalResult(BaseModel):
    """One ranked chunk returned by a retriever."""

    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine SIMILARITY (1 - cosine distance), 0-1, higher is better
    rank: int  # 1-based
    source_type: str
    persona_visibility: PersonaVisibility
    title: str
    filename: str
    chunk_index: int
    token_count: int


class RetrievalResponse(BaseModel):
    """The full result of one retrieval call, plus instrumentation."""

    query: str
    persona: str
    results: list[RetrievalResult]
    query_embedding_tokens: int
    latency_ms: float
    strategy: str
