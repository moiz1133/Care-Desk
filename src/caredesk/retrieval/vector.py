"""Vector-only retrieval baseline.

Pure cosine-similarity top-k search against `chunks.embedding` — no hybrid
search, keyword matching, reciprocal rank fusion, reranking, query
expansion, or query rewriting. This is the Week 1 baseline every later
retrieval strategy is measured against; improving retrieval quality here
would make that comparison meaningless.

The persona_visibility filter is the one exception: it's a security
control, not a quality optimisation, and it applies here unconditionally.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache

from sqlalchemy import select

from caredesk.config import Settings
from caredesk.ingestion.chunker import get_encoding
from caredesk.ingestion.embedder import embed_text
from caredesk.ingestion.loader import SourceType
from caredesk.retrieval.types import RetrievalResponse, RetrievalResult
from caredesk.storage.models import ChunkRecord, DocumentRecord
from caredesk.storage.session import session_scope

logger = logging.getLogger(__name__)

_VALID_PERSONAS: tuple[str, ...] = ("patient", "staff")

# persona_visibility values a given caller persona may see. "both" is a
# CHUNK-level value (visible to everyone); it is never a valid caller
# persona itself, so it never appears as a key here.
_ALLOWED_VISIBILITY: dict[str, tuple[str, ...]] = {
    "patient": ("patient", "both"),
    "staff": ("staff", "both"),
}


class RetrievalError(ValueError):
    """Raised for an invalid retrieval argument (e.g. an unknown persona)."""


def _validate_persona(persona: str) -> str:
    if persona not in _VALID_PERSONAS:
        raise RetrievalError(f"Invalid persona {persona!r}; must be one of {_VALID_PERSONAS}")
    return persona


class _QueryEmbeddingCache:
    """Exact-match, in-process LRU cache for query embeddings.

    This is NOT the semantic cache planned for Week 3 (which will match on
    embedding similarity above a threshold). This only ever hits on the
    exact same query string, with no similarity matching at all — purely
    so eval runs that repeat identical queries don't pay to re-embed them.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, list[float]] = OrderedDict()

    def get(self, key: str) -> list[float] | None:
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def set(self, key: str, value: list[float]) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)


@lru_cache(maxsize=4)
def _get_query_cache(maxsize: int) -> _QueryEmbeddingCache:
    return _QueryEmbeddingCache(maxsize=maxsize)


async def retrieve(
    query: str,
    persona: str,
    settings: Settings,
    *,
    k: int | None = None,
    source_types: Sequence[SourceType] | None = None,
) -> RetrievalResponse:
    """Retrieve the top-k chunks by cosine similarity, persona-filtered.

    `persona` is required with no default — a caller that omits it gets a
    TypeError, not a permissive fallback. An unrecognised value raises
    `RetrievalError` rather than silently returning no results.

    The persona filter is applied in the SQL WHERE clause, before ranking:
    it constrains the ANN search itself rather than filtering an
    already-ranked Python list, so "top-k" always means the same thing.
    """
    validated_persona = _validate_persona(persona)
    top_k = k if k is not None else settings.vector_retrieval_k

    overall_start = time.monotonic()

    # --- Query embedding (exact-match cached) -----------------------------
    embed_start = time.monotonic()
    cache = _get_query_cache(settings.query_embedding_cache_size)
    cached_vector = cache.get(query)

    encoding = get_encoding(settings.embedding_model)
    query_tokens = len(encoding.encode(query))

    if cached_vector is not None:
        query_vector = cached_vector
    else:
        query_vector = await embed_text(query, settings)
        cache.set(query, query_vector)
    embed_latency_ms = (time.monotonic() - embed_start) * 1000

    # --- Database query ----------------------------------------------------
    db_start = time.monotonic()
    allowed_visibility = _ALLOWED_VISIBILITY[validated_persona]

    # pgvector's <=> operator returns cosine DISTANCE (0 = identical chunk,
    # up to 2 = opposite), not similarity. Converting with (1 - distance)
    # below is what turns that into a similarity score where higher is
    # better — returning the raw distance as "score" would still rank
    # correctly but the numbers would read exactly backwards.
    distance = ChunkRecord.embedding.cosine_distance(query_vector)

    stmt = (
        select(
            ChunkRecord, DocumentRecord.title, DocumentRecord.filename, distance.label("distance")
        )
        .join(DocumentRecord, ChunkRecord.doc_id == DocumentRecord.doc_id)
        .where(ChunkRecord.persona_visibility.in_(allowed_visibility))
    )
    if source_types is not None:
        stmt = stmt.where(
            ChunkRecord.source_type.in_([source_type.value for source_type in source_types])
        )
    stmt = stmt.order_by(distance).limit(top_k)

    with session_scope(settings) as session:
        rows = session.execute(stmt).all()
    db_latency_ms = (time.monotonic() - db_start) * 1000

    results = [
        RetrievalResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=1.0 - chunk_distance,
            rank=rank,
            source_type=chunk.source_type,
            persona_visibility=chunk.persona_visibility,
            title=title,
            filename=filename,
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
        )
        for rank, (chunk, title, filename, chunk_distance) in enumerate(rows, start=1)
    ]

    if len(results) < top_k:
        # HNSW is an approximate index; combined with a restrictive persona
        # (and optional source_type) WHERE clause, it can under-return
        # relative to k even when k matching rows exist, especially on a
        # small corpus. Logged, not worked around — see module docstring.
        logger.debug(
            "Retrieved fewer results than requested: %d/%d (persona=%s, source_types=%s)",
            len(results),
            top_k,
            validated_persona,
            [source_type.value for source_type in source_types] if source_types else None,
        )

    total_latency_ms = (time.monotonic() - overall_start) * 1000

    logger.info(
        "vector_retrieval",
        extra={
            "query_tokens": query_tokens,
            "k_requested": top_k,
            "results_returned": len(results),
            "top_score": results[0].score if results else None,
            "bottom_score": results[-1].score if results else None,
            "embed_latency_ms": embed_latency_ms,
            "db_latency_ms": db_latency_ms,
            "total_latency_ms": total_latency_ms,
            "persona": validated_persona,
            "strategy": "vector_only",
        },
    )

    return RetrievalResponse(
        query=query,
        persona=validated_persona,
        results=results,
        query_embedding_tokens=query_tokens,
        latency_ms=total_latency_ms,
        strategy="vector_only",
    )
