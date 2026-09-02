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
from caredesk.ingestion.embedder import embed_text, estimate_cost
from caredesk.ingestion.loader import SourceType
from caredesk.observability.tracing import start_generation, start_span
from caredesk.observability.vocabulary import SpanMetadataKey
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


def _source_type_values(source_types: Sequence[SourceType] | None) -> list[str] | None:
    if source_types is None:
        return None
    return [source_type.value for source_type in source_types]


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

    # persona is deliberately not in this span's input or metadata: it's
    # invariant for the whole request and already on the trace (a tag and
    # trace-level metadata, set once in start_query_trace) -- repeating it
    # per span would just be the same duplication commit 9 removed
    # elsewhere, and every span already inherits it automatically via
    # Langfuse's propagate_attributes.
    with start_span(
        settings,
        "retrieve",
        input={"query": query, "k": top_k},
        metadata={
            SpanMetadataKey.STRATEGY: "vector_only",
            SpanMetadataKey.SOURCE_TYPE_FILTER: _source_type_values(source_types),
        },
    ) as retrieve_span:
        # --- Query embedding (exact-match cached) -------------------------
        embed_start = time.monotonic()
        cache = _get_query_cache(settings.query_embedding_cache_size)
        cached_vector = cache.get(query)
        cache_hit = cached_vector is not None

        encoding = get_encoding(settings.embedding_model)
        query_tokens = len(encoding.encode(query))

        with start_generation(
            settings, "embed_query", model=settings.embedding_model, input=query
        ) as embed_span:
            if cached_vector is not None:
                query_vector = cached_vector
            else:
                query_vector = await embed_text(query, settings)
                cache.set(query, query_vector)
            embed_latency_ms = (time.monotonic() - embed_start) * 1000

            # Emitted even on a cache hit -- a silently skipped span would
            # make cache behaviour invisible in the trace, which is exactly
            # what needs to be visible once Week 3's semantic cache lands.
            embed_span.update(
                metadata={
                    SpanMetadataKey.CACHE_HIT: cache_hit,
                    SpanMetadataKey.LATENCY_MS: embed_latency_ms,
                },
                usage_details={"input": query_tokens},
                cost_details={"total": 0.0 if cache_hit else estimate_cost(query_tokens, settings)},
            )

        # --- Database query --------------------------------------------------
        db_start = time.monotonic()
        allowed_visibility = _ALLOWED_VISIBILITY[validated_persona]

        # pgvector's <=> operator returns cosine DISTANCE (0 = identical
        # chunk, up to 2 = opposite), not similarity. Converting with
        # (1 - distance) below is what turns that into a similarity score
        # where higher is better — returning the raw distance as "score"
        # would still rank correctly but the numbers would read exactly
        # backwards.
        distance = ChunkRecord.embedding.cosine_distance(query_vector)

        stmt = (
            select(
                ChunkRecord,
                DocumentRecord.title,
                DocumentRecord.filename,
                distance.label("distance"),
            )
            .join(DocumentRecord, ChunkRecord.doc_id == DocumentRecord.doc_id)
            .where(ChunkRecord.persona_visibility.in_(allowed_visibility))
        )
        if source_types is not None:
            stmt = stmt.where(
                ChunkRecord.source_type.in_([source_type.value for source_type in source_types])
            )
        stmt = stmt.order_by(distance).limit(top_k)

        with start_span(
            settings,
            "vector_search",
            metadata={
                SpanMetadataKey.K: top_k,
                SpanMetadataKey.PERSONA_FILTER_APPLIED: list(allowed_visibility),
                # How many candidates the persona filter actually excluded
                # would need a second query (the same ANN search without
                # the persona_visibility WHERE clause) to compute -- not
                # worth doubling every retrieval's DB cost just to answer
                # "did filtering matter here". Left None rather than
                # omitted: the key exists in the known vocabulary from day
                # one, ready for a cheap answer (e.g. a cached per-corpus
                # persona/source_type histogram) if one shows up later.
                SpanMetadataKey.PERSONA_FILTER_EXCLUDED_COUNT: None,
            },
        ) as search_span:
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

            # Ranked IDs and scores only -- chunk text (possibly clinical
            # leaflet content) never lands in this span's output.
            search_span.update(
                output=[{"chunk_id": r.chunk_id, "score": r.score} for r in results],
                metadata={SpanMetadataKey.LATENCY_MS: db_latency_ms},
            )

        if len(results) < top_k:
            # HNSW is an approximate index; combined with a restrictive
            # persona (and optional source_type) WHERE clause, it can
            # under-return relative to k even when k matching rows exist,
            # especially on a small corpus. Logged, not worked around — see
            # module docstring.
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

        retrieve_span.update(
            output={
                "results_returned": len(results),
                "top_score": results[0].score if results else None,
                "bottom_score": results[-1].score if results else None,
            }
        )

    return RetrievalResponse(
        query=query,
        persona=validated_persona,
        results=results,
        query_embedding_tokens=query_tokens,
        latency_ms=total_latency_ms,
        strategy="vector_only",
    )
