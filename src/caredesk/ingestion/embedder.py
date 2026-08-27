"""Embedding client.

Wraps the OpenAI embeddings API: batches chunks, retries transient
failures with exponential backoff, and bounds concurrency with a
semaphore. Contains no database code — `caredesk.ingestion.indexer` owns
persistence and reporting (token/cost/timing accounting lives there,
derived from `Chunk.token_count`, which is already known before any API
call is made).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel

from caredesk.config import Settings
from caredesk.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# {model: dimension} for models we can validate Settings.embedding_dimension
# against. A model not listed here simply isn't checked — we can't know its
# dimension without an API call, so we let it through and trust the caller.
KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Transient/retryable: rate limits and connection hiccups. Auth and
# invalid-request errors are deliberately absent — retrying those can't
# succeed and would just burn time hiding a real configuration problem.
_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


class EmbedderError(ValueError):
    """Raised for embedding configuration problems or exhausted retries."""


class EmbeddedChunk(BaseModel):
    """A `Chunk` paired with its embedding vector."""

    chunk: Chunk
    embedding: list[float]
    embedding_model: str


def validate_embedding_dimension(settings: Settings) -> None:
    """Raise if `embedding_model` is known to disagree with `embedding_dimension`.

    Meant to be called at the start of any embedding or indexing entry
    point, so a misconfigured Settings fails loudly before any API call or
    database write rather than silently writing wrongly-shaped vectors.
    """
    expected = KNOWN_EMBEDDING_DIMENSIONS.get(settings.embedding_model)
    if expected is not None and expected != settings.embedding_dimension:
        raise EmbedderError(
            f"Settings.embedding_model={settings.embedding_model!r} produces "
            f"{expected}-dimensional vectors, but Settings.embedding_dimension="
            f"{settings.embedding_dimension!r}. Fix the mismatch before indexing."
        )


def estimate_cost(total_tokens: int, settings: Settings) -> float:
    """Estimate embedding cost in USD for `total_tokens`, per Settings pricing."""
    return (total_tokens / 1_000_000) * settings.embedding_price_per_million_tokens_usd


async def _embed_batch_with_retry(
    client: AsyncOpenAI, settings: Settings, texts: list[str]
) -> list[list[float]]:
    kwargs: dict[str, object] = {"model": settings.embedding_model, "input": texts}
    known_default_dim = KNOWN_EMBEDDING_DIMENSIONS.get(settings.embedding_model)
    if known_default_dim is not None and known_default_dim != settings.embedding_dimension:
        # Only pass `dimensions` when deliberately truncating below the
        # model's native size — some older models reject this parameter
        # outright, so we avoid sending it for the common (default) case.
        kwargs["dimensions"] = settings.embedding_dimension

    attempt = 0
    while True:
        try:
            response = await client.embeddings.create(**kwargs)  # type: ignore[arg-type]
            return [item.embedding for item in response.data]
        except _RETRYABLE_ERRORS as exc:
            attempt += 1
            if attempt > settings.embedding_max_retries:
                raise EmbedderError(
                    f"Embedding call failed after {attempt - 1} retries: {exc}"
                ) from exc
            delay = settings.embedding_retry_base_delay_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.1)  # noqa: S311 - jitter, not security-sensitive
            logger.warning(
                "Embedding batch failed (%s), retry %d/%d in %.1fs",
                type(exc).__name__,
                attempt,
                settings.embedding_max_retries,
                delay,
            )
            await asyncio.sleep(delay)
        except OpenAIError as exc:
            # Auth / invalid-request / permission / not-found / anything
            # else OpenAI-specific and non-retryable: fail immediately with
            # a clear message instead of a raw SDK traceback.
            raise EmbedderError(f"Embedding call failed ({type(exc).__name__}): {exc}") from exc


async def embed_chunks(chunks: Sequence[Chunk], settings: Settings) -> list[EmbeddedChunk]:
    """Embed `chunks` in batches of `settings.embedding_batch_size`.

    Up to `settings.embedding_max_concurrency` batches are in flight at
    once. Order of the returned list is not guaranteed to match `chunks`
    (batches complete concurrently) — match results back up via
    `EmbeddedChunk.chunk`.
    """
    validate_embedding_dimension(settings)
    if not chunks:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    semaphore = asyncio.Semaphore(settings.embedding_max_concurrency)
    batch_size = settings.embedding_batch_size
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    async def embed_one_batch(batch: Sequence[Chunk]) -> list[EmbeddedChunk]:
        async with semaphore:
            vectors = await _embed_batch_with_retry(client, settings, [c.text for c in batch])
        return [
            EmbeddedChunk(chunk=chunk, embedding=vector, embedding_model=settings.embedding_model)
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

    results = await asyncio.gather(*(embed_one_batch(batch) for batch in batches))
    return [embedded for batch_result in results for embedded in batch_result]
