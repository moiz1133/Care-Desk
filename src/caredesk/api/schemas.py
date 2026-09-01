"""API request/response schemas.

The public contract for the API boundary, kept separate from the internal
retrieval/generation types (`caredesk.retrieval.types`,
`caredesk.generation.types`) so the wire format can stay stable even as
those internals evolve.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

Persona = Literal["patient", "staff"]


class QueryRequest(BaseModel):
    """Body of `POST /query`.

    `persona` has no default: an omitted persona is a validation error
    (422), never an assumed "patient". The retriever's persona filter is a
    security control, not a convenience default, and a permissive default
    here would silently undo it before the request even reaches that layer.
    """

    query: str = Field(min_length=1, max_length=2000)
    persona: Persona
    conversation_id: str | None = Field(
        default=None,
        description="Optional UUID. Generated if absent. Not yet used for anything (no "
        "conversation state exists until the Week 4 decision engine) -- accepted and "
        "echoed only.",
    )
    k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _require_nonempty_after_strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only")
        return stripped

    @field_validator("conversation_id")
    @classmethod
    def _require_valid_uuid_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            UUID(value)
        except ValueError as exc:
            raise ValueError("conversation_id must be a valid UUID") from exc
        return value


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    filename: str


class ContextChunkOut(BaseModel):
    """Raw retrieved chunk, only included when `?include_context=true`."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    rank: int


class RetrievalMetaOut(BaseModel):
    """Retrieval instrumentation.

    Exposed in the response deliberately: this is an internal-facing
    system at this stage, and inspecting retrieval behaviour via the API
    response (rather than reading logs) is worth the extra payload. This
    block should be gated or removed before any external-facing exposure.
    """

    strategy: str
    k: int
    results_returned: int
    top_score: float | None
    latency_ms: float


class GenerationMetaOut(BaseModel):
    """Generation instrumentation, exposed for the same reason as `RetrievalMetaOut`."""

    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class QueryResponse(BaseModel):
    request_id: str
    conversation_id: str
    query: str
    persona: Persona
    answered: bool
    answer: str | None
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    retrieval: RetrievalMetaOut
    generation: GenerationMetaOut
    total_latency_ms: float
    context: list[ContextChunkOut] | None = None
