"""Generation result types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RefusalReason = Literal[
    "no_results",
    "low_relevance",
    "model_insufficient",
    "hallucinated_citation",
    "no_citations",
]


class Citation(BaseModel):
    """One verified [chunk_id] reference from a generated answer."""

    chunk_id: str
    doc_id: str
    title: str
    filename: str


class GeneratedAnswer(BaseModel):
    """The result of one `generate_answer` call: a cited answer or a refusal.

    Refusal is a first-class outcome, not an error — `refused=True` with a
    populated `refusal_reason` is a normal, expected response shape.
    """

    answer_text: str | None
    refused: bool
    refusal_reason: RefusalReason | None
    citations: list[Citation]
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
