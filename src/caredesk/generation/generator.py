"""Grounded answer generation.

Turns a RetrievalResponse into either a cited GeneratedAnswer or a
refusal. The refusal path is the primary feature of this module, not an
edge case: an answer unsupported by retrieved context must never be
produced. Does not implement the resolve/guide/clarify/escalate decision
engine (Week 4) — this module only produces the answer-or-refusal signal;
routing it is someone else's job.
"""

from __future__ import annotations

import logging
import re
import time

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from caredesk.config import Settings
from caredesk.generation.prompts import (
    GROUNDED_ANSWER_V1,
    PROMPT_VERSION,
    REFUSAL_SENTINEL,
    format_user_message,
)
from caredesk.generation.types import Citation, GeneratedAnswer, RefusalReason
from caredesk.observability.tracing import start_generation, start_span
from caredesk.retrieval.types import RetrievalResponse, RetrievalResult

logger = logging.getLogger(__name__)

# Matches [chunk_id] where chunk_id looks like "<doc_id>::<chunk_index>",
# e.g. [caredesk_faq_hours::0002] — the exact shape chunker.py produces.
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+?::\d+)\]")


class GeneratorError(ValueError):
    """Raised when the generator model call itself fails."""


def _extract_citations(answer_text: str) -> list[str]:
    """[chunk_id] references in first-occurrence order, deduplicated."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in _CITATION_PATTERN.finditer(answer_text):
        chunk_id = match.group(1)
        if chunk_id not in seen_set:
            seen_set.add(chunk_id)
            seen.append(chunk_id)
    return seen


def _estimate_cost(input_tokens: int, output_tokens: int, settings: Settings) -> float:
    input_cost = (input_tokens / 1_000_000) * settings.generator_input_price_per_million_tokens_usd
    output_cost = (
        output_tokens / 1_000_000
    ) * settings.generator_output_price_per_million_tokens_usd
    return input_cost + output_cost


def _refusal(
    reason: RefusalReason,
    settings: Settings,
    start: float,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text=None,
        refused=True,
        refusal_reason=reason,
        citations=[],
        model=settings.generator_model,
        prompt_version=PROMPT_VERSION,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimate_cost(input_tokens, output_tokens, settings),
        latency_ms=(time.monotonic() - start) * 1000,
    )


async def generate_answer(
    query: str,
    persona: str,
    retrieval: RetrievalResponse,
    settings: Settings,
) -> GeneratedAnswer:
    """Produce a cited answer or a refusal, grounded in `retrieval.results`.

    Refuses without calling the model at all when retrieval returned
    nothing, or when the top result's score is below
    `settings.min_relevance_score` — these are corpus/retrieval gaps, not
    generation failures, and are logged separately (`pre_model_refusal`)
    from model-decided (`model_refusal`) and citation-verification
    (`no_citations_refusal` / `hallucinated_citation`) refusals, so eval
    can tell the causes apart.

    After generation, every [chunk_id] the model cites is verified against
    the chunk_ids actually supplied in context. A citation that wasn't in
    context is a hallucination and fails the whole response — there is no
    partial credit. An answer with no citations at all (and no refusal
    sentinel) fails the same way.
    """
    start = time.monotonic()

    if not retrieval.results:
        logger.info(
            "pre_model_refusal",
            extra={"reason": "no_results", "query": query, "persona": persona},
        )
        return _refusal("no_results", settings, start)

    top_score = retrieval.results[0].score
    if top_score < settings.min_relevance_score:
        logger.info(
            "pre_model_refusal",
            extra={
                "reason": "low_relevance",
                "query": query,
                "persona": persona,
                "top_score": top_score,
                "min_relevance_score": settings.min_relevance_score,
            },
        )
        return _refusal("low_relevance", settings, start)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": GROUNDED_ANSWER_V1},
        {"role": "user", "content": format_user_message(query, retrieval.results)},
    ]

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    # `input` is the fully rendered prompt, formatted context included --
    # deliberately, so a wrong answer can be traced back to exactly what
    # the model saw rather than reconstructed after the fact. That means
    # retrieved corpus content (leaflet text included) lands in the trace;
    # accepted for this Week-1 debuggability need.
    with start_generation(
        settings,
        "generate",
        model=settings.generator_model,
        model_parameters={
            "temperature": settings.generation_temperature,
            "max_completion_tokens": settings.max_answer_tokens,
        },
        input=messages,
        version=PROMPT_VERSION,
    ) as generate_span:
        try:
            response = await client.chat.completions.create(
                model=settings.generator_model,
                temperature=settings.generation_temperature,
                max_completion_tokens=settings.max_answer_tokens,
                messages=messages,
            )
        except OpenAIError as exc:
            raise GeneratorError(f"Generation call failed ({type(exc).__name__}): {exc}") from exc

        raw_answer = (response.choices[0].message.content or "").strip()
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        generate_span.update(
            output=raw_answer,
            usage_details={"input": input_tokens, "output": output_tokens},
            cost_details={"total": _estimate_cost(input_tokens, output_tokens, settings)},
        )

    # Exact-match only: a hedged reply like "INSUFFICIENT_CONTEXT, but..."
    # must NOT be accepted as a clean refusal (that's exactly the softened
    # refusal the prompt forbids) — it falls through to citation
    # verification below instead, where it will fail as no_citations or
    # hallucinated_citation. Either path still refuses; only the recorded
    # reason differs.
    if raw_answer == REFUSAL_SENTINEL:
        logger.info(
            "model_refusal",
            extra={
                "reason": "model_insufficient",
                "query": query,
                "persona": persona,
                "model": settings.generator_model,
            },
        )
        return _refusal(
            "model_insufficient",
            settings,
            start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    cited_ids = _extract_citations(raw_answer)

    with start_span(settings, "verify_citations", input={"cited_ids": cited_ids}) as verify_span:
        if not cited_ids:
            logger.warning(
                "no_citations_refusal",
                extra={
                    "reason": "no_citations",
                    "query": query,
                    "persona": persona,
                    "model": settings.generator_model,
                },
            )
            verify_span.update(
                output={"verified_ids": [], "hallucinated_ids": [], "outcome": "no_citations"}
            )
            return _refusal(
                "no_citations",
                settings,
                start,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        context_by_id: dict[str, RetrievalResult] = {
            result.chunk_id: result for result in retrieval.results
        }
        hallucinated_ids = [chunk_id for chunk_id in cited_ids if chunk_id not in context_by_id]
        verified_ids = [chunk_id for chunk_id in cited_ids if chunk_id in context_by_id]

        if hallucinated_ids:
            logger.error(
                "hallucinated_citation",
                extra={
                    "reason": "hallucinated_citation",
                    "query": query,
                    "persona": persona,
                    "model": settings.generator_model,
                    "fabricated_chunk_ids": hallucinated_ids,
                },
            )
            verify_span.update(
                level="ERROR",
                output={
                    "verified_ids": verified_ids,
                    "hallucinated_ids": hallucinated_ids,
                    "outcome": "hallucinated_citation",
                },
            )
            return _refusal(
                "hallucinated_citation",
                settings,
                start,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        verify_span.update(
            output={"verified_ids": verified_ids, "hallucinated_ids": [], "outcome": "verified"}
        )

    citations = [
        Citation(
            chunk_id=chunk_id,
            doc_id=context_by_id[chunk_id].doc_id,
            title=context_by_id[chunk_id].title,
            filename=context_by_id[chunk_id].filename,
        )
        for chunk_id in cited_ids
    ]

    return GeneratedAnswer(
        answer_text=raw_answer,
        refused=False,
        refusal_reason=None,
        citations=citations,
        model=settings.generator_model,
        prompt_version=PROMPT_VERSION,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimate_cost(input_tokens, output_tokens, settings),
        latency_ms=(time.monotonic() - start) * 1000,
    )
