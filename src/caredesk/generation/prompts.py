"""Prompt templates for grounded answer generation.

Kept separate from generator.py so prompts can be versioned and evaluated
independently. Month 5 introduces proper prompt versioning (a registry +
eval gates) — this module just needs a stable, explicit version string
recorded from the start, so early eval results stay attributable once that
exists.
"""

from __future__ import annotations

from collections.abc import Sequence

from caredesk.retrieval.types import RetrievalResult

PROMPT_VERSION = "v1"

# The exact string the model must return, and only this string, when the
# provided context does not support an answer. Checked with a strict
# equality match in generator.py — see the comment there for why.
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

GROUNDED_ANSWER_V1 = f"""You are CareDesk's answer generator. You answer strictly from the
CONTEXT provided with each question, and from nothing else.

Rules:

1. Answer ONLY using information present in the CONTEXT below. Do not use general
   knowledge, do not infer, do not extrapolate, and do not fill gaps with assumptions
   — even if the answer seems obvious.

2. Every factual claim in your answer must be followed by an inline citation to the
   chunk it came from, in the exact form [chunk_id], e.g. [caredesk_faq_hours::0002].
   Use the chunk_id exactly as given in the context. Never invent, paraphrase, or
   guess a chunk_id.

3. If the CONTEXT does not contain enough information to answer the question, respond
   with exactly this token and nothing else: {REFUSAL_SENTINEL}
   Do not soften this into a partial or hedged answer. Do not answer the part you can
   and stay silent on the rest. If you cannot fully support an answer from the
   context, refuse the whole thing.

4. When your answer draws on a chunk whose source_type is medication_leaflet,
   reproduce the relevant wording from that chunk exactly — quote it — rather than
   paraphrasing or summarizing it. This applies only to what the leaflet literally
   states. You are not giving medication advice: do not answer dosage questions,
   "should I stop taking this" questions, double-dose questions, or any other
   question that requires clinical judgement rather than a literal reading of the
   leaflet text. If a question requires that judgement, it is out of scope for you —
   refuse it with {REFUSAL_SENTINEL}.

5. Never present a refusal as an answer, and never present an answer as a refusal.
   If you can fully answer from the context, do so with citations. If you cannot,
   return only {REFUSAL_SENTINEL}.
"""


def format_context(results: Sequence[RetrievalResult]) -> str:
    """Format retrieved chunks into the CONTEXT block for the prompt.

    Chunks are presented in retrieval rank order (best match first) —
    already the order `results` is in. Position bias in LLM context
    windows is real: models tend to weight earlier content more heavily.
    This ordering is the simplest choice for a baseline, not a considered
    one, and should be revisited once eval data exists to show whether
    rank order actually helps or hurts.

    Similarity score is deliberately omitted: including it would invite
    the model to reason about retrieval quality, which isn't its job.
    """
    blocks = [
        f"[{result.chunk_id}]\n"
        f"Title: {result.title}\n"
        f"Source type: {result.source_type}\n"
        f"{result.text}"
        for result in results
    ]
    return "\n\n".join(blocks)


def format_user_message(query: str, results: Sequence[RetrievalResult]) -> str:
    """Build the user-turn message: the question plus its CONTEXT block."""
    return f"Question: {query}\n\nCONTEXT:\n{format_context(results)}"
