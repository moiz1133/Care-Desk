"""Eval metric computation: retrieval, decision, citation, and cost/latency.

Every function here is a pure function over `CaseMetricInput` -- a small,
easily-constructed dataclass decoupled from the harness's full HTTP/pydantic
result shapes, so `tests/evals/test_metrics.py` can build fixtures directly
without needing a real API response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseMetricInput:
    """One case's outcome, in the shape every metric function below consumes.

    `actual_decision=None` and `failed=True` both mean "the harness could
    not get a usable result for this case" (timeout, connection error, or
    an unrecovered 5xx after the one retry) -- distinct from a successful
    call that simply refused, which has a real `actual_decision` of REFUSE.
    """

    case_id: str
    slice: str
    expected_decision: str
    actual_decision: str | None
    expected_doc_ids: list[str]
    # Deduped by doc_id, in rank order, from the deepest retrieval call
    # available for this case (the k=10 diagnostic call when present).
    retrieved_doc_ids_ranked: list[str]
    top_score: float | None  # baseline (primary, deployed-k) call's top score
    refusal_reason: str | None
    citations_count: int
    failed: bool
    retried: bool
    query: str
    total_latency_ms: float | None
    retrieval_latency_ms: float | None
    generation_latency_ms: float | None
    cost_usd: float | None  # primary call only -- see harness.py for why
    input_tokens: int | None
    output_tokens: int | None


def _with_expected_docs(cases: list[CaseMetricInput]) -> list[CaseMetricInput]:
    """Cases usable for recall/precision/MRR/NDCG.

    Unanswerable-slice cases have empty expected_source_docs by
    construction -- recall over an empty set is undefined, so they're
    excluded here and covered instead by false_retrieval_rate/mean_top_score.
    """
    return [case for case in cases if case.expected_doc_ids]


def _first_hit_rank(expected: list[str], retrieved_ranked: list[str]) -> int | None:
    """1-indexed rank of the first retrieved doc_id in `expected`, or None."""
    expected_set = set(expected)
    for rank, doc_id in enumerate(retrieved_ranked, start=1):
        if doc_id in expected_set:
            return rank
    return None


# ---------------------------------------------------------------------------
# Retrieval metrics (document-level, against expected_source_docs)
# ---------------------------------------------------------------------------


def recall_at_k(cases: list[CaseMetricInput], k: int) -> float:
    """Fraction of cases where at least one expected doc appears in top k."""
    scored = _with_expected_docs(cases)
    if not scored:
        return 0.0
    hits = sum(
        1 for case in scored if set(case.retrieved_doc_ids_ranked[:k]) & set(case.expected_doc_ids)
    )
    return hits / len(scored)


def full_recall_at_k(cases: list[CaseMetricInput], k: int) -> float:
    """Fraction of cases where ALL expected docs appear in top k."""
    scored = _with_expected_docs(cases)
    if not scored:
        return 0.0
    hits = sum(
        1 for case in scored if set(case.expected_doc_ids) <= set(case.retrieved_doc_ids_ranked[:k])
    )
    return hits / len(scored)


def precision_at_k(cases: list[CaseMetricInput], k: int) -> float:
    """Mean, per case, of the fraction of the top-k retrieved docs that are expected."""
    scored = _with_expected_docs(cases)
    if not scored:
        return 0.0
    per_case: list[float] = []
    for case in scored:
        top_k = case.retrieved_doc_ids_ranked[:k]
        denom = min(k, len(top_k))
        if denom == 0:
            per_case.append(0.0)
            continue
        hits = len(set(top_k) & set(case.expected_doc_ids))
        per_case.append(hits / denom)
    return sum(per_case) / len(per_case)


def mrr(cases: list[CaseMetricInput]) -> float:
    """Mean reciprocal rank of the first expected doc; 0 for a case with no hit at all."""
    scored = _with_expected_docs(cases)
    if not scored:
        return 0.0
    total = 0.0
    for case in scored:
        rank = _first_hit_rank(case.expected_doc_ids, case.retrieved_doc_ids_ranked)
        total += (1.0 / rank) if rank is not None else 0.0
    return total / len(scored)


def ndcg_at_5(cases: list[CaseMetricInput]) -> float:
    """NDCG@5 with binary relevance -- every expected doc is equally relevant."""
    scored = _with_expected_docs(cases)
    if not scored:
        return 0.0
    total = 0.0
    for case in scored:
        expected_set = set(case.expected_doc_ids)
        top5 = case.retrieved_doc_ids_ranked[:5]
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(top5, start=1)
            if doc_id in expected_set
        )
        ideal_hits = min(len(expected_set), 5)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(scored)


def false_retrieval_rate(cases: list[CaseMetricInput], *, relevance_threshold: float) -> float:
    """Unanswerable slice: fraction whose top score meets/exceeds the
    pre-model refusal threshold -- i.e. cases where the model would be
    called with something that looked plausible, even though nothing in
    the corpus supports an answer."""
    candidates = [case for case in cases if not case.failed and case.top_score is not None]
    if not candidates:
        return 0.0
    hits = sum(
        1
        for case in candidates
        if case.top_score is not None and case.top_score >= relevance_threshold
    )
    return hits / len(candidates)


def mean_top_score(cases: list[CaseMetricInput]) -> float | None:
    """Unanswerable slice: how close the irrelevant matches sit to the threshold."""
    scores = [case.top_score for case in cases if not case.failed and case.top_score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Decision metrics
# ---------------------------------------------------------------------------


def decision_accuracy(cases: list[CaseMetricInput]) -> float:
    if not cases:
        return 0.0
    correct = sum(1 for case in cases if case.actual_decision == case.expected_decision)
    return correct / len(cases)


def per_slice_accuracy(cases: list[CaseMetricInput]) -> dict[str, float]:
    slices = sorted({case.slice for case in cases})
    return {s: decision_accuracy([case for case in cases if case.slice == s]) for s in slices}


def confusion_matrix(cases: list[CaseMetricInput]) -> dict[str, dict[str, int]]:
    """expected_decision -> actual_decision (or "FAILED") -> count.

    A failed case (timeout/error/unrecovered 5xx) has no real
    actual_decision; bucketing it under the literal string "FAILED" keeps
    it visible in the matrix instead of silently vanishing or being
    miscounted as a wrong-but-real decision.
    """
    matrix: dict[str, dict[str, int]] = {}
    for case in cases:
        actual = case.actual_decision if case.actual_decision is not None else "FAILED"
        row = matrix.setdefault(case.expected_decision, {})
        row[actual] = row.get(actual, 0) + 1
    return matrix


def refusal_rate_on_unanswerable(cases: list[CaseMetricInput]) -> float:
    unanswerable = [case for case in cases if case.slice == "unanswerable"]
    if not unanswerable:
        return 0.0
    return sum(1 for case in unanswerable if case.actual_decision == "REFUSE") / len(unanswerable)


def false_refusal_rate_on_easy(cases: list[CaseMetricInput]) -> float:
    """A system that refuses everything scores perfectly on
    refusal_rate_on_unanswerable; this is the metric that catches that."""
    easy = [case for case in cases if case.slice == "easy"]
    if not easy:
        return 0.0
    return sum(1 for case in easy if case.actual_decision == "REFUSE") / len(easy)


def escalation_recall(cases: list[CaseMetricInput]) -> float:
    """0.0 until Week 4's decision engine exists -- reported anyway, not hidden."""
    escalate = [case for case in cases if case.slice == "escalate"]
    if not escalate:
        return 0.0
    return sum(1 for case in escalate if case.actual_decision == "ESCALATE") / len(escalate)


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------

_PRE_MODEL_REFUSAL_REASONS = {"no_results", "low_relevance"}


def hallucinated_citation_rate(cases: list[CaseMetricInput]) -> float:
    """Rate across cases where the model was actually called.

    Pre-model refusals (no_results, low_relevance) never reach citation
    checking at all; including them in the denominator would understate
    the rate relative to what generation actually did when it ran.
    """
    model_called = [
        case
        for case in cases
        if not case.failed and case.refusal_reason not in _PRE_MODEL_REFUSAL_REASONS
    ]
    if not model_called:
        return 0.0
    hallucinated = sum(1 for case in model_called if case.refusal_reason == "hallucinated_citation")
    return hallucinated / len(model_called)


def citation_count_mean(cases: list[CaseMetricInput]) -> float:
    resolved = [case for case in cases if case.actual_decision == "RESOLVE"]
    if not resolved:
        return 0.0
    return sum(case.citations_count for case in resolved) / len(resolved)


def answers_with_zero_citations(cases: list[CaseMetricInput]) -> int:
    """Should be zero by construction -- generate_answer() refuses (no_citations)
    rather than ever resolving with an empty citation list. This just confirms
    the invariant holds in practice, not only in the generator's own code."""
    return sum(
        1 for case in cases if case.actual_decision == "RESOLVE" and case.citations_count == 0
    )


# ---------------------------------------------------------------------------
# Cost and latency
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def latency_percentiles(cases: list[CaseMetricInput], field: str) -> dict[str, float | None]:
    values = [
        value for case in cases if not case.failed and (value := getattr(case, field)) is not None
    ]
    return {"p50": _percentile(values, 50), "p95": _percentile(values, 95)}


def cost_stats(cases: list[CaseMetricInput]) -> dict[str, float]:
    costs = [case.cost_usd for case in cases if not case.failed and case.cost_usd is not None]
    total_cost = sum(costs)
    total_input = sum(case.input_tokens for case in cases if case.input_tokens is not None)
    total_output = sum(case.output_tokens for case in cases if case.output_tokens is not None)
    return {
        "total_cost_usd": total_cost,
        "cost_per_case_usd": (total_cost / len(cases)) if cases else 0.0,
        "total_input_tokens": float(total_input),
        "total_output_tokens": float(total_output),
    }


def embed_cache_hit_rate(cases: list[CaseMetricInput]) -> float:
    """Inferred from repeated query strings within the run, in submission
    order -- /query doesn't expose cache_hit in its response, but the
    underlying query-embedding cache (caredesk.retrieval.vector) is
    exact-match and process-wide, so a query string seen earlier in the
    same run implies a cache hit on this call. `cases` must be in the
    order requests were sent for this to mean anything; with the eval
    dataset's own no-duplicate-queries invariant (see
    tests/evals/test_cases.py) this is always 0.0 today, but the logic is
    written for the general case, not this dataset specifically.
    """
    if not cases:
        return 0.0
    seen: set[str] = set()
    hits = 0
    for case in cases:
        if case.query in seen:
            hits += 1
        seen.add(case.query)
    return hits / len(cases)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

RECALL_KS = (1, 3, 5, 10)


def compute_all_metrics(
    cases: list[CaseMetricInput],
    *,
    relevance_threshold: float,
    implemented_decisions: set[str],
) -> dict[str, object]:
    """Build the full metrics.json structure from one run's cases."""
    unanswerable = [case for case in cases if case.slice == "unanswerable"]

    not_implemented: dict[str, dict[str, object]] = {}
    for decision in sorted({case.expected_decision for case in cases} - implemented_decisions):
        matching = [case for case in cases if case.expected_decision == decision]
        slices = sorted({case.slice for case in matching})
        not_implemented[decision] = {"slices": slices, "count": len(matching)}

    implementable_cases = [
        case for case in cases if case.expected_decision in implemented_decisions
    ]

    return {
        "run": {
            "total_cases": len(cases),
            "failed_cases": sum(1 for case in cases if case.failed),
            "retried_cases": sum(1 for case in cases if case.retried),
        },
        "retrieval": {
            **{f"recall_at_{k}": recall_at_k(cases, k) for k in RECALL_KS},
            **{f"full_recall_at_{k}": full_recall_at_k(cases, k) for k in RECALL_KS},
            **{f"precision_at_{k}": precision_at_k(cases, k) for k in RECALL_KS},
            "mrr": mrr(cases),
            "ndcg_at_5": ndcg_at_5(cases),
            "unanswerable": {
                "false_retrieval_rate": false_retrieval_rate(
                    unanswerable, relevance_threshold=relevance_threshold
                ),
                "mean_top_score": mean_top_score(unanswerable),
            },
        },
        "decision": {
            "accuracy_overall": decision_accuracy(cases),
            "accuracy_implementable": decision_accuracy(implementable_cases),
            "per_slice_accuracy": per_slice_accuracy(cases),
            "confusion_matrix": confusion_matrix(cases),
            "refusal_rate_on_unanswerable": refusal_rate_on_unanswerable(cases),
            "false_refusal_rate_on_easy": false_refusal_rate_on_easy(cases),
            "escalation_recall": escalation_recall(cases),
            "not_yet_implemented": not_implemented,
        },
        "citation": {
            "hallucinated_citation_rate": hallucinated_citation_rate(cases),
            "citation_count_mean": citation_count_mean(cases),
            "answers_with_zero_citations": answers_with_zero_citations(cases),
        },
        "cost_latency": {
            **cost_stats(cases),
            "total_latency_ms": latency_percentiles(cases, "total_latency_ms"),
            "retrieval_latency_ms": latency_percentiles(cases, "retrieval_latency_ms"),
            "generation_latency_ms": latency_percentiles(cases, "generation_latency_ms"),
            "embed_cache_hit_rate": embed_cache_hit_rate(cases),
        },
    }
