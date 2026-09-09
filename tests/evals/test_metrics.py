"""Tests for evals/metrics.py, against hand-constructed fixtures with known answers."""

from __future__ import annotations

import pytest

from evals.harness import map_actual_decision
from evals.metrics import (
    CaseMetricInput,
    confusion_matrix,
    decision_accuracy,
    false_retrieval_rate,
    mean_top_score,
    mrr,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _case(
    case_id: str = "c1",
    *,
    slice: str = "easy",
    expected_decision: str = "RESOLVE",
    actual_decision: str | None = "RESOLVE",
    expected_doc_ids: list[str] | None = None,
    retrieved_doc_ids_ranked: list[str] | None = None,
    top_score: float | None = 0.5,
    refusal_reason: str | None = None,
    citations_count: int = 1,
    failed: bool = False,
    retried: bool = False,
    query: str = "some query",
) -> CaseMetricInput:
    return CaseMetricInput(
        case_id=case_id,
        slice=slice,
        expected_decision=expected_decision,
        actual_decision=actual_decision,
        expected_doc_ids=expected_doc_ids if expected_doc_ids is not None else ["doc_a"],
        retrieved_doc_ids_ranked=retrieved_doc_ids_ranked
        if retrieved_doc_ids_ranked is not None
        else ["doc_a"],
        top_score=top_score,
        refusal_reason=refusal_reason,
        citations_count=citations_count,
        failed=failed,
        retried=retried,
        query=query,
        total_latency_ms=100.0,
        retrieval_latency_ms=40.0,
        generation_latency_ms=60.0,
        cost_usd=0.001,
        input_tokens=100,
        output_tokens=20,
    )


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def test_recall_at_k_for_a_known_ranking() -> None:
    cases = [
        # expected doc at rank 2 -- hit at k=3/5, miss at k=1.
        _case(
            "c1", expected_doc_ids=["doc_b"], retrieved_doc_ids_ranked=["doc_a", "doc_b", "doc_c"]
        ),
        # expected doc not retrieved at all -- miss at every k.
        _case(
            "c2", expected_doc_ids=["doc_z"], retrieved_doc_ids_ranked=["doc_a", "doc_b", "doc_c"]
        ),
    ]

    assert recall_at_k(cases, k=1) == 0.0
    assert recall_at_k(cases, k=3) == 0.5
    assert recall_at_k(cases, k=5) == 0.5


def test_mrr_including_a_case_with_no_hit() -> None:
    cases = [
        # first (and only) expected doc at rank 1 -> reciprocal rank 1.0
        _case("c1", expected_doc_ids=["doc_a"], retrieved_doc_ids_ranked=["doc_a", "doc_b"]),
        # first expected doc at rank 4 -> reciprocal rank 0.25
        _case(
            "c2",
            expected_doc_ids=["doc_d"],
            retrieved_doc_ids_ranked=["doc_a", "doc_b", "doc_c", "doc_d"],
        ),
        # no expected doc retrieved at all -> reciprocal rank 0.0
        _case("c3", expected_doc_ids=["doc_z"], retrieved_doc_ids_ranked=["doc_a", "doc_b"]),
    ]

    assert mrr(cases) == (1.0 + 0.25 + 0.0) / 3


def test_unanswerable_cases_excluded_from_recall_included_in_false_retrieval_rate() -> None:
    unanswerable = [
        _case(
            "u1",
            slice="unanswerable",
            expected_doc_ids=[],
            retrieved_doc_ids_ranked=["doc_a"],
            top_score=0.6,
        ),
        _case(
            "u2",
            slice="unanswerable",
            expected_doc_ids=[],
            retrieved_doc_ids_ranked=["doc_b"],
            top_score=0.1,
        ),
    ]
    answerable = [_case("a1", expected_doc_ids=["doc_a"], retrieved_doc_ids_ranked=["doc_a"])]

    # recall over an empty expected set is undefined -- excluded entirely,
    # so recall over the combined list matches the answerable-only result.
    assert recall_at_k(unanswerable + answerable, k=5) == recall_at_k(answerable, k=5) == 1.0

    # false_retrieval_rate: one of the two unanswerable cases sits above
    # the 0.3 threshold (0.6 >= 0.3), the other doesn't (0.1 < 0.3).
    assert false_retrieval_rate(unanswerable, relevance_threshold=0.3) == 0.5


def test_mean_top_score_over_unanswerable_slice() -> None:
    unanswerable = [
        _case("u1", slice="unanswerable", expected_doc_ids=[], top_score=0.2),
        _case("u2", slice="unanswerable", expected_doc_ids=[], top_score=0.4),
    ]
    assert mean_top_score(unanswerable) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Decision metrics
# ---------------------------------------------------------------------------


def test_confusion_matrix_sums_to_case_count() -> None:
    cases = [
        _case("c1", expected_decision="RESOLVE", actual_decision="RESOLVE"),
        _case("c2", expected_decision="RESOLVE", actual_decision="REFUSE"),
        _case("c3", expected_decision="UNANSWERABLE_STAND_IN", actual_decision="REFUSE"),
        # a failed (timed-out/errored) case: no actual_decision at all.
        _case("c4", expected_decision="RESOLVE", actual_decision=None, failed=True),
    ]

    matrix = confusion_matrix(cases)
    total = sum(count for row in matrix.values() for count in row.values())
    assert total == len(cases)
    assert matrix["RESOLVE"]["FAILED"] == 1


def test_failed_case_is_recorded_not_dropped() -> None:
    """A case that timed out (or errored, or 5xx'd after its retry) still
    counts toward every aggregate -- it's an incorrect result, not an
    absence. This is the metrics-side half of "a timeout is a recorded
    failure, not a crash"; the HTTP-level half lives in evals/harness.py.
    """
    cases = [
        _case("c1", expected_decision="RESOLVE", actual_decision="RESOLVE", failed=False),
        _case("c2", expected_decision="RESOLVE", actual_decision=None, failed=True),
    ]

    assert decision_accuracy(cases) == 0.5  # the failed case counts as wrong, and is counted
    matrix = confusion_matrix(cases)
    assert sum(count for row in matrix.values() for count in row.values()) == 2
    assert matrix["RESOLVE"]["FAILED"] == 1


# ---------------------------------------------------------------------------
# Decision mapping (evals/harness.py, exercised via metrics-adjacent fixtures)
# ---------------------------------------------------------------------------


def test_decision_mapping_handles_every_refusal_reason() -> None:
    assert map_actual_decision(answered=True, refused=False, refusal_reason=None) == "RESOLVE"

    for reason in (
        "no_results",
        "low_relevance",
        "model_insufficient",
        "hallucinated_citation",
        "no_citations",
    ):
        assert map_actual_decision(answered=False, refused=True, refusal_reason=reason) == "REFUSE"


def test_decision_mapping_raises_on_unrecognized_combination() -> None:
    with pytest.raises(ValueError, match="Unmapped pipeline outcome"):
        map_actual_decision(answered=False, refused=True, refusal_reason="some_new_reason")
