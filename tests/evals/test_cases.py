"""Tests for the eval dataset itself (evals/cases/*.yaml via evals/schema.py).

These test the dataset's internal consistency -- file structure, unique
IDs, valid references -- not whether any case's expected_decision is
"correct" in some external sense. That judgment is the whole point of the
dataset and isn't something a unit test can check.
"""

from __future__ import annotations

from collections import Counter

import pytest

from caredesk.config import get_settings
from caredesk.ingestion.loader import load_manifest
from evals.schema import EvalCase, load_cases

EXPECTED_SLICE_COUNTS = {
    "easy": 15,
    "ambiguous": 12,
    "escalate": 13,
    "unanswerable": 10,
}


@pytest.fixture(scope="module")
def all_cases() -> list[EvalCase]:
    return load_cases()


def test_all_files_load_and_validate(all_cases: list[EvalCase]) -> None:
    # load_cases() already ran full validation (unique IDs, known doc_ids,
    # valid enums via pydantic) -- reaching this line without raising is
    # itself most of the test.
    assert len(all_cases) == sum(EXPECTED_SLICE_COUNTS.values())


def test_slice_counts_exact(all_cases: list[EvalCase]) -> None:
    counts = Counter(case.slice for case in all_cases)
    assert dict(counts) == EXPECTED_SLICE_COUNTS


def test_case_ids_unique_across_all_files(all_cases: list[EvalCase]) -> None:
    ids = [case.case_id for case in all_cases]
    duplicates = [case_id for case_id, count in Counter(ids).items() if count > 1]
    assert duplicates == []


def test_every_expected_doc_id_exists_in_manifest(all_cases: list[EvalCase]) -> None:
    known_doc_ids = {entry.doc_id for entry in load_manifest(get_settings().corpus_root)}
    unknown: list[tuple[str, str]] = [
        (case.case_id, doc_id)
        for case in all_cases
        for doc_id in case.expected_source_docs
        if doc_id not in known_doc_ids
    ]
    assert unknown == []


def test_unanswerable_cases_have_empty_expected_source_docs(all_cases: list[EvalCase]) -> None:
    offenders = [
        case.case_id
        for case in all_cases
        if case.slice == "unanswerable" and case.expected_source_docs
    ]
    assert offenders == []


def test_every_source_type_appears_in_at_least_one_easy_case(all_cases: list[EvalCase]) -> None:
    manifest_by_doc_id = {
        entry.doc_id: entry for entry in load_manifest(get_settings().corpus_root)
    }
    all_source_types = {entry.source_type for entry in manifest_by_doc_id.values()}

    covered: set[str] = set()
    for case in all_cases:
        if case.slice != "easy":
            continue
        for doc_id in case.expected_source_docs:
            entry = manifest_by_doc_id.get(doc_id)
            if entry is not None:
                covered.add(entry.source_type)

    missing = all_source_types - covered
    assert missing == set()


def test_no_duplicate_queries_across_slices(all_cases: list[EvalCase]) -> None:
    queries = [case.query.strip().lower() for case in all_cases]
    duplicates = [query for query, count in Counter(queries).items() if count > 1]
    assert duplicates == []


def test_expected_decision_matches_slice_convention(all_cases: list[EvalCase]) -> None:
    """Not one of the explicitly requested tests, but a cheap, high-value
    guard: catches a copy-paste slice/decision mismatch (e.g. an escalate
    case accidentally left as RESOLVE) that none of the other checks would
    notice."""
    expected_decision_by_slice = {
        "easy": "RESOLVE",
        "ambiguous": "CLARIFY",
        "escalate": "ESCALATE",
        "unanswerable": "REFUSE",
    }
    mismatches = [
        (case.case_id, case.slice, case.expected_decision)
        for case in all_cases
        if case.expected_decision != expected_decision_by_slice[case.slice]
    ]
    assert mismatches == []
