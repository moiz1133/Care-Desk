"""Eval case schema, loader, and validation.

This is the most-depended-on artifact of Week 1: every metric for the
next four weeks is measured against `evals/cases/*.yaml`. Validation here
is deliberately strict and fails loudly rather than warning -- a silently
wrong `expected_source_docs` entry (a typo'd doc_id, a slice miscounted
because two cases share a case_id) produces a permanently wrong recall or
slice number for every eval run downstream, and nobody would know to look
for it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from caredesk.config import get_settings
from caredesk.ingestion.loader import load_manifest

CASES_DIR = Path(__file__).parent / "cases"

Slice = Literal["easy", "ambiguous", "escalate", "unanswerable"]
CasePersona = Literal["patient", "staff"]
ExpectedDecision = Literal["RESOLVE", "CLARIFY", "ESCALATE", "REFUSE"]
Difficulty = Literal["easy", "medium", "hard"]


class EvalCase(BaseModel):
    """One hand-authored eval case.

    `extra="forbid"`: a typo'd field name in a YAML row (e.g. `tag` instead
    of `tags`) must fail loudly at load time, not silently vanish as an
    ignored extra key.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    slice: Slice
    query: str = Field(min_length=1)
    persona: CasePersona
    expected_decision: ExpectedDecision
    expected_source_docs: list[str] = Field(default_factory=list)
    notes: str = Field(min_length=1)
    difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _query_not_just_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return value


class EvalDatasetError(ValueError):
    """Raised for anything wrong with the dataset itself, not one case's
    fields -- a duplicate case_id, an expected_source_docs entry that
    doesn't resolve against the corpus manifest, or a case file that isn't
    a YAML list. Distinct from pydantic's `ValidationError` (a single
    case's own fields) so callers can tell "this case is malformed" apart
    from "this dataset is inconsistent across cases."
    """


def _case_files() -> list[Path]:
    return sorted(CASES_DIR.glob("*.yaml"))


def _known_doc_ids() -> set[str]:
    settings = get_settings()
    return {entry.doc_id for entry in load_manifest(settings.corpus_root)}


def _load_all_cases() -> list[EvalCase]:
    known_doc_ids = _known_doc_ids()
    cases: list[EvalCase] = []
    case_id_origin: dict[str, Path] = {}

    for path in _case_files():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise EvalDatasetError(
                f"{path}: expected a YAML list of cases, got {type(raw).__name__}"
            )

        for index, row in enumerate(raw):
            try:
                case = EvalCase.model_validate(row)
            except ValidationError as exc:
                raise EvalDatasetError(f"{path} (item {index}): {exc}") from exc

            if case.case_id in case_id_origin:
                raise EvalDatasetError(
                    f"duplicate case_id {case.case_id!r}: defined in both "
                    f"{case_id_origin[case.case_id]} and {path}"
                )
            case_id_origin[case.case_id] = path

            unknown_docs = sorted(
                doc_id for doc_id in case.expected_source_docs if doc_id not in known_doc_ids
            )
            if unknown_docs:
                raise EvalDatasetError(
                    f"{path} case {case.case_id!r}: expected_source_docs references "
                    f"doc_id(s) not in manifest.json: {unknown_docs}"
                )

            cases.append(case)

    return cases


def load_cases(slice: str | None = None) -> list[EvalCase]:
    """Load and validate every eval case across evals/cases/*.yaml.

    Validation always runs against the full dataset, even when `slice`
    filters what's returned: a duplicate case_id or a bad doc_id in a
    slice nobody asked for is still wrong for every other caller, so a
    filtered load can't be allowed to succeed quietly while that's broken.
    """
    cases = _load_all_cases()
    if slice is None:
        return cases
    return [case for case in cases if case.slice == slice]


def print_stats() -> None:
    """Print counts per slice, persona, and difficulty, plus which source
    types the easy slice covers -- the one slice with an explicit
    "spread across all five source types" requirement, so it's the one
    coverage is worth checking by source type rather than just by count.
    """
    manifest_by_doc_id = {
        entry.doc_id: entry for entry in load_manifest(get_settings().corpus_root)
    }
    cases = load_cases()

    print(f"Total cases: {len(cases)}")

    print("\nBy slice:")
    for slice_name, count in sorted(Counter(case.slice for case in cases).items()):
        print(f"  {slice_name}: {count}")

    print("\nBy persona:")
    for persona, count in sorted(Counter(case.persona for case in cases).items()):
        print(f"  {persona}: {count}")

    print("\nBy difficulty:")
    for difficulty, count in sorted(Counter(case.difficulty for case in cases).items()):
        print(f"  {difficulty}: {count}")

    all_source_types = {entry.source_type for entry in manifest_by_doc_id.values()}
    covered: set[str] = set()
    for case in cases:
        if case.slice != "easy":
            continue
        for doc_id in case.expected_source_docs:
            entry = manifest_by_doc_id.get(doc_id)
            if entry is not None:
                covered.add(entry.source_type)

    print("\nSource type coverage (easy slice):")
    for source_type in sorted(all_source_types):
        mark = "x" if source_type in covered else " "
        print(f"  [{mark}] {source_type}")
    missing = sorted(all_source_types - covered)
    if missing:
        print(f"  NOT COVERED: {missing}")


if __name__ == "__main__":
    print_stats()
