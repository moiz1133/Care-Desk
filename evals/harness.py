"""Eval harness: runs evals/cases/*.yaml against a live /query endpoint.

Calls the real HTTP endpoint, not internal pipeline functions -- this
measures the actual path a caller goes through, serialization included,
not a shortcut through the Python objects underneath it.

Two HTTP calls happen per case:

- "primary": k left at the server's own default (Settings.vector_retrieval_k,
  the real deployed baseline). Drives every decision, citation, and
  cost/latency metric -- this is "what the system actually does."
- "retrieval_diagnostic": same query/persona, k forced to
  Settings.eval_retrieval_diagnostic_k (10 by default). Exists only to
  extend recall/precision/MRR/NDCG past whatever the deployed k=5 can show;
  its answer, citations, and cost are never used for anything else. A
  larger k changes what gets fed to generation (more chunks in the
  prompt), so reusing this call for decision metrics would silently
  measure a different, unconfigured pipeline instead of the real baseline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from caredesk.config import Settings
from evals.metrics import CaseMetricInput
from evals.schema import EvalCase, load_cases

# ---------------------------------------------------------------------------
# The decision mapping problem
# ---------------------------------------------------------------------------
#
# evals/cases/*.yaml expects RESOLVE / CLARIFY / ESCALATE / REFUSE. The
# pipeline today only ever answers or refuses -- there is no decision
# engine until Week 4, so CLARIFY and ESCALATE are not producible by any
# input, ever. Every refusal, regardless of reason, maps to REFUSE.
# Listed explicitly (not a bare `else`) so a refusal_reason this mapping
# doesn't recognize fails loudly here instead of silently becoming REFUSE.

_REFUSAL_REASON_TO_DECISION: dict[str, str] = {
    "no_results": "REFUSE",
    "low_relevance": "REFUSE",
    "model_insufficient": "REFUSE",
    "hallucinated_citation": "REFUSE",
    "no_citations": "REFUSE",
}


def map_actual_decision(*, answered: bool, refused: bool, refusal_reason: str | None) -> str:
    """Map one /query response onto the dataset's decision vocabulary.

    A case whose expected_decision is CLARIFY or ESCALATE will never see
    actual_decision equal that -- that is correct and expected, not a
    mapping bug (see Settings.eval_implemented_decisions and
    evals/metrics.py's not_yet_implemented handling).
    """
    if answered and not refused:
        return "RESOLVE"
    if refused and refusal_reason in _REFUSAL_REASON_TO_DECISION:
        return _REFUSAL_REASON_TO_DECISION[refusal_reason]
    raise ValueError(
        f"Unmapped pipeline outcome: answered={answered} refused={refused} "
        f"refusal_reason={refusal_reason!r}. This means a genuinely new, "
        "unhandled refusal_reason appeared -- not that CLARIFY/ESCALATE need "
        "a branch here; the pipeline cannot produce those yet."
    )


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    score: float
    rank: int


@dataclass
class QueryCallOutcome:
    """One HTTP call to /query and what came back (or didn't)."""

    status_code: int | None = None
    request_id: str | None = None
    answered: bool | None = None
    answer: str | None = None
    refused: bool | None = None
    refusal_reason: str | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    retrieval_strategy: str | None = None
    retrieval_k: int | None = None
    results_returned: int | None = None
    top_score: float | None = None
    retrieval_latency_ms: float | None = None
    generation_model: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    context: list[RetrievedChunkRecord] = field(default_factory=list)
    retried: bool = False
    timed_out: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class CaseResult:
    case_id: str
    slice: str
    persona: str
    query: str
    expected_decision: str
    expected_source_docs: list[str]
    difficulty: str
    tags: list[str]
    actual_decision: str | None
    failed: bool
    primary: QueryCallOutcome
    retrieval_diagnostic: QueryCallOutcome | None


# ---------------------------------------------------------------------------
# HTTP calls
# ---------------------------------------------------------------------------


async def _call_query(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    query: str,
    persona: str,
    k: int | None,
    timeout_seconds: float,
) -> QueryCallOutcome:
    """POST /query once, retrying exactly once on a 5xx.

    A timeout or a connection-level error is recorded as a failure
    immediately, not retried -- only a real 5xx response gets the one
    retry the spec asks for.
    """
    body: dict[str, Any] = {"query": query, "persona": persona}
    if k is not None:
        body["k"] = k
    headers = {"X-Client": "eval"}
    params = {"include_context": "true"}

    outcome = QueryCallOutcome()
    start = time.monotonic()
    attempt = 0
    response: httpx.Response | None = None

    while True:
        attempt += 1
        try:
            response = await client.post(
                f"{base_url}/query",
                json=body,
                headers=headers,
                params=params,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException:
            outcome.timed_out = True
            outcome.error = f"timed out after {timeout_seconds}s"
            outcome.elapsed_seconds = time.monotonic() - start
            return outcome
        except httpx.RequestError as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.elapsed_seconds = time.monotonic() - start
            return outcome

        outcome.status_code = response.status_code

        if response.status_code >= 500 and attempt == 1:
            outcome.retried = True
            continue

        break

    outcome.elapsed_seconds = time.monotonic() - start

    if response.status_code >= 500:
        outcome.error = f"5xx after one retry: {response.status_code}"
        return outcome
    if response.status_code != 200:
        outcome.error = f"unexpected status {response.status_code}: {response.text[:300]}"
        return outcome

    payload = response.json()
    outcome.request_id = payload.get("request_id")
    outcome.answered = payload.get("answered")
    outcome.answer = payload.get("answer")
    outcome.refused = payload.get("refused")
    outcome.refusal_reason = payload.get("refusal_reason")
    outcome.citations = payload.get("citations", [])

    retrieval = payload.get("retrieval") or {}
    outcome.retrieval_strategy = retrieval.get("strategy")
    outcome.retrieval_k = retrieval.get("k")
    outcome.results_returned = retrieval.get("results_returned")
    outcome.top_score = retrieval.get("top_score")
    outcome.retrieval_latency_ms = retrieval.get("latency_ms")

    generation = payload.get("generation") or {}
    outcome.generation_model = generation.get("model")
    outcome.prompt_version = generation.get("prompt_version")
    outcome.input_tokens = generation.get("input_tokens")
    outcome.output_tokens = generation.get("output_tokens")
    outcome.cost_usd = generation.get("cost_usd")
    outcome.generation_latency_ms = generation.get("latency_ms")

    outcome.total_latency_ms = payload.get("total_latency_ms")
    outcome.context = [
        RetrievedChunkRecord(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            title=chunk["title"],
            score=chunk["score"],
            rank=chunk["rank"],
        )
        for chunk in (payload.get("context") or [])
    ]
    return outcome


async def _run_case(
    client: httpx.AsyncClient,
    base_url: str,
    case: EvalCase,
    *,
    primary_k: int | None,
    diagnostic_k: int,
    timeout_seconds: float,
) -> CaseResult:
    primary = await _call_query(
        client,
        base_url,
        query=case.query,
        persona=case.persona,
        k=primary_k,
        timeout_seconds=timeout_seconds,
    )

    failed = primary.error is not None
    actual_decision: str | None = None
    if not failed:
        actual_decision = map_actual_decision(
            answered=bool(primary.answered),
            refused=bool(primary.refused),
            refusal_reason=primary.refusal_reason,
        )

    diagnostic: QueryCallOutcome | None = None
    if not failed:
        # A failed diagnostic call doesn't fail the case -- to_metric_input
        # falls back to the primary call's (shallower) retrieval list, and
        # no decision/citation/cost metric ever reads this call at all.
        diagnostic = await _call_query(
            client,
            base_url,
            query=case.query,
            persona=case.persona,
            k=diagnostic_k,
            timeout_seconds=timeout_seconds,
        )

    return CaseResult(
        case_id=case.case_id,
        slice=case.slice,
        persona=case.persona,
        query=case.query,
        expected_decision=case.expected_decision,
        expected_source_docs=case.expected_source_docs,
        difficulty=case.difficulty,
        tags=case.tags,
        actual_decision=actual_decision,
        failed=failed,
        primary=primary,
        retrieval_diagnostic=diagnostic,
    )


async def run_cases_async(
    cases: list[EvalCase],
    *,
    base_url: str,
    concurrency: int,
    primary_k: int | None,
    diagnostic_k: int,
    timeout_seconds: float,
) -> list[CaseResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[CaseResult | None] = [None] * len(cases)

    async def _worker(index: int, case: EvalCase, client: httpx.AsyncClient) -> None:
        async with semaphore:
            results[index] = await _run_case(
                client,
                base_url,
                case,
                primary_k=primary_k,
                diagnostic_k=diagnostic_k,
                timeout_seconds=timeout_seconds,
            )

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(_worker(i, case, client) for i, case in enumerate(cases)))

    return [result for result in results if result is not None]


def run_eval(
    *,
    base_url: str,
    settings: Settings,
    slices: list[str] | None = None,
    case_ids: list[str] | None = None,
    k: int | None = None,
    concurrency: int | None = None,
) -> list[CaseResult]:
    """Load matching cases and run them against a live /query.

    Synchronous entry point (scripts/run_eval.py is a plain CLI) wrapping
    the async runner above.
    """
    cases = load_cases()
    if slices:
        wanted_slices = set(slices)
        cases = [case for case in cases if case.slice in wanted_slices]
    if case_ids:
        wanted_ids = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted_ids]

    effective_primary_k = k if k is not None else settings.vector_retrieval_k
    diagnostic_k = max(effective_primary_k, settings.eval_retrieval_diagnostic_k)

    return asyncio.run(
        run_cases_async(
            cases,
            base_url=base_url,
            concurrency=concurrency if concurrency is not None else settings.eval_concurrency,
            primary_k=k,
            diagnostic_k=diagnostic_k,
            timeout_seconds=settings.eval_case_timeout_seconds,
        )
    )


# ---------------------------------------------------------------------------
# Conversion to the metrics module's input shape
# ---------------------------------------------------------------------------


def to_metric_input(result: CaseResult) -> CaseMetricInput:
    diagnostic_usable = (
        result.retrieval_diagnostic is not None and result.retrieval_diagnostic.error is None
    )
    source = result.retrieval_diagnostic if diagnostic_usable else result.primary

    retrieved_doc_ids_ranked: list[str] = []
    seen: set[str] = set()
    context = source.context if source is not None else []
    for chunk in sorted(context, key=lambda c: c.rank):
        if chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            retrieved_doc_ids_ranked.append(chunk.doc_id)

    return CaseMetricInput(
        case_id=result.case_id,
        slice=result.slice,
        expected_decision=result.expected_decision,
        actual_decision=result.actual_decision,
        expected_doc_ids=result.expected_source_docs,
        retrieved_doc_ids_ranked=retrieved_doc_ids_ranked,
        top_score=result.primary.top_score,
        refusal_reason=result.primary.refusal_reason,
        citations_count=len(result.primary.citations),
        failed=result.failed,
        retried=result.primary.retried
        or (result.retrieval_diagnostic.retried if result.retrieval_diagnostic else False),
        query=result.query,
        total_latency_ms=result.primary.total_latency_ms,
        retrieval_latency_ms=result.primary.retrieval_latency_ms,
        generation_latency_ms=result.primary.generation_latency_ms,
        cost_usd=result.primary.cost_usd,
        input_tokens=result.primary.input_tokens,
        output_tokens=result.primary.output_tokens,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def result_to_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "slice": result.slice,
        "persona": result.persona,
        "query": result.query,
        "expected_decision": result.expected_decision,
        "expected_source_docs": result.expected_source_docs,
        "difficulty": result.difficulty,
        "tags": result.tags,
        "actual_decision": result.actual_decision,
        "failed": result.failed,
        "primary": dataclasses.asdict(result.primary),
        "retrieval_diagnostic": (
            dataclasses.asdict(result.retrieval_diagnostic) if result.retrieval_diagnostic else None
        ),
    }


def write_raw_jsonl(results: list[CaseResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result_to_dict(result)) + "\n")


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_and_write_raw(
    *,
    base_url: str,
    settings: Settings,
    run_dir: Path,
    slices: list[str] | None = None,
    case_ids: list[str] | None = None,
    k: int | None = None,
    concurrency: int | None = None,
) -> list[CaseResult]:
    """Run the matching cases and persist raw.jsonl immediately.

    Writing happens before any metric computation runs -- a bug in metric
    computation must never lose an already-completed, possibly expensive,
    eval run.
    """
    results = run_eval(
        base_url=base_url,
        settings=settings,
        slices=slices,
        case_ids=case_ids,
        k=k,
        concurrency=concurrency,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_raw_jsonl(results, run_dir / "raw.jsonl")
    return results
