"""Fire a fixed batch of requests against a running /query endpoint and
print a manual checklist for confirming the Langfuse trace hierarchy.

This is a chore script, not a test: it makes no assertions of its own and
changes no pipeline or tracing code. Its only job is to generate a
controlled, labeled set of traces and hand back the URLs plus a checklist
of what a human should confirm by looking at the Langfuse UI.

Usage:
    uv run uvicorn caredesk.api.main:app --host 127.0.0.1 --port 8000  # in one shell
    uv run python scripts/verify_traces.py                             # in another

Requires a running server (this talks real HTTP, not the in-process
pipeline scripts/ask.py and scripts/search.py use) and Langfuse credentials
configured the same way the server has them -- this script computes trace
URLs itself, it doesn't need to be the process that traced the request.
"""

from __future__ import annotations

import argparse
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from caredesk.config import Settings, get_settings
from caredesk.ingestion.loader import ManifestEntry, load_manifest
from caredesk.observability.tracing import get_langfuse_client

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_WAIT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Request batch
# ---------------------------------------------------------------------------


@dataclass
class RequestSpec:
    number: str  # "1".."9", or "8.1"/"8.2"/"8.3" for the conversation group
    category: str  # e.g. "patient / resolved"
    persona: str
    query: str
    expected: str  # human-readable: which doc this should hit, or why not
    conversation_id: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def _doc(manifest: list[ManifestEntry], doc_id: str) -> ManifestEntry:
    for entry in manifest:
        if entry.doc_id == doc_id:
            return entry
    raise SystemExit(f"verify_traces: doc_id {doc_id!r} not found in manifest.json")


def build_batch(manifest: list[ManifestEntry]) -> list[RequestSpec]:
    """The 9 requests from the commit spec, in order.

    Query strings come from the corpus itself (manifest titles, which in
    this corpus already read as the question a user would ask) rather than
    invented text, so each request's expected document is traceable back
    to a real `doc_id`.
    """
    faq_hours = _doc(manifest, "faq_clinic_hours")
    faq_insurance = _doc(manifest, "faq_insurance_accepted")
    faq_cancellation = _doc(manifest, "faq_appointment_cancellation")
    policy_rx = _doc(manifest, "policy_prescription_drugs")
    rb_prior_auth = _doc(manifest, "rb_prior_auth_appeals")

    conversation_id = str(uuid.uuid4())

    return [
        RequestSpec(
            number="1",
            category="patient / resolved (FAQ)",
            persona="patient",
            query=faq_hours.title,
            expected=f"{faq_hours.doc_id} -- {faq_hours.title!r}",
        ),
        RequestSpec(
            number="2",
            category="patient / resolved (policy)",
            persona="patient",
            query=f"What is the {policy_rx.title.lower()}?",
            expected=f"{policy_rx.doc_id} -- {policy_rx.title!r}",
        ),
        RequestSpec(
            number="3",
            category="staff / resolved (runbook)",
            persona="staff",
            query=f"What is the {rb_prior_auth.title.lower()}?",
            expected=f"{rb_prior_auth.doc_id} -- {rb_prior_auth.title!r}",
        ),
        # NOTE: labeled "no results" per the commit spec's intent (a query
        # entirely outside the corpus), but vector-only top-k retrieval
        # always returns its k nearest neighbors regardless of how weak the
        # match is, so refusal_reason will most likely still come back
        # low_relevance rather than a literal empty result set -- true
        # no_results isn't reachable through this retriever for any patient
        # query, since most of the corpus is persona_visibility="both".
        # Both are pre-model refusals, which is what the checklist actually
        # cares about (no generate span at all), so this doesn't weaken the
        # check -- just don't be surprised by the printed reason.
        RequestSpec(
            number="4",
            category="patient / refused -- no results",
            persona="patient",
            query="What's the weather forecast for tomorrow?",
            expected="none -- entirely outside the corpus",
        ),
        RequestSpec(
            number="5",
            category="patient / refused -- low relevance",
            persona="patient",
            query="Can I bring my emotional support dog to my appointment?",
            expected="none -- clinic-adjacent wording, but not documented anywhere",
        ),
        RequestSpec(
            number="6",
            category="patient / persona block",
            persona="patient",
            query=f"What is the {rb_prior_auth.title.lower()}?",
            expected=(
                f"MUST NOT surface {rb_prior_auth.doc_id} (persona_visibility=staff) -- "
                "same query as request 3, asked as patient"
            ),
        ),
        RequestSpec(
            number="7",
            category="patient / cache hit",
            persona="patient",
            query=faq_hours.title,
            expected=f"{faq_hours.doc_id} -- verbatim repeat of request 1's query",
        ),
        RequestSpec(
            number="8.1",
            category="patient / conversation (turn 1/3)",
            persona="patient",
            query=faq_hours.title,
            expected=f"{faq_hours.doc_id} -- {faq_hours.title!r}",
            conversation_id=conversation_id,
        ),
        RequestSpec(
            number="8.2",
            category="patient / conversation (turn 2/3)",
            persona="patient",
            query=faq_insurance.title,
            expected=f"{faq_insurance.doc_id} -- {faq_insurance.title!r}",
            conversation_id=conversation_id,
        ),
        RequestSpec(
            number="8.3",
            category="patient / conversation (turn 3/3)",
            persona="patient",
            query=faq_cancellation.title,
            expected=f"{faq_cancellation.doc_id} -- {faq_cancellation.title!r}",
            conversation_id=conversation_id,
        ),
        RequestSpec(
            number="9",
            category="eval client",
            persona="patient",
            query=faq_hours.title,
            expected=f"{faq_hours.doc_id} -- same query as request 1, sent with X-Client: eval",
            headers={"X-Client": "eval"},
        ),
    ]


def _select(batch: list[RequestSpec], skip: str | None) -> list[RequestSpec]:
    """--skip N actually means "run only item N" -- see --help."""
    if skip is None:
        return batch
    selected = [spec for spec in batch if spec.number == skip or spec.number.startswith(f"{skip}.")]
    if not selected:
        raise SystemExit(f"verify_traces: no request numbered {skip!r} (valid: 1-9)")
    return selected


# ---------------------------------------------------------------------------
# Sending requests
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    spec: RequestSpec
    status_code: int | None = None
    request_id: str | None = None
    answered: bool | None = None
    refused: bool | None = None
    refusal_reason: str | None = None
    error: str | None = None


def _send(client: httpx.Client, base_url: str, spec: RequestSpec) -> RequestResult:
    body: dict[str, object] = {"query": spec.query, "persona": spec.persona}
    if spec.conversation_id is not None:
        body["conversation_id"] = spec.conversation_id

    result = RequestResult(spec=spec)
    try:
        response = client.post(f"{base_url}/query", json=body, headers=spec.headers, timeout=30.0)
    except httpx.RequestError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.status_code = response.status_code
    try:
        payload = response.json()
    except ValueError:
        result.error = f"non-JSON response body: {response.text[:200]!r}"
        return result

    result.request_id = payload.get("request_id")
    result.answered = payload.get("answered")
    result.refused = payload.get("refused")
    result.refusal_reason = payload.get("refusal_reason")
    return result


def _trace_id_for(request_id: str) -> str:
    """Same transform tracing.py uses: a UUID4 minus its dashes is already
    a valid 32-hex-char Langfuse trace_id. Duplicated here deliberately
    (a one-line, load-bearing-only-in-the-sense-of-being-a-format-fact
    transform) rather than importing a private helper from tracing.py,
    which this script is not supposed to reach into.
    """
    return request_id.replace("-", "").lower()


def _trace_url(settings: Settings, trace_id: str) -> str:
    client = get_langfuse_client(settings)
    if client is not None:
        try:
            url = client.get_trace_url(trace_id=trace_id)
        except Exception:
            url = None
        if url:
            return url
    # No-op mode, or get_trace_url couldn't resolve a project id over the
    # network -- fall back to a plain construction from configured host.
    return f"{settings.langfuse_host.rstrip('/')}/trace/{trace_id}"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CHECKLIST = """
TRACE HIERARCHY
[ ] caredesk.query is the root, with retrieve / generate /
    verify_citations as children
[ ] embed_query and vector_search nest under retrieve, not under the root
[ ] no orphaned or flattened spans

COST AND TOKENS
[ ] generation spans show input and output token counts
[ ] cost rolls up to trace level, non-zero on resolved requests
[ ] request 7 shows cache_hit=true, zero cost on embed_query
[ ] requests 4 and 5 show no generate span at all (pre-model refusal)

CONTEXT AND FILTERING
[ ] persona visible on the trace, absent from child spans
[ ] the three requests in item 8 group as one session, ordered by time
[ ] request 9 is distinguishable as eval traffic
[ ] filtering by persona=staff returns only request 3
[ ] filtering by refused=true returns requests 4, 5, and 6

RETRIEVAL DETAIL
[ ] retrieve span shows ranked (chunk_id, score) pairs
[ ] the top chunk matches the expected document printed above
[ ] request 6 shows the persona filter applied, staff content absent

GENERATION DETAIL
[ ] generate span input contains the fully rendered prompt with context
[ ] verify_citations shows cited IDs and verification outcome
[ ] prompt_version recorded on every generation span
"""


def _print_pre(spec: RequestSpec) -> None:
    extra = f"  headers={spec.headers}" if spec.headers else ""
    print(f"[{spec.number}] {spec.category}")
    print(f"    persona: {spec.persona}")
    print(f"    query:   {spec.query!r}{extra}")
    print(f"    expects: {spec.expected}")


def _print_result(settings: Settings, result: RequestResult) -> None:
    if result.error:
        print(f"    FAILED: {result.error}")
        return
    print(f"    status:  {result.status_code}")
    print(
        f"    answered={result.answered}  refused={result.refused}  reason={result.refusal_reason}"
    )
    print(f"    request_id: {result.request_id}")
    if result.request_id:
        trace_id = _trace_id_for(result.request_id)
        print(f"    trace_id:   {trace_id}")
        print(f"    trace_url:  {_trace_url(settings, trace_id)}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Running /query server's base URL."
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait for Langfuse's background flush before printing trace URLs.",
    )
    parser.add_argument(
        "--skip",
        metavar="N",
        default=None,
        help="Run only request N (1-9) instead of the full batch. Named --skip because it "
        "skips straight to N, not because it omits N.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    manifest = load_manifest(settings.corpus_root)
    batch = _select(build_batch(manifest), args.skip)

    print(f"Sending {len(batch)} request(s) to {args.base_url} ...\n")

    results: list[RequestResult] = []
    with httpx.Client() as client:
        for spec in batch:
            _print_pre(spec)
            result = _send(client, args.base_url, spec)
            results.append(result)
            if result.error:
                print(f"    FAILED: {result.error}\n")
            else:
                print(f"    status:  {result.status_code}  (request_id={result.request_id})\n")

    print(
        f"All requests sent. Waiting {args.wait:.0f}s for Langfuse to flush "
        "before printing trace URLs ..."
    )
    print("(traces that aren't there yet look like traces that were never created)\n")
    time.sleep(args.wait)

    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    for result in results:
        _print_pre(result.spec)
        _print_result(settings, result)

    print("=" * 72)
    print("MANUAL CHECKLIST -- confirm each in the Langfuse UI")
    print("=" * 72)
    print(CHECKLIST)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
