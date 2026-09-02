"""Closed vocabulary for trace tags and metadata keys.

Every tag value and metadata key that reaches Langfuse must come from
here. A typo'd tag or an ad-hoc metadata key at a call site is invisible
in a filtering UI -- it just silently fails to match anything -- so this
module exists to make that class of mistake a code-review-visible import
instead of a free-form string. `tracing.py` validates every metadata dict
against `TRACE_METADATA_KEYS | SPAN_METADATA_KEYS` before it reaches
Langfuse (see `_check_metadata_keys`), so an unlisted key is caught at
run time, not just by convention.

Two levels, matching where each key is actually written:

- Trace-level: invariant-for-the-request identity (persona, client,
  turn_index, environment) is written once, at trace creation, by
  `start_query_trace`. Outcome keys (answered, refused, results_returned,
  ...) are written once, at trace finalize, by `QueryTrace.finalize`. No
  key is ever written from both places -- that split is what prevents the
  persona duplication this commit was written to fix.
- Span-level: per-span detail (cache_hit, persona_filter_applied, ...).
  Never persona/conversation_id/client/environment -- those are trace
  invariants and every span inherits them automatically via Langfuse's
  `propagate_attributes`, not by being set per span.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ClientType(StrEnum):
    """Closed set for `RequestContext.client` / the `X-Client` header.

    Distinguishing these as a trace tag is the whole point of the field:
    it is what lets a cost/latency dashboard exclude eval traffic without
    reading logs.
    """

    API = "api"
    EVAL = "eval"
    CLI = "cli"


class TraceMetadataKey(StrEnum):
    """Keys written to the trace root's metadata dict, and nowhere else."""

    # Written once, at trace creation -- invariant for the whole request.
    PERSONA = "persona"
    CLIENT = "client"
    TURN_INDEX = "turn_index"
    ENVIRONMENT = "environment"

    # Written once, at trace finalize -- the pipeline's outcome.
    K = "k"
    ANSWERED = "answered"
    REFUSED = "refused"
    REFUSAL_REASON = "refusal_reason"
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    RESULTS_RETURNED = "results_returned"
    PROMPT_VERSION = "prompt_version"
    CITATION_COUNT = "citation_count"
    TOTAL_LATENCY_MS = "total_latency_ms"


class SpanMetadataKey(StrEnum):
    """Keys written to individual span metadata dicts."""

    STRATEGY = "strategy"
    SOURCE_TYPE_FILTER = "source_type_filter"
    CACHE_HIT = "cache_hit"
    LATENCY_MS = "latency_ms"
    K = "k"
    PERSONA_FILTER_APPLIED = "persona_filter_applied"
    PERSONA_FILTER_EXCLUDED_COUNT = "persona_filter_excluded_count"


TRACE_METADATA_KEYS: Final[frozenset[str]] = frozenset(TraceMetadataKey)
SPAN_METADATA_KEYS: Final[frozenset[str]] = frozenset(SpanMetadataKey)
KNOWN_METADATA_KEYS: Final[frozenset[str]] = TRACE_METADATA_KEYS | SPAN_METADATA_KEYS
