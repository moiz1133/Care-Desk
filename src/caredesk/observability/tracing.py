"""Langfuse tracing: client lifecycle, span/generation helpers, no-op fallback.

One trace per /query request (wired up in `services.pipeline`). The span
names and attribute keys used throughout this module and its callers are
load-bearing: Week 3 builds Grafana panels off these fields, Month 5
attaches eval scores to these traces, and Month 9 exports them to
OpenTelemetry. Renaming something here later means rewriting dashboards
and losing historical comparability -- treat a rename as a breaking
change, not a refactor.

    trace: caredesk.query
      span: retrieve
        span: embed_query      (generation -- it's a model call)
        span: vector_search
      span: generate           (generation)
      span: verify_citations

Week 3's model tiering adds a `classify` span ahead of `retrieve`. Week
4's decision engine adds a `decide` span between `generate` and
`verify_citations`. Both slot in as additional children of the trace
root without restructuring anything here.

Every helper in this module is failure-isolated: a Langfuse error is
logged and swallowed, never raised into the request path. When
credentials are absent or `Settings.langfuse_enabled` is False,
`get_langfuse_client` returns None and every span/generation helper
degrades to a no-op transparently -- no caller outside this module ever
branches on whether tracing is actually active.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Literal

from langfuse import Langfuse, propagate_attributes

from caredesk.config import Settings
from caredesk.observability.context import (
    RequestContext,
    TraceState,
    bind_trace,
    get_current_request_context,
    get_current_trace,
)
from caredesk.observability.vocabulary import KNOWN_METADATA_KEYS, ClientType, TraceMetadataKey

logger = logging.getLogger(__name__)

_warned_no_op = False
_warned_no_request_context = False


@lru_cache(maxsize=1)
def _build_client(public_key: str, secret_key: str, host: str, environment: str) -> Langfuse:
    return Langfuse(
        public_key=public_key, secret_key=secret_key, host=host, environment=environment
    )


def get_langfuse_client(settings: Settings) -> Langfuse | None:
    """The process-wide Langfuse client, or None to signal no-op mode.

    None (not a fake client) is the no-op signal every helper below
    checks for. `langfuse_enabled=False`, missing credentials, and a
    failed client construction all map to it identically.
    """
    global _warned_no_op

    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        if not _warned_no_op:
            logger.warning("langfuse_no_op", extra={"reason": "missing_credentials_or_disabled"})
            _warned_no_op = True
        return None
    try:
        return _build_client(
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_host,
            settings.environment,
        )
    except Exception:
        if not _warned_no_op:
            logger.warning("langfuse_client_init_failed", exc_info=True)
            _warned_no_op = True
        return None


def _trace_id_for(request_id: str) -> str:
    """Reformat request_id into a valid Langfuse trace_id, not a derived one.

    Langfuse trace IDs must be exactly 32 lowercase hex characters. A
    UUID4 minus its dashes already is exactly that -- so this is a
    lossless, reversible reformat of request_id (re-insert the dashes at
    positions 8/12/16/20 to recover it), not a hash or a separate ID. A
    log line and a trace can be correlated with no extra lookup.
    """
    return request_id.replace("-", "").lower()


def _check_metadata_keys(metadata: Any, *, span_name: str) -> None:
    """Warn (never raise) if `metadata` carries a key outside the closed
    vocabulary in `observability.vocabulary`.

    A typo'd or ad-hoc metadata key is invisible in a filtering UI -- it
    just silently fails to match anything -- so this is the one place
    that class of mistake gets surfaced, regardless of which span it came
    from.
    """
    if not isinstance(metadata, dict):
        return
    unknown = set(metadata) - KNOWN_METADATA_KEYS
    if unknown:
        logger.warning(
            "unknown_metadata_key", extra={"span": span_name, "unknown_keys": sorted(unknown)}
        )


class SpanHandle:
    """Update-only view over an active (possibly absent) observation.

    Every method swallows Langfuse errors internally, so callers never
    need their own try/except around a `.update(...)` call.
    """

    __slots__ = ("_raw", "_name")

    def __init__(self, raw: Any, name: str) -> None:
        self._raw = raw
        self._name = name

    def update(self, **kwargs: Any) -> None:
        if "metadata" in kwargs:
            _check_metadata_keys(kwargs["metadata"], span_name=self._name)
        if self._raw is None:
            return
        try:
            self._raw.update(**kwargs)
        except Exception:
            logger.warning("langfuse_span_update_failed", extra={"span": self._name}, exc_info=True)


@contextmanager
def _observation(
    settings: Settings, name: str, *, as_type: Literal["span", "generation"], **kwargs: Any
) -> Iterator[SpanHandle]:
    if "metadata" in kwargs:
        _check_metadata_keys(kwargs["metadata"], span_name=name)
    trace = get_current_trace()
    # No bound trace (e.g. scripts/ask.py run outside any request) still
    # gets real spans -- there's no sampling context to consult, so default
    # to full detail rather than silently going dark for CLI usage.
    detail_wanted = trace is None or trace.record_detail

    client = get_langfuse_client(settings) if detail_wanted else None

    raw = None
    cm = None
    if client is not None:
        try:
            cm = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
            raw = cm.__enter__()
        except Exception:
            logger.warning("langfuse_span_start_failed", extra={"span": name}, exc_info=True)
            cm = None

    handle = SpanHandle(raw, name)
    try:
        yield handle
    except Exception as exc:
        # The caller's own work failed -- force full detail for anything
        # traced from this point on, and mark this span, but never swallow
        # the caller's exception itself.
        if trace is not None:
            trace.force()
        handle.update(level="ERROR", status_message=str(exc)[:500])
        raise
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.warning("langfuse_span_end_failed", extra={"span": name}, exc_info=True)


def start_span(settings: Settings, name: str, **kwargs: Any) -> Any:
    """Start a plain (non-generation) child span under the current trace."""
    return _observation(settings, name, as_type="span", **kwargs)


def start_generation(settings: Settings, name: str, **kwargs: Any) -> Any:
    """Start a generation-type child span (anything calling a model)."""
    return _observation(settings, name, as_type="generation", **kwargs)


class QueryTrace:
    """Handle for the root `caredesk.query` span of one /query request."""

    def __init__(self, state: TraceState, handle: SpanHandle) -> None:
        self.state = state
        self._handle = handle

    @property
    def trace_id(self) -> str:
        return self.state.trace_id

    def finalize(self, *, output: str | None, metadata: dict[str, Any]) -> None:
        """Record the pipeline's outcome on the trace root.

        Always called for real, regardless of the sampling roll -- this is
        the one write that makes "errors and refusals are always traced"
        true without buffering every span. `metadata` should carry only
        outcome keys (answered, refused, refusal_reason, results_returned,
        ...) -- persona/client/turn_index/environment are request
        invariants already written once, at trace creation, by
        `start_query_trace`; writing them again here is exactly the
        duplication this module exists to prevent, so don't.
        """
        if metadata.get("refused") or metadata.get("error"):
            self.state.force()
        self._handle.update(output=output, metadata=metadata)


def _placeholder_request_context(settings: Settings) -> RequestContext:
    """Identity for a trace opened with no bound RequestContext.

    Background jobs and one-off scripts (scripts/ask.py, scripts/search.py)
    call retrieve()/generate_answer() directly, outside any HTTP request,
    so there's nothing for api.middleware to have bound. Warned once, not
    every call -- this path is expected, not a bug, for that usage.
    """
    global _warned_no_request_context
    if not _warned_no_request_context:
        logger.warning("trace_started_without_request_context")
        _warned_no_request_context = True
    return RequestContext(
        request_id=str(uuid.uuid4()),
        environment=settings.environment,
        client=str(ClientType.CLI),
        persona="unknown",
        conversation_id="unknown",
    )


@contextmanager
def start_query_trace(settings: Settings, *, query: str) -> Iterator[QueryTrace]:
    """Open the root `caredesk.query` span for one /query request.

    Identity (request_id, persona, conversation_id, client, turn_index,
    environment) comes entirely from the ambient `RequestContext` --
    established once by `api.middleware`, enriched once by the route --
    never as parameters passed in here. That is what makes "every span
    inherits persona/conversation_id/client automatically, no extra
    wiring" true by construction: this function is the only place that
    ever translates identity into Langfuse's tag/session/metadata
    mechanisms, and it does so once, from one source.

    Unlike `start_span`/`start_generation`, the root is always created for
    real whenever tracing is enabled at all -- sampling only gates the
    detailed child spans beneath it (`TraceState.record_detail`). A
    sampled-out trace still shows up with its final answered/refused
    outcome; only the retrieve/generate/verify detail underneath is
    thinner.
    """
    request_context = get_current_request_context() or _placeholder_request_context(settings)
    request_id = request_context.request_id
    persona = request_context.persona or "unknown"
    conversation_id = request_context.conversation_id or "unknown"
    client = request_context.client

    sampled = random.random() < settings.trace_sample_rate
    trace_id = _trace_id_for(request_id)
    state = TraceState(trace_id=trace_id, sampled=sampled)

    langfuse_client = get_langfuse_client(settings)

    root_cm = None
    prop_cm = None
    raw_root = None
    if langfuse_client is not None:
        try:
            root_cm = langfuse_client.start_as_current_observation(
                name="caredesk.query",
                trace_context={"trace_id": trace_id},
                input=query,
                metadata={
                    TraceMetadataKey.PERSONA: persona,
                    TraceMetadataKey.CLIENT: client,
                    TraceMetadataKey.TURN_INDEX: request_context.turn_index,
                    TraceMetadataKey.ENVIRONMENT: request_context.environment,
                },
            )
            raw_root = root_cm.__enter__()
        except Exception:
            logger.warning(
                "langfuse_trace_start_failed", extra={"request_id": request_id}, exc_info=True
            )
            root_cm = None

        if raw_root is not None:
            try:
                # persona and client are the two closed-set dimensions worth
                # filtering the trace LIST on directly; conversation_id maps
                # to session_id (not a tag -- too high-cardinality to filter
                # by value, but exactly what groups a conversation's traces).
                prop_cm = propagate_attributes(
                    user_id=conversation_id,
                    session_id=conversation_id,
                    trace_name="caredesk.query",
                    environment=request_context.environment,
                    tags=[persona, client],
                )
                prop_cm.__enter__()
            except Exception:
                logger.warning(
                    "langfuse_propagate_failed", extra={"request_id": request_id}, exc_info=True
                )
                prop_cm = None

    handle = SpanHandle(raw_root, "caredesk.query")
    trace = QueryTrace(state, handle)

    with bind_trace(state):
        try:
            yield trace
        except Exception as exc:
            state.force()
            handle.update(level="ERROR", output=f"{type(exc).__name__}: {exc}"[:500])
            raise
        finally:
            if prop_cm is not None:
                try:
                    prop_cm.__exit__(None, None, None)
                except Exception:
                    logger.warning("langfuse_propagate_end_failed", exc_info=True)
            if root_cm is not None:
                try:
                    root_cm.__exit__(None, None, None)
                except Exception:
                    logger.warning("langfuse_trace_end_failed", exc_info=True)


async def shutdown(settings: Settings) -> None:
    """Flush buffered spans and shut the client down, bounded by a timeout.

    Called once from the API lifespan on application shutdown. Never
    raises -- a slow or unreachable Langfuse backend must not block or
    fail process shutdown.
    """
    client = get_langfuse_client(settings)
    if client is None:
        return
    try:
        await asyncio.wait_for(
            asyncio.to_thread(client.shutdown), timeout=settings.trace_flush_timeout_seconds
        )
    except Exception:
        logger.warning("langfuse_shutdown_failed", exc_info=True)
