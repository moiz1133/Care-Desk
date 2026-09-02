"""Contextvars carrying request identity and trace state across the async
call graph.

Kept separate from tracing.py (which owns the Langfuse client and span
creation) for two reasons: `retrieve()` and `generate_answer()` can read
what trace they're part of -- and contribute to it -- without importing
the client itself, and the propagation behaviour here can be unit tested
without a real Langfuse dependency.

Two distinct contextvars live here, for two distinct concerns:

- `RequestContext` is business identity (persona, conversation_id,
  client, ...): established once by `api.middleware`, read by
  `tracing.start_query_trace` to build the trace's tags/metadata, and
  never constructed a second time by anything downstream.
- `TraceState` is Langfuse-specific bookkeeping (trace_id, the sampling
  decision, whether an error/refusal has forced full detail). It doesn't
  duplicate `RequestContext`'s fields -- `start_query_trace` reads
  identity from `RequestContext` once and derives `trace_id` from it;
  `TraceState` only tracks what's uniquely its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class RequestContext:
    """Request-scoped identity, established once per request and read
    everywhere downstream instead of being threaded through as parameters.

    Mutable, not frozen: `persona` and `conversation_id` are only known
    once the request body is validated, which happens inside the route --
    after the HTTP middleware that creates this object has already run.
    The route enriches this *same* instance in place (see
    `api.middleware.set_query_identity`) rather than constructing a
    competing one, which is what keeps "established once" true even
    though it's populated in two steps.
    """

    request_id: str
    environment: str
    client: str
    persona: str | None = None
    conversation_id: str | None = None
    turn_index: int = 0


_current_request_context: ContextVar[RequestContext | None] = ContextVar(
    "current_request_context", default=None
)


def get_current_request_context() -> RequestContext | None:
    """The active RequestContext, or None outside of a traced request (e.g. CLI scripts)."""
    return _current_request_context.get()


@contextmanager
def bind_request_context(context: RequestContext) -> Iterator[RequestContext]:
    """Bind `context` as the current request identity for the duration of the block.

    Reset on exit -- including on exception -- so the contextvar can't
    leak into a later request that reuses the same worker/task.
    """
    token = _current_request_context.set(context)
    try:
        yield context
    finally:
        _current_request_context.reset(token)


class TraceState:
    """Mutable Langfuse-tracing state for the current trace, bound once
    per /query request.

    `sampled` is decided once, at request start, from
    `Settings.trace_sample_rate`. `forced` starts False and can flip to
    True mid-request the moment an error or refusal becomes known --
    letting any span created *after* that point still get full detail
    even under a sample rate that would otherwise have skipped it. Spans
    already created before the flip aren't retroactively upgraded; only
    the trace root's own final record is unconditional (see
    `tracing.start_query_trace`), which is what actually makes "errors
    and refusals are always traced" hold without buffering every span
    until the outcome is known.
    """

    __slots__ = ("trace_id", "sampled", "forced")

    def __init__(self, *, trace_id: str, sampled: bool) -> None:
        self.trace_id = trace_id
        self.sampled = sampled
        self.forced = False

    @property
    def record_detail(self) -> bool:
        return self.sampled or self.forced

    def force(self) -> None:
        self.forced = True


_current_trace: ContextVar[TraceState | None] = ContextVar("current_trace", default=None)


def get_current_trace() -> TraceState | None:
    """The active TraceState, or None outside of a traced request (e.g. CLI scripts)."""
    return _current_trace.get()


@contextmanager
def bind_trace(state: TraceState) -> Iterator[TraceState]:
    """Bind `state` as the current trace for the duration of the block.

    Reset on exit -- including on exception -- so the contextvar can't
    leak into a later request that reuses the same worker/task.
    """
    token = _current_trace.set(state)
    try:
        yield state
    finally:
        _current_trace.reset(token)
