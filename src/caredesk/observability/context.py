"""Contextvars carrying trace identity across the async call graph.

Kept separate from tracing.py (which owns the Langfuse client and span
creation) for two reasons: `retrieve()` and `generate_answer()` can read
what trace they're part of -- and contribute to it -- without importing
the client itself, and the propagation behaviour here can be unit tested
without a real Langfuse dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class TraceState:
    """Mutable identity of the current trace, bound once per /query request.

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

    __slots__ = ("request_id", "conversation_id", "trace_id", "sampled", "forced")

    def __init__(
        self, *, request_id: str, conversation_id: str, trace_id: str, sampled: bool
    ) -> None:
        self.request_id = request_id
        self.conversation_id = conversation_id
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
