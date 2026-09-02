"""Tests for caredesk.observability.tracing (and context.py's propagation).

No real Langfuse client anywhere -- `get_langfuse_client` and the
module-level `propagate_attributes` are swapped for fakes that record
calls and reproduce real nesting semantics (a span's parent is whatever
observation was current, per-task, at the moment it was created), so
hierarchy and attribute assertions reflect what a real client would
actually see.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any

import pytest

from caredesk.config import Settings
from caredesk.observability import tracing
from caredesk.observability.context import get_current_trace

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeObservation:
    """Stands in for a LangfuseSpan/LangfuseGeneration: itself the context
    manager `start_as_current_observation` returns, and the object
    `.update()` is called on after `__enter__`.

    "Current observation" is tracked via a contextvar on the client, not a
    plain shared stack -- a plain stack would let sibling coroutines under
    `asyncio.gather` see each other's spans as parents (each gets its own
    copy of the contextvar's value at task-creation time instead), which
    is what real OTel/Langfuse context propagation also relies on.
    """

    def __init__(
        self, client: FakeLangfuseClient, name: str, as_type: str, kwargs: dict[str, Any]
    ) -> None:
        self.client = client
        self.name = name
        self.as_type = as_type
        self.creation_kwargs = kwargs
        self.parent: FakeObservation | None = client._current.get()
        self.updates: list[dict[str, Any]] = []
        self._token: Any = None

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def __enter__(self) -> FakeObservation:
        self._token = self.client._current.set(self)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.client._current.reset(self._token)
        return False

    @property
    def last_update(self) -> dict[str, Any]:
        return self.updates[-1] if self.updates else {}


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.calls: list[FakeObservation] = []
        self._current: ContextVar[FakeObservation | None] = ContextVar(
            "fake_current_observation", default=None
        )
        self.shutdown_called = False

    def start_as_current_observation(
        self, *, name: str, as_type: str = "span", **kwargs: Any
    ) -> FakeObservation:
        obs = FakeObservation(self, name, as_type, kwargs)
        self.calls.append(obs)
        return obs

    def shutdown(self) -> None:
        self.shutdown_called = True

    def flush(self) -> None:
        pass


class _RaisingClient:
    """A client whose start_as_current_observation always blows up."""

    def start_as_current_observation(self, **kwargs: Any) -> Any:
        raise RuntimeError("langfuse is down")

    def shutdown(self) -> None:
        raise RuntimeError("langfuse is down")


class _FakePropagateCM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> _FakePropagateCM:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _fake_propagate_attributes(**kwargs: Any) -> _FakePropagateCM:
    return _FakePropagateCM(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_enabled": True,
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _use_fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeLangfuseClient:
    client = FakeLangfuseClient()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda settings: client)
    monkeypatch.setattr(tracing, "propagate_attributes", _fake_propagate_attributes)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_successful_query_produces_expected_span_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    with tracing.start_query_trace(
        settings,
        request_id="11111111-1111-4111-8111-111111111111",
        conversation_id="c1",
        query="q",
        persona="patient",
    ) as trace:
        with tracing.start_span(settings, "retrieve"):
            with tracing.start_generation(settings, "embed_query", model="text-embedding-3-small"):
                pass
            with tracing.start_span(settings, "vector_search"):
                pass
        with tracing.start_generation(settings, "generate", model="gpt-4o"):
            pass
        with tracing.start_span(settings, "verify_citations"):
            pass

        trace.finalize(
            output="answer", tags=["patient", "vector_only", "v1"], metadata={"refused": False}
        )

    names_and_parents = [
        (obs.name, obs.parent.name if obs.parent else None) for obs in client.calls
    ]
    assert names_and_parents == [
        ("caredesk.query", None),
        ("retrieve", "caredesk.query"),
        ("embed_query", "retrieve"),
        ("vector_search", "retrieve"),
        ("generate", "caredesk.query"),
        ("verify_citations", "caredesk.query"),
    ]


def test_trace_id_is_request_id_reformatted_not_a_separate_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_fake_client(monkeypatch)
    settings = _settings()
    request_id = "618f734e-f168-486d-bff3-700740b1cc38"

    with tracing.start_query_trace(
        settings, request_id=request_id, conversation_id="c1", query="q", persona="patient"
    ) as trace:
        pass

    assert trace.trace_id == request_id.replace("-", "")
    # Lossless: dashes reinserted at 8/12/16/20 recover the original UUID.
    recovered = "-".join(
        [
            trace.trace_id[0:8],
            trace.trace_id[8:12],
            trace.trace_id[12:16],
            trace.trace_id[16:20],
            trace.trace_id[20:],
        ]
    )
    assert recovered == request_id


def test_embed_cache_hit_emits_span_with_zero_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    with tracing.start_generation(settings, "embed_query", model="text-embedding-3-small") as span:
        span.update(
            metadata={"cache_hit": True, "latency_ms": 0.4},
            usage_details={"input": 12},
            cost_details={"total": 0.0},
        )

    obs = client.calls[0]
    assert obs.last_update["metadata"]["cache_hit"] is True
    assert obs.last_update["cost_details"]["total"] == 0.0


def test_generation_span_carries_token_counts_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    with tracing.start_generation(settings, "generate", model="gpt-4o") as span:
        span.update(
            output="answer",
            usage_details={"input": 993, "output": 76},
            cost_details={"total": 0.0032},
        )

    obs = client.calls[0]
    assert obs.last_update["usage_details"] == {"input": 993, "output": 76}
    assert obs.last_update["cost_details"] == {"total": 0.0032}


def test_hallucinated_citation_marks_verify_span_error_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    with tracing.start_span(
        settings, "verify_citations", input={"cited_ids": ["a::0", "fake::0"]}
    ) as span:
        span.update(
            level="ERROR",
            output={
                "verified_ids": ["a::0"],
                "hallucinated_ids": ["fake::0"],
                "outcome": "hallucinated_citation",
            },
        )

    obs = client.calls[0]
    assert obs.last_update["level"] == "ERROR"
    assert obs.last_update["output"]["hallucinated_ids"] == ["fake::0"]


def test_refusal_is_traced_with_refusal_reason_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    with tracing.start_query_trace(
        settings,
        request_id="22222222-2222-4222-8222-222222222222",
        conversation_id="c1",
        query="q",
        persona="patient",
    ) as trace:
        trace.finalize(
            output="no_results",
            tags=["patient", "vector_only", "v1"],
            metadata={"refused": True, "refusal_reason": "no_results", "answered": False},
        )

    root = client.calls[0]
    assert root.name == "caredesk.query"
    assert root.last_update["metadata"]["refusal_reason"] == "no_results"
    # A refusal forces full detail for anything traced after it's known.
    assert trace.state.forced is True


def test_langfuse_raising_does_not_fail_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda settings: _RaisingClient())
    settings = _settings()

    ran = False
    with tracing.start_span(settings, "retrieve") as span:
        ran = True
        span.update(output="fine")  # must not raise even though span creation failed

    assert ran is True


def test_missing_credentials_runs_in_no_op_mode_without_error() -> None:
    settings = _settings(langfuse_public_key=None, langfuse_secret_key=None)

    assert tracing.get_langfuse_client(settings) is None

    ran = False
    with tracing.start_span(settings, "retrieve") as span:
        ran = True
        span.update(output="fine")

    assert ran is True


async def test_span_nesting_survives_an_await_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    with tracing.start_query_trace(
        settings,
        request_id="33333333-3333-4333-8333-333333333333",
        conversation_id="c1",
        query="q",
        persona="patient",
    ) as trace:
        with tracing.start_span(settings, "retrieve"):
            await asyncio.sleep(0)  # real await boundary
            with tracing.start_span(settings, "vector_search"):
                pass
        trace.finalize(output="answer", tags=[], metadata={"refused": False})

    names_and_parents = {
        obs.name: (obs.parent.name if obs.parent else None) for obs in client.calls
    }
    assert names_and_parents["retrieve"] == "caredesk.query"
    assert names_and_parents["vector_search"] == "retrieve"


async def test_gather_keeps_concurrent_spans_parented_to_the_same_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    async def child(name: str) -> None:
        await asyncio.sleep(0)
        with tracing.start_span(settings, name):
            await asyncio.sleep(0)

    with tracing.start_span(settings, "retrieve"):
        await asyncio.gather(child("a"), child("b"))

    parents = {obs.name: (obs.parent.name if obs.parent else None) for obs in client.calls}
    assert parents["a"] == "retrieve"
    assert parents["b"] == "retrieve"


def test_sample_rate_zero_still_traces_an_errored_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fake_client(monkeypatch)
    monkeypatch.setattr(tracing.random, "random", lambda: 0.999)  # would-be "not sampled"
    settings = _settings(trace_sample_rate=0.0)

    with (
        pytest.raises(ValueError, match="boom"),
        tracing.start_query_trace(
            settings,
            request_id="44444444-4444-4444-8444-444444444444",
            conversation_id="c1",
            query="q",
            persona="patient",
        ) as trace,
    ):
        assert trace.state.sampled is False
        raise ValueError("boom")

    # The root is still recorded for real, error and all, despite sample_rate 0.0.
    root = client.calls[0]
    assert root.name == "caredesk.query"
    assert root.last_update["level"] == "ERROR"
    assert "boom" in root.last_update["output"]


def test_sample_rate_zero_skips_child_span_detail_on_the_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _use_fake_client(monkeypatch)
    monkeypatch.setattr(tracing.random, "random", lambda: 0.999)
    settings = _settings(trace_sample_rate=0.0)

    with tracing.start_query_trace(
        settings,
        request_id="55555555-5555-4555-8555-555555555555",
        conversation_id="c1",
        query="q",
        persona="patient",
    ) as trace:
        assert trace.state.sampled is False
        with tracing.start_span(settings, "retrieve"):
            pass  # a normal, uneventful retrieval -- never forces detail
        trace.finalize(output="answer", tags=[], metadata={"refused": False})

    # Root is always real; the child never was (sampled out, never forced).
    names = [obs.name for obs in client.calls]
    assert names == ["caredesk.query"]


def test_trace_context_not_bound_outside_a_traced_request() -> None:
    assert get_current_trace() is None
