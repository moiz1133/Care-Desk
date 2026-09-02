"""Tests for commit 9: RequestContext, the closed tag/metadata vocabulary,
and the trace-level identity start_query_trace builds from it.

test_tracing.py already covers span hierarchy, sampling, and failure
isolation (commit 8) -- this file is scoped to what commit 9 actually
changed: where identity comes from, that it's written exactly once per
level, and that an unbound context degrades gracefully instead of
raising.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi import Request
from fastapi.datastructures import Headers

from caredesk.api.middleware import _resolve_client, set_query_identity
from caredesk.config import Settings
from caredesk.observability import tracing
from caredesk.observability.context import (
    RequestContext,
    bind_request_context,
    get_current_request_context,
)
from caredesk.observability.vocabulary import ClientType

# ---------------------------------------------------------------------------
# Fakes (same shape as test_tracing.py's -- kept local so this file doesn't
# depend on another test module's internals)
# ---------------------------------------------------------------------------


class _FakeObservation:
    def __init__(self, name: str, kwargs: dict[str, Any]) -> None:
        self.name = name
        self.creation_kwargs = kwargs
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def __enter__(self) -> _FakeObservation:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[_FakeObservation] = []
        self.propagate_calls: list[dict[str, Any]] = []

    def start_as_current_observation(self, *, name: str, **kwargs: Any) -> _FakeObservation:
        obs = _FakeObservation(name, kwargs)
        self.observations.append(obs)
        return obs


class _FakePropagateCM:
    def __init__(self, calls: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        calls.append(kwargs)

    def __enter__(self) -> _FakePropagateCM:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_enabled": True,
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _use_fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeLangfuseClient:
    client = _FakeLangfuseClient()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda settings: client)
    monkeypatch.setattr(
        tracing,
        "propagate_attributes",
        lambda **kwargs: _FakePropagateCM(client.propagate_calls, kwargs),
    )
    return client


def _headers(**raw: str) -> Headers:
    return Headers({k.replace("_", "-"): v for k, v in raw.items()})


# ---------------------------------------------------------------------------
# RequestContext binding and enrichment
# ---------------------------------------------------------------------------


def test_bind_request_context_is_readable_and_resets_on_exit() -> None:
    assert get_current_request_context() is None
    context = RequestContext(request_id="r1", environment="dev", client="api")

    with bind_request_context(context):
        assert get_current_request_context() is context

    assert get_current_request_context() is None


def test_set_query_identity_enriches_the_bound_context_in_place() -> None:
    context = RequestContext(request_id="r1", environment="dev", client="api")
    with bind_request_context(context):
        assert context.persona is None
        assert context.conversation_id is None

        set_query_identity(persona="staff", conversation_id="conv-1")

        # Same object, not a new one -- "no component constructs its own".
        bound = get_current_request_context()
        assert bound is context
        assert bound.persona == "staff"
        assert bound.conversation_id == "conv-1"


def test_set_query_identity_without_bound_context_warns_and_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert get_current_request_context() is None
    with caplog.at_level(logging.WARNING, logger="caredesk.api.middleware"):
        set_query_identity(persona="patient", conversation_id="conv-1")  # must not raise
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# X-Client header resolution (closed set)
# ---------------------------------------------------------------------------


def _request_with_headers(headers: Headers) -> Request:
    scope = {"type": "http", "headers": headers.raw, "method": "GET", "path": "/"}
    return Request(scope)


def test_resolve_client_defaults_to_api_when_header_absent() -> None:
    assert _resolve_client(_request_with_headers(_headers())) == "api"


def test_resolve_client_accepts_known_values() -> None:
    assert _resolve_client(_request_with_headers(_headers(x_client="eval"))) == "eval"
    assert _resolve_client(_request_with_headers(_headers(x_client="cli"))) == "cli"


def test_resolve_client_falls_back_to_api_on_unknown_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="caredesk.api.middleware"):
        result = _resolve_client(_request_with_headers(_headers(x_client="typo-value")))
    assert result == "api"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# start_query_trace reading identity from RequestContext
# ---------------------------------------------------------------------------


def test_persona_and_client_are_tags_and_conversation_id_is_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()
    context = RequestContext(
        request_id="11111111-1111-4111-8111-111111111111",
        environment="dev",
        client=str(ClientType.EVAL),
        persona="staff",
        conversation_id="conv-42",
    )

    with bind_request_context(context), tracing.start_query_trace(settings, query="q"):
        pass

    assert len(client.propagate_calls) == 1
    propagated = client.propagate_calls[0]
    assert propagated["tags"] == ["staff", "eval"]
    assert propagated["session_id"] == "conv-42"
    assert propagated["user_id"] == "conv-42"


def test_persona_client_turn_index_environment_are_trace_metadata_written_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()
    context = RequestContext(
        request_id="22222222-2222-4222-8222-222222222222",
        environment="staging",
        client=str(ClientType.API),
        persona="patient",
        conversation_id="conv-1",
        turn_index=3,
    )

    with bind_request_context(context), tracing.start_query_trace(settings, query="q") as trace:
        trace.finalize(output="answer", metadata={"refused": False})

    root = client.observations[0]
    assert root.name == "caredesk.query"
    creation_metadata = root.creation_kwargs["metadata"]
    assert creation_metadata == {
        "persona": "patient",
        "client": "api",
        "turn_index": 3,
        "environment": "staging",
    }
    # finalize()'s metadata is the outcome only -- persona must not
    # reappear here, which is exactly the duplication commit 9 fixed.
    finalize_metadata = root.updates[-1]["metadata"]
    assert "persona" not in finalize_metadata
    assert "client" not in finalize_metadata
    assert finalize_metadata["refused"] is False


def test_trace_started_without_request_context_uses_placeholders_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _use_fake_client(monkeypatch)
    settings = _settings()

    assert get_current_request_context() is None
    with (
        caplog.at_level(logging.WARNING, logger="caredesk.observability.tracing"),
        tracing.start_query_trace(settings, query="q") as trace,
    ):
        pass

    assert trace.trace_id  # still produced a usable trace_id
    assert client.observations  # still emitted a real root span
    assert any(r.levelno == logging.WARNING for r in caplog.records)
