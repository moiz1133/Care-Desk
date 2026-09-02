"""Tests for POST /query.

The retrieval and generation services are mocked at the
`caredesk.services.pipeline` import boundary -- no real DB or OpenAI call
happens here. Those layers are covered directly by
`tests/retrieval/test_vector.py` and `tests/generation/test_generator.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError

from caredesk.api.dependencies import get_settings_dependency
from caredesk.api.main import app
from caredesk.config import Settings
from caredesk.generation.generator import GeneratorError
from caredesk.generation.types import Citation, GeneratedAnswer
from caredesk.retrieval.types import RetrievalResponse, RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _retrieval_result(**overrides: object) -> RetrievalResult:
    base: dict[str, object] = {
        "chunk_id": "doc1::0000",
        "doc_id": "doc1",
        "text": "chunk text",
        "score": 0.9,
        "rank": 1,
        "source_type": "faq_markdown",
        "persona_visibility": "patient",
        "title": "Title",
        "filename": "doc1.txt",
        "chunk_index": 0,
        "token_count": 10,
    }
    base.update(overrides)
    return RetrievalResult(**base)  # type: ignore[arg-type]


def _retrieval_response(
    results: list[RetrievalResult], *, persona: str = "patient"
) -> RetrievalResponse:
    return RetrievalResponse(
        query="test query",
        persona=persona,
        results=results,
        query_embedding_tokens=5,
        latency_ms=12.0,
        strategy="vector_only",
    )


def _answer(**overrides: object) -> GeneratedAnswer:
    default_citation = Citation(
        chunk_id="doc1::0000", doc_id="doc1", title="Title", filename="doc1.txt"
    )
    base: dict[str, object] = {
        "answer_text": "Answer [doc1::0000].",
        "refused": False,
        "refusal_reason": None,
        "citations": [default_citation],
        "model": "gpt-4o",
        "prompt_version": "v1",
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": 0.001,
        "latency_ms": 250.0,
    }
    base.update(overrides)
    return GeneratedAnswer(**base)  # type: ignore[arg-type]


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retrieval: RetrievalResponse | None = None,
    answer: GeneratedAnswer | None = None,
    retrieve_error: Exception | None = None,
    generate_error: Exception | None = None,
) -> dict[str, object]:
    """Stub the retrieve/generate_answer calls `services.pipeline` makes.

    Returns a dict the fakes write their received arguments into, so tests
    can assert on what the pipeline passed downstream (e.g. persona).
    """
    captured: dict[str, object] = {}

    async def fake_retrieve(
        query: str, persona: str, settings: Settings, *, k: int | None = None, **_: object
    ) -> RetrievalResponse:
        captured["query"] = query
        captured["persona"] = persona
        captured["k"] = k
        if retrieve_error is not None:
            raise retrieve_error
        assert retrieval is not None
        return retrieval

    async def fake_generate_answer(
        query: str, persona: str, retrieval_response: RetrievalResponse, settings: Settings
    ) -> GeneratedAnswer:
        if generate_error is not None:
            raise generate_error
        assert answer is not None
        return answer

    monkeypatch.setattr("caredesk.services.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("caredesk.services.pipeline.generate_answer", fake_generate_answer)
    return captured


@pytest.fixture
def use_settings() -> AsyncIterator[type]:
    class _Setter:
        @staticmethod
        def set(settings: Settings) -> None:
            app.dependency_overrides[get_settings_dependency] = lambda: settings

    yield _Setter
    app.dependency_overrides.pop(get_settings_dependency, None)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_valid_request_returns_expected_shape(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch, retrieval=_retrieval_response([_retrieval_result()]), answer=_answer()
    )

    response = await client.post(
        "/query", json={"query": "What are your hours?", "persona": "patient"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["refused"] is False
    assert body["refusal_reason"] is None
    assert body["answer"] == "Answer [doc1::0000]."
    assert body["citations"] == [
        {"chunk_id": "doc1::0000", "doc_id": "doc1", "title": "Title", "filename": "doc1.txt"}
    ]
    assert body["retrieval"]["strategy"] == "vector_only"
    assert body["retrieval"]["results_returned"] == 1
    assert body["retrieval"]["top_score"] == 0.9
    assert body["generation"]["model"] == "gpt-4o"
    assert body["generation"]["prompt_version"] == "v1"
    assert body["query"] == "What are your hours?"
    assert body["persona"] == "patient"
    assert body["context"] is None
    assert isinstance(body["total_latency_ms"], float)


async def test_missing_persona_returns_422(client: AsyncClient) -> None:
    response = await client.post("/query", json={"query": "hi"})
    assert response.status_code == 422


async def test_invalid_persona_value_returns_422(client: AsyncClient) -> None:
    response = await client.post("/query", json={"query": "hi", "persona": "doctor"})
    assert response.status_code == 422


async def test_whitespace_only_query_returns_422(client: AsyncClient) -> None:
    response = await client.post("/query", json={"query": "   ", "persona": "patient"})
    assert response.status_code == 422


async def test_empty_query_returns_422(client: AsyncClient) -> None:
    response = await client.post("/query", json={"query": "", "persona": "patient"})
    assert response.status_code == 422


async def test_query_over_length_limit_returns_422(client: AsyncClient) -> None:
    response = await client.post("/query", json={"query": "a" * 2001, "persona": "patient"})
    assert response.status_code == 422


async def test_k_out_of_range_returns_422(client: AsyncClient) -> None:
    too_low = await client.post("/query", json={"query": "hi", "persona": "patient", "k": 0})
    too_high = await client.post("/query", json={"query": "hi", "persona": "patient", "k": 21})
    assert too_low.status_code == 422
    assert too_high.status_code == 422


async def test_refusal_returns_200_not_error(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch,
        retrieval=_retrieval_response([]),
        answer=_answer(answer_text=None, refused=True, refusal_reason="no_results", citations=[]),
    )

    response = await client.post(
        "/query", json={"query": "unrelated question", "persona": "patient"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answered"] is False
    assert body["refusal_reason"] == "no_results"
    assert body["answer"] is None
    assert body["citations"] == []


async def test_generation_failure_returns_503_without_leaking_provider_error(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch,
        retrieval=_retrieval_response([_retrieval_result()]),
        generate_error=GeneratorError(
            "Generation call failed (AuthenticationError): sk-live-secret-do-not-leak"
        ),
    )

    response = await client.post("/query", json={"query": "hi", "persona": "patient"})

    assert response.status_code == 503
    assert "sk-live-secret-do-not-leak" not in response.text
    assert "AuthenticationError" not in response.text
    body = response.json()
    assert "request_id" in body
    assert response.headers.get("Retry-After") is not None


async def test_database_unavailable_returns_503(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch,
        retrieve_error=OperationalError("SELECT 1", {}, Exception("connection refused")),
    )

    response = await client.post("/query", json={"query": "hi", "persona": "patient"})

    assert response.status_code == 503


async def test_pipeline_timeout_returns_503(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings(api_request_timeout_seconds=0.01))

    async def slow_retrieve(
        query: str, persona: str, settings: Settings, *, k: int | None = None, **_: object
    ) -> RetrievalResponse:
        await asyncio.sleep(1)
        return _retrieval_response([])

    monkeypatch.setattr("caredesk.services.pipeline.retrieve", slow_retrieve)

    response = await client.post("/query", json={"query": "hi", "persona": "patient"})

    assert response.status_code == 503


async def test_request_id_present_in_body_and_header(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch, retrieval=_retrieval_response([_retrieval_result()]), answer=_answer()
    )

    response = await client.post("/query", json={"query": "hi", "persona": "patient"})

    body = response.json()
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


async def test_conversation_id_echoed_when_supplied(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch, retrieval=_retrieval_response([_retrieval_result()]), answer=_answer()
    )
    conversation_id = str(uuid4())

    response = await client.post(
        "/query",
        json={"query": "hi", "persona": "patient", "conversation_id": conversation_id},
    )

    assert response.json()["conversation_id"] == conversation_id


async def test_conversation_id_generated_when_absent(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch, retrieval=_retrieval_response([_retrieval_result()]), answer=_answer()
    )

    response = await client.post("/query", json={"query": "hi", "persona": "patient"})

    conversation_id = response.json()["conversation_id"]
    assert conversation_id
    UUID(conversation_id)  # raises if not a valid UUID


async def test_include_context_adds_chunk_text_absent_by_default(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    use_settings.set(_settings())
    _patch_pipeline(
        monkeypatch,
        retrieval=_retrieval_response([_retrieval_result(text="the retrieved chunk text")]),
        answer=_answer(),
    )

    default_response = await client.post("/query", json={"query": "hi", "persona": "patient"})
    assert default_response.json()["context"] is None

    context_response = await client.post(
        "/query",
        json={"query": "hi", "persona": "patient"},
        params={"include_context": "true"},
    )
    context = context_response.json()["context"]
    assert context is not None
    assert context[0]["text"] == "the retrieved chunk text"
    assert context[0]["chunk_id"] == "doc1::0000"


async def test_persona_passed_through_to_retriever_unchanged(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, use_settings: type
) -> None:
    """Boundary test: the security filter lives in the retriever, but the
    route is where persona could get dropped or defaulted on the way
    there. This catches that regression class specifically."""
    use_settings.set(_settings())
    captured = _patch_pipeline(
        monkeypatch,
        retrieval=_retrieval_response(
            [_retrieval_result(persona_visibility="staff")], persona="staff"
        ),
        answer=_answer(),
    )

    await client.post("/query", json={"query": "internal runbook question", "persona": "staff"})

    assert captured["persona"] == "staff"
