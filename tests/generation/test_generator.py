"""Tests for caredesk.generation.generator.

No real API calls — the OpenAI client is always stubbed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from caredesk.config import Settings
from caredesk.generation.generator import generate_answer
from caredesk.generation.prompts import PROMPT_VERSION, REFUSAL_SENTINEL
from caredesk.retrieval.types import RetrievalResponse, RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_public_key": "test-pub",
        "langfuse_secret_key": "test-secret",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _result(
    chunk_id: str,
    *,
    doc_id: str = "doc1",
    score: float = 0.9,
    rank: int = 1,
    source_type: str = "faq_markdown",
    persona_visibility: str = "patient",
    title: str = "Title",
    filename: str = "doc1.txt",
    chunk_index: int = 0,
    text: str = "chunk text",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        score=score,
        rank=rank,
        source_type=source_type,
        persona_visibility=persona_visibility,  # type: ignore[arg-type]
        title=title,
        filename=filename,
        chunk_index=chunk_index,
        token_count=10,
    )


def _retrieval(results: list[RetrievalResult]) -> RetrievalResponse:
    return RetrievalResponse(
        query="test query",
        persona="patient",
        results=results,
        query_embedding_tokens=5,
        latency_ms=10.0,
        strategy="vector_only",
    )


def _patch_openai(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
) -> list[dict[str, object]]:
    """Stub AsyncOpenAI so generate_answer never makes a real API call.

    Returns the list of kwargs each create() call was made with.
    """
    calls: list[dict[str, object]] = []

    async def fake_create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr("caredesk.generation.generator.AsyncOpenAI", lambda **kwargs: fake_client)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
#
# Hallucinated citations first: this is the test that protects the whole
# system, and the easiest one to accidentally break in a later refactor.


async def test_hallucinated_citation_converts_to_refusal_and_logs_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings()
    retrieval = _retrieval([_result("doc1::0000", score=0.9)])
    _patch_openai(
        monkeypatch,
        "You can do X [doc1::0000] and Y [doc_nonexistent::0001].",
    )

    with caplog.at_level(logging.ERROR, logger="caredesk.generation.generator"):
        result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is True
    assert result.refusal_reason == "hallucinated_citation"
    assert result.answer_text is None
    assert result.citations == []

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert getattr(error_records[0], "fabricated_chunk_ids", None) == ["doc_nonexistent::0001"]


async def test_partial_hallucination_still_fully_refuses_not_partial_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    retrieval = _retrieval(
        [_result("doc1::0000", score=0.9), _result("doc2::0000", doc_id="doc2", score=0.8, rank=2)]
    )
    # One real citation, one fabricated -- must refuse entirely, not
    # return the real one and drop the fake.
    _patch_openai(monkeypatch, "Real [doc1::0000] and fake [doc_fake::0000].")

    result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is True
    assert result.refusal_reason == "hallucinated_citation"
    assert result.citations == []


async def test_empty_retrieval_refuses_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    retrieval = _retrieval([])
    calls = _patch_openai(monkeypatch, "should never be seen")

    result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is True
    assert result.refusal_reason == "no_results"
    assert result.answer_text is None
    assert len(calls) == 0
    assert result.input_tokens == 0
    assert result.output_tokens == 0


async def test_low_relevance_refuses_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(min_relevance_score=0.5)
    retrieval = _retrieval([_result("doc1::0000", score=0.2)])
    calls = _patch_openai(monkeypatch, "should never be seen")

    result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is True
    assert result.refusal_reason == "low_relevance"
    assert len(calls) == 0


async def test_model_sentinel_produces_model_insufficient_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    retrieval = _retrieval([_result("doc1::0000", score=0.9)])
    _patch_openai(monkeypatch, REFUSAL_SENTINEL)

    result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is True
    assert result.refusal_reason == "model_insufficient"
    assert result.answer_text is None
    assert result.citations == []


async def test_well_cited_answer_parses_citations_and_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    retrieval = _retrieval(
        [
            _result("doc1::0000", doc_id="doc1", title="Doc 1", filename="doc1.txt", score=0.9),
            _result(
                "doc2::0000", doc_id="doc2", title="Doc 2", filename="doc2.txt", score=0.8, rank=2
            ),
        ]
    )
    _patch_openai(monkeypatch, "You can do X [doc1::0000] and also Y [doc2::0000].")

    result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is False
    assert result.refusal_reason is None
    assert result.answer_text == "You can do X [doc1::0000] and also Y [doc2::0000]."
    assert [c.chunk_id for c in result.citations] == ["doc1::0000", "doc2::0000"]


async def test_no_citations_and_no_sentinel_converts_to_refusal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings()
    retrieval = _retrieval([_result("doc1::0000", score=0.9)])
    _patch_openai(monkeypatch, "This is an answer with no citation markers at all.")

    with caplog.at_level(logging.WARNING, logger="caredesk.generation.generator"):
        result = await generate_answer("query", "patient", retrieval, settings)

    assert result.refused is True
    assert result.refusal_reason == "no_citations"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_citations_are_structured_objects_matching_context_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    chunk = _result("doc1::0000", doc_id="doc1", title="Doc Title", filename="doc1.txt", score=0.9)
    retrieval = _retrieval([chunk])
    _patch_openai(monkeypatch, "Answer [doc1::0000].")

    result = await generate_answer("query", "patient", retrieval, settings)

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.chunk_id == chunk.chunk_id
    assert citation.doc_id == chunk.doc_id
    assert citation.title == chunk.title
    assert citation.filename == chunk.filename


async def test_prompt_version_populated_on_every_result(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()

    refusal = await generate_answer("q", "patient", _retrieval([]), settings)
    assert refusal.prompt_version == PROMPT_VERSION

    retrieval = _retrieval([_result("doc1::0000", score=0.9)])
    _patch_openai(monkeypatch, "Answer [doc1::0000].")
    answer = await generate_answer("q", "patient", retrieval, settings)
    assert answer.prompt_version == PROMPT_VERSION


async def test_token_counts_and_cost_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    retrieval = _retrieval([_result("doc1::0000", score=0.9)])
    _patch_openai(monkeypatch, "Answer [doc1::0000].", prompt_tokens=150, completion_tokens=30)

    result = await generate_answer("query", "patient", retrieval, settings)

    assert result.input_tokens == 150
    assert result.output_tokens == 30
    assert result.cost_usd > 0
