"""Tests for caredesk.ingestion.chunker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tiktoken

from caredesk.config import Settings
from caredesk.ingestion.chunker import Chunk, ChunkerError, chunk_document
from caredesk.ingestion.loader import LoadedDocument

EMBEDDING_MODEL = "text-embedding-3-small"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_public_key": "test-pub",
        "langfuse_secret_key": "test-secret",
        "embedding_model": EMBEDDING_MODEL,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _doc(text: str, **overrides: object) -> LoadedDocument:
    base: dict[str, object] = {
        "doc_id": "doc_one",
        "filename": "doc_one.txt",
        "source_type": "faq_markdown",
        "persona_visibility": "patient",
        "title": "Doc One",
        "text": text,
        "char_count": len(text),
        "provenance": "Synthetic — authored for tests.",
        "notes": "",
    }
    base.update(overrides)
    return LoadedDocument(**base)  # type: ignore[arg-type]


def _text_with_token_count(encoding: tiktoken.Encoding, n: int) -> str:
    """Build text that encodes to exactly `n` tokens under `encoding`."""
    filler = "The quick brown fox jumps over the lazy dog near the river bank. " * 50
    tokens = encoding.encode(filler)
    assert len(tokens) >= n, "filler text too short for requested token count"
    return encoding.decode(tokens[:n])


def test_short_document_produces_exactly_one_chunk() -> None:
    settings = _settings()
    doc = _doc("This is a short document, well under the default chunk size.")

    chunks = chunk_document(doc, settings)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_overlap_is_correct() -> None:
    settings = _settings(chunk_size=30, chunk_overlap=8, chunk_min_tokens=5)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    doc = _doc(_text_with_token_count(encoding, 70))

    chunks = chunk_document(doc, settings)

    assert len(chunks) >= 2
    for prev_chunk, next_chunk in zip(chunks, chunks[1:], strict=False):
        tail = encoding.encode(prev_chunk.text)[-settings.chunk_overlap :]
        head = encoding.encode(next_chunk.text)[: settings.chunk_overlap]
        assert tail == head


def test_chunk_ids_unique_and_stable_across_runs() -> None:
    settings = _settings(chunk_size=30, chunk_overlap=8, chunk_min_tokens=5)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    doc = _doc(_text_with_token_count(encoding, 90))

    first_run = chunk_document(doc, settings)
    second_run = chunk_document(doc, settings)

    first_ids = [c.chunk_id for c in first_run]
    second_ids = [c.chunk_id for c in second_run]

    assert len(set(first_ids)) == len(first_ids)
    assert first_ids == second_ids


def test_char_offsets_round_trip() -> None:
    settings = _settings(chunk_size=30, chunk_overlap=8, chunk_min_tokens=5)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    doc = _doc(_text_with_token_count(encoding, 90))

    chunks = chunk_document(doc, settings)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


def test_persona_visibility_preserved_on_every_chunk() -> None:
    settings = _settings(chunk_size=30, chunk_overlap=8, chunk_min_tokens=5)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    doc = _doc(_text_with_token_count(encoding, 90), persona_visibility="staff")

    chunks = chunk_document(doc, settings)

    assert len(chunks) >= 2
    assert all(chunk.persona_visibility == "staff" for chunk in chunks)


def test_missing_persona_visibility_raises() -> None:
    settings = _settings()
    doc_without_visibility = SimpleNamespace(
        doc_id="doc_no_visibility",
        filename="doc.txt",
        source_type="faq_markdown",
        title="No Visibility",
        text="Some document text.",
        char_count=20,
        provenance="Synthetic — authored for tests.",
        notes="",
    )

    with pytest.raises(ChunkerError, match="doc_no_visibility"):
        chunk_document(doc_without_visibility, settings)  # type: ignore[arg-type]


def test_trailing_fragment_under_min_tokens_is_dropped() -> None:
    settings = _settings(chunk_size=20, chunk_overlap=2, chunk_min_tokens=5)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    # step = 18; chunks at token starts 0 and 18 are full (20 tokens each,
    # covering [0,20) and [18,38)); the next start (36) would leave only
    # 3 tokens (39 - 36), under chunk_min_tokens, and must be dropped.
    doc = _doc(_text_with_token_count(encoding, 39))

    chunks = chunk_document(doc, settings)

    assert len(chunks) == 2
    assert all(chunk.token_count >= settings.chunk_min_tokens for chunk in chunks)


def test_chunk_overlap_greater_than_or_equal_to_chunk_size_raises() -> None:
    settings = _settings(chunk_size=20, chunk_overlap=20)
    doc = _doc("Some text.")

    with pytest.raises(ChunkerError):
        chunk_document(doc, settings)


def test_strategy_reflects_configured_chunk_size_and_overlap() -> None:
    settings = _settings(chunk_size=100, chunk_overlap=10)
    doc = _doc("Some short text for a strategy-label check.")

    chunks = chunk_document(doc, settings)

    assert chunks[0].strategy == "fixed_100_10"


def test_default_settings_strategy_label() -> None:
    assert Settings.model_fields["chunk_size"].default == 512
    assert Settings.model_fields["chunk_overlap"].default == 50
    settings = _settings()
    doc = _doc("Some short text for the default strategy label.")

    chunks = chunk_document(doc, settings)

    assert chunks[0].strategy == "fixed_512_50"


def test_chunk_model_has_no_none_fields() -> None:
    settings = _settings()
    doc = _doc("Some text used to check that every Chunk field is populated.")

    chunks = chunk_document(doc, settings)

    chunk: Chunk = chunks[0]
    for field_name in Chunk.model_fields:
        value = getattr(chunk, field_name)
        assert value is not None
        assert value != ""
