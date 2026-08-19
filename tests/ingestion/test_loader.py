"""Tests for caredesk.ingestion.loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from caredesk.ingestion.loader import (
    DocumentLoadError,
    ManifestError,
    SourceType,
    load_corpus,
    load_document,
    load_manifest,
)

FIXTURE_CORPUS = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def _entry(**overrides: object) -> dict[str, object]:
    """A valid manifest entry dict, with fields overridden per test."""
    base: dict[str, object] = {
        "doc_id": "test_doc",
        "filename": "doc.txt",
        "source_type": "faq_markdown",
        "persona_visibility": "patient",
        "title": "Test Doc",
        "provenance": "Synthetic — authored for tests.",
        "added_date": "2026-08-18",
        "notes": "",
    }
    base.update(overrides)
    return base


def _build_corpus(tmp_path: Path, entries: list[dict[str, object]], files: dict[str, str]) -> Path:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    for rel_path, content in files.items():
        file_path = corpus_root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content.encode("utf-8"))
    (corpus_root / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return corpus_root


def test_load_corpus_full_matches_manifest_length() -> None:
    entries = load_manifest(FIXTURE_CORPUS)
    docs = list(load_corpus(FIXTURE_CORPUS))
    assert len(docs) == len(entries) == 5


def test_source_types_filter_returns_only_requested_types() -> None:
    docs = list(load_corpus(FIXTURE_CORPUS, source_types=[SourceType.FAQ_MARKDOWN]))
    assert len(docs) == 1
    assert docs[0].doc_id == "faq_sample"
    assert docs[0].source_type == "faq_markdown"


def test_missing_file_raises_with_doc_id_named(tmp_path: Path) -> None:
    entries = [_entry(doc_id="ghost_doc", filename="does_not_exist.txt")]
    corpus_root = _build_corpus(tmp_path, entries, files={})

    with pytest.raises(ManifestError, match="ghost_doc"):
        load_manifest(corpus_root)


def test_duplicate_doc_id_raises(tmp_path: Path) -> None:
    entries = [
        _entry(doc_id="dupe", filename="a.txt"),
        _entry(doc_id="dupe", filename="b.txt"),
    ]
    corpus_root = _build_corpus(
        tmp_path, entries, files={"a.txt": "Content A.", "b.txt": "Content B."}
    )

    with pytest.raises(ManifestError, match="dupe"):
        load_manifest(corpus_root)


def test_unknown_source_type_raises(tmp_path: Path) -> None:
    entries = [_entry(doc_id="bad_type", filename="a.txt", source_type="not_a_real_type")]
    corpus_root = _build_corpus(tmp_path, entries, files={"a.txt": "Content."})

    with pytest.raises(ManifestError, match="bad_type"):
        load_manifest(corpus_root)


def test_missing_persona_visibility_raises(tmp_path: Path) -> None:
    entry = _entry(doc_id="no_visibility", filename="a.txt")
    del entry["persona_visibility"]
    corpus_root = _build_corpus(tmp_path, [entry], files={"a.txt": "Content."})

    with pytest.raises(ManifestError, match="no_visibility"):
        load_manifest(corpus_root)


def test_unmanifested_file_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="caredesk.ingestion.loader"):
        load_manifest(FIXTURE_CORPUS)

    assert any("faq_unlisted_extra.txt" in record.message for record in caplog.records)


def test_persona_visibility_preserved_exactly() -> None:
    docs = {doc.doc_id: doc for doc in load_corpus(FIXTURE_CORPUS)}
    expected = {
        "faq_sample": "patient",
        "policy_sample": "both",
        "leaflet_sample": "both",
        "ticket_sample": "staff",
        "runbook_sample": "staff",
    }
    for doc_id, persona_visibility in expected.items():
        assert docs[doc_id].persona_visibility == persona_visibility


def test_normalization_crlf_and_blank_line_collapse(tmp_path: Path) -> None:
    raw = "First line.  \r\nSecond line.\r\n\r\n\r\n\r\nThird line after four blank lines.\r\n"
    entries = [_entry(doc_id="crlf_doc", filename="crlf.txt")]
    corpus_root = _build_corpus(tmp_path, entries, files={"crlf.txt": raw})

    (doc,) = list(load_corpus(corpus_root))

    assert "\r" not in doc.text
    assert "\n\n\n\n" not in doc.text
    assert doc.text == ("First line.\nSecond line.\n\n\nThird line after four blank lines.")


def test_document_normalizing_to_empty_raises(tmp_path: Path) -> None:
    entries = [_entry(doc_id="blank_doc", filename="blank.txt")]
    corpus_root = _build_corpus(tmp_path, entries, files={"blank.txt": "   \n\n   \n\t\n"})

    with pytest.raises(DocumentLoadError, match="blank_doc"):
        (entry,) = load_manifest(corpus_root)
        load_document(entry, corpus_root)
