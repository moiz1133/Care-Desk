"""Tests for caredesk.ingestion.indexer.

Requires a real, reachable Postgres+pgvector database — NOT the dev
database — configured via Settings.test_database_url (env TEST_DATABASE_URL
/ .env), e.g.:

    TEST_DATABASE_URL=postgresql+psycopg://caredesk:caredesk@localhost:5432/caredesk_test

Create it once against the same Postgres server docker-compose already
runs (pgvector is enabled per-database, so this needs its own
`CREATE EXTENSION vector;`):

    CREATE DATABASE caredesk_test;
    \\c caredesk_test
    CREATE EXTENSION vector;

If TEST_DATABASE_URL is unset or the database is unreachable, this whole
module is skipped rather than failing the suite (testcontainers would be
the alternative, but it requires a working Docker daemon, which this
project doesn't assume is present everywhere `pytest` runs).

The embedding API is always mocked — no test in this file makes a real
OpenAI call.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from caredesk.config import Settings
from caredesk.ingestion.chunker import Chunk
from caredesk.ingestion.embedder import EmbeddedChunk
from caredesk.ingestion.indexer import index_corpus
from caredesk.ingestion.loader import LoadedDocument, ManifestEntry, SourceType
from caredesk.storage.models import Base, ChunkRecord, DocumentRecord
from caredesk.storage.session import get_engine, session_scope


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_enabled": False,
        "embedding_cost_ceiling_usd": 1000.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


_bootstrap_settings = _make_settings()

if not _bootstrap_settings.test_database_url:
    pytest.skip(
        "Settings.test_database_url (env TEST_DATABASE_URL) is not set; skipping "
        "the DB-backed indexer suite. Point it at a real, non-dev Postgres+pgvector "
        "database to run it.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings() -> Iterator[Settings]:
    # NOT overridden: embedding_dimension. caredesk.storage.models reads
    # Settings.embedding_dimension once, at import time, to size the
    # `chunks.embedding` column — that already happened by the time this
    # fixture runs, using whatever env vars were present at collection.
    # Overriding it here would desync fake embedding vectors below from the
    # column's real, already-fixed width.
    settings = _make_settings(database_url=_bootstrap_settings.test_database_url)

    engine = get_engine(settings.database_url)
    try:
        with engine.connect():
            pass
    except OperationalError as exc:
        pytest.skip(f"Settings.test_database_url is unreachable: {exc}")

    Base.metadata.create_all(engine)
    yield settings
    Base.metadata.drop_all(engine)


@pytest.fixture
def clean_db(test_settings: Settings) -> Settings:
    engine = get_engine(test_settings.database_url)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE chunks, documents CASCADE"))
    return test_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loaded_doc(
    doc_id: str,
    doc_text: str,
    *,
    source_type: SourceType = SourceType.FAQ_MARKDOWN,
    persona_visibility: str = "patient",
) -> LoadedDocument:
    return LoadedDocument(
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        source_type=source_type.value,
        persona_visibility=persona_visibility,  # type: ignore[arg-type]
        title=doc_id,
        text=doc_text,
        char_count=len(doc_text),
        provenance="Synthetic — authored for tests.",
        notes="",
    )


def _patch_corpus(monkeypatch: pytest.MonkeyPatch, docs: list[LoadedDocument]) -> None:
    entries = [
        ManifestEntry(
            doc_id=doc.doc_id,
            filename=doc.filename,
            source_type=SourceType(doc.source_type),
            persona_visibility=doc.persona_visibility,
            title=doc.title,
            provenance=doc.provenance,
            added_date="2026-01-01",
            notes=doc.notes,
        )
        for doc in docs
    ]

    def fake_load_corpus(
        corpus_root: object, source_types: list[SourceType] | None = None
    ) -> Iterator[LoadedDocument]:
        allowed = set(source_types) if source_types is not None else None
        for doc in docs:
            if allowed is None or SourceType(doc.source_type) in allowed:
                yield doc

    def fake_load_manifest(corpus_root: object) -> list[ManifestEntry]:
        return entries

    monkeypatch.setattr("caredesk.ingestion.indexer.load_corpus", fake_load_corpus)
    monkeypatch.setattr("caredesk.ingestion.indexer.load_manifest", fake_load_manifest)


def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> list[list[Chunk]]:
    """Replace the real embedding call with a fake one; return the list of
    per-call chunk batches so tests can assert on call count and contents.
    """
    calls: list[list[Chunk]] = []

    async def fake_embed_chunks(chunks: list[Chunk], settings: Settings) -> list[EmbeddedChunk]:
        calls.append(list(chunks))
        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=[0.1] * settings.embedding_dimension,
                embedding_model=settings.embedding_model,
            )
            for chunk in chunks
        ]

    monkeypatch.setattr("caredesk.ingestion.indexer.embed_chunks", fake_embed_chunks)
    return calls


def _count(session: Session, model: type) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar_one()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_fresh_index_inserts_all_documents_and_chunks(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    docs = [_loaded_doc("doc_a", "Some text about A."), _loaded_doc("doc_b", "Some text about B.")]
    _patch_corpus(monkeypatch, docs)
    _patch_embedder(monkeypatch)

    report = await index_corpus(settings, confirm=True)

    assert report.documents_inserted == 2
    assert report.documents_updated == 0
    assert report.documents_skipped == 0
    assert report.chunks_written > 0

    with session_scope(settings) as session:
        assert _count(session, DocumentRecord) == 2
        assert _count(session, ChunkRecord) == report.chunks_written


async def test_rerun_with_unchanged_content_makes_zero_embedding_calls(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    docs = [_loaded_doc("doc_a", "Some text about A.")]
    _patch_corpus(monkeypatch, docs)
    embed_calls = _patch_embedder(monkeypatch)

    await index_corpus(settings, confirm=True)
    assert len(embed_calls) == 1

    report = await index_corpus(settings, confirm=True)

    assert report.documents_skipped == 1
    assert report.documents_inserted == 0
    assert report.documents_updated == 0
    assert report.chunks_written == 0
    assert len(embed_calls) == 1  # no new embedding call on the unchanged rerun


async def test_changing_one_document_reembeds_only_that_document(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    doc_a = _loaded_doc("doc_a", "Original text for A.")
    doc_b = _loaded_doc("doc_b", "Original text for B.")
    _patch_corpus(monkeypatch, [doc_a, doc_b])
    embed_calls = _patch_embedder(monkeypatch)

    await index_corpus(settings, confirm=True)
    assert len(embed_calls) == 1
    assert {c.doc_id for c in embed_calls[0]} == {"doc_a", "doc_b"}

    changed_doc_a = _loaded_doc("doc_a", "Completely different text for A now.")
    _patch_corpus(monkeypatch, [changed_doc_a, doc_b])

    report = await index_corpus(settings, confirm=True)

    assert report.documents_updated == 1
    assert report.documents_skipped == 1
    assert len(embed_calls) == 2
    assert {c.doc_id for c in embed_calls[1]} == {"doc_a"}


async def test_strategy_change_triggers_full_reembed(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    docs = [_loaded_doc("doc_a", "Text A."), _loaded_doc("doc_b", "Text B.")]
    _patch_corpus(monkeypatch, docs)
    embed_calls = _patch_embedder(monkeypatch)

    await index_corpus(settings, confirm=True)
    assert len(embed_calls) == 1

    changed_settings = settings.model_copy(update={"chunk_size": settings.chunk_size + 1})
    report = await index_corpus(changed_settings, confirm=True)

    assert report.documents_updated == 2
    assert report.documents_skipped == 0
    assert len(embed_calls) == 2
    assert {c.doc_id for c in embed_calls[1]} == {"doc_a", "doc_b"}


async def test_orphaned_document_reported_and_pruned_only_with_prune(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    doc_a = _loaded_doc("doc_a", "Text A.")
    doc_b = _loaded_doc("doc_b", "Text B.")
    _patch_corpus(monkeypatch, [doc_a, doc_b])
    _patch_embedder(monkeypatch)
    await index_corpus(settings, confirm=True)

    _patch_corpus(monkeypatch, [doc_a])  # doc_b removed from the manifest

    report = await index_corpus(settings, confirm=True, prune=False)
    assert report.orphaned_doc_ids == ["doc_b"]
    assert report.documents_pruned == 0
    with session_scope(settings) as session:
        assert _count(session, DocumentRecord) == 2

    report_pruned = await index_corpus(settings, confirm=True, prune=True)
    assert report_pruned.orphaned_doc_ids == ["doc_b"]
    assert report_pruned.documents_pruned == 1
    with session_scope(settings) as session:
        remaining = session.execute(select(DocumentRecord.doc_id)).scalars().all()
        assert list(remaining) == ["doc_a"]


async def test_failure_mid_document_leaves_no_partial_chunks(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    doc = _loaded_doc("doc_fail", "Text that will fail to index cleanly.")
    _patch_corpus(monkeypatch, [doc])
    _patch_embedder(monkeypatch)

    def fake_chunk_document(document: LoadedDocument, _settings: Settings) -> list[Chunk]:
        # Two chunks sharing a chunk_id: violates the chunks.chunk_id
        # primary key on the second insert, forcing a failure partway
        # through this document's write transaction.
        first = Chunk(
            chunk_id=f"{document.doc_id}::0000",
            doc_id=document.doc_id,
            chunk_index=0,
            text="first",
            token_count=1,
            char_start=0,
            char_end=5,
            source_type=document.source_type,
            persona_visibility=document.persona_visibility,
            title=document.title,
            filename=document.filename,
            strategy="fixed_1_0",
        )
        duplicate_id = first.model_copy(update={"chunk_index": 1, "text": "second"})
        return [first, duplicate_id]

    monkeypatch.setattr("caredesk.ingestion.indexer.chunk_document", fake_chunk_document)

    with pytest.raises(IntegrityError):
        await index_corpus(settings, confirm=True)

    with session_scope(settings) as session:
        assert _count(session, DocumentRecord) == 0
        assert _count(session, ChunkRecord) == 0


async def test_persona_visibility_and_source_type_written_to_chunks(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    doc = _loaded_doc(
        "doc_staff",
        "Staff-only procedure text.",
        source_type=SourceType.STAFF_RUNBOOK,
        persona_visibility="staff",
    )
    _patch_corpus(monkeypatch, [doc])
    _patch_embedder(monkeypatch)

    await index_corpus(settings, confirm=True)

    with session_scope(settings) as session:
        chunks = (
            session.execute(select(ChunkRecord).where(ChunkRecord.doc_id == "doc_staff"))
            .scalars()
            .all()
        )
        assert len(chunks) > 0
        assert all(chunk.persona_visibility == "staff" for chunk in chunks)
        assert all(chunk.source_type == "staff_runbook" for chunk in chunks)


def test_unique_constraint_on_doc_id_and_chunk_index_holds(clean_db: Settings) -> None:
    settings = clean_db
    now = datetime.now(UTC)

    with session_scope(settings) as session:
        session.add(
            DocumentRecord(
                doc_id="doc_uniq",
                filename="doc_uniq.txt",
                source_type="faq_markdown",
                persona_visibility="patient",
                title="Doc Uniq",
                provenance="",
                notes="",
                char_count=10,
                content_hash="abc123",
                indexed_at=now,
            )
        )

    with pytest.raises(IntegrityError), session_scope(settings) as session:
        session.add(
            ChunkRecord(
                chunk_id="doc_uniq::0000",
                doc_id="doc_uniq",
                chunk_index=0,
                text="a",
                token_count=1,
                char_start=0,
                char_end=1,
                source_type="faq_markdown",
                persona_visibility="patient",
                strategy="fixed_1_0",
                embedding=[0.1] * settings.embedding_dimension,
                embedding_model=settings.embedding_model,
                indexed_at=now,
            )
        )
        session.add(
            ChunkRecord(
                chunk_id="doc_uniq::0001",
                doc_id="doc_uniq",
                chunk_index=0,  # same (doc_id, chunk_index) as above
                text="b",
                token_count=1,
                char_start=0,
                char_end=1,
                source_type="faq_markdown",
                persona_visibility="patient",
                strategy="fixed_1_0",
                embedding=[0.2] * settings.embedding_dimension,
                embedding_model=settings.embedding_model,
                indexed_at=now,
            )
        )
