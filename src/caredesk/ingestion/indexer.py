"""Indexing orchestration: loader -> chunker -> embedder -> Postgres.

Owns idempotency (skip unchanged documents, re-embed only what changed),
the cost guard in front of the embedding API, and per-document
transactional writes. Contains no embedding-API code itself (see
`caredesk.ingestion.embedder`) and no chunking logic (see
`caredesk.ingestion.chunker`).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, text

from caredesk.config import Settings
from caredesk.ingestion import embedder
from caredesk.ingestion.chunker import Chunk, chunk_document, current_strategy
from caredesk.ingestion.embedder import EmbeddedChunk, embed_chunks, estimate_cost
from caredesk.ingestion.loader import LoadedDocument, SourceType, load_corpus, load_manifest
from caredesk.storage.models import ChunkRecord, DocumentRecord
from caredesk.storage.session import get_engine, session_scope

logger = logging.getLogger(__name__)


class IndexerError(ValueError):
    """Raised for indexing configuration problems or a refused/cancelled run."""


class IndexReport(BaseModel):
    """Summary of one `index_corpus` run."""

    documents_inserted: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    documents_pruned: int = 0
    orphaned_doc_ids: list[str] = Field(default_factory=list)
    chunks_written: int = 0
    tokens_embedded: int = 0
    api_calls: int = 0
    estimated_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class _DocPlan:
    doc: LoadedDocument
    action: Literal["insert", "update"]
    content_hash: str


def compute_content_hash(text_content: str) -> str:
    """SHA-256 of normalised document text, used to detect unchanged content."""
    return hashlib.sha256(text_content.encode("utf-8")).hexdigest()


def ensure_hnsw_index(settings: Settings) -> None:
    """Create the HNSW index on `chunks.embedding` if it doesn't exist yet.

    Deliberately not part of the Alembic migration (see
    migrations/versions/0001_create_documents_and_chunks.py): building an
    HNSW index against an empty table and then bulk-inserting is slower
    than bulk-loading first and building the index once. Called after the
    indexing loop below; `IF NOT EXISTS` makes repeat calls a cheap no-op.
    """
    m = int(settings.hnsw_m)
    ef_construction = int(settings.hnsw_ef_construction)
    engine = get_engine(settings.database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks "
                "USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {m}, ef_construction = {ef_construction})"
            )
        )


async def index_corpus(
    settings: Settings,
    *,
    prune: bool = False,
    confirm: bool = False,
    dry_run: bool = False,
    source_types: Sequence[SourceType] | None = None,
    doc_ids: Sequence[str] | None = None,
) -> IndexReport:
    """Index the corpus into Postgres, embedding only new or changed documents.

    A document is re-embedded when its normalised text hash changes, or
    when `current_strategy(settings)` differs from what's stored on its
    existing chunks (e.g. after a chunk_size/chunk_overlap change) —
    otherwise it's skipped with zero embedding calls. Documents present in
    the database but absent from the manifest are always reported via
    `IndexReport.orphaned_doc_ids`, and deleted only when `prune=True`.

    `source_types`/`doc_ids` narrow which documents THIS RUN considers for
    insert/update/skip — they do not narrow orphan detection, which always
    compares the full manifest against the full `documents` table, so a
    filtered run can never mistake an out-of-scope document for orphaned.
    """
    embedder.validate_embedding_dimension(settings)
    start = time.monotonic()

    docs = list(load_corpus(settings.corpus_root, source_types=source_types))
    if doc_ids is not None:
        wanted_ids = set(doc_ids)
        docs = [doc for doc in docs if doc.doc_id in wanted_ids]

    full_manifest_doc_ids = {entry.doc_id for entry in load_manifest(settings.corpus_root)}
    expected_strategy = current_strategy(settings)

    plans: list[_DocPlan] = []
    skipped = 0
    with session_scope(settings) as session:
        for doc in docs:
            content_hash = compute_content_hash(doc.text)
            existing = session.get(DocumentRecord, doc.doc_id)

            if existing is None:
                plans.append(_DocPlan(doc=doc, action="insert", content_hash=content_hash))
                continue

            existing_strategies = set(
                session.execute(
                    select(ChunkRecord.strategy).where(ChunkRecord.doc_id == doc.doc_id).distinct()
                ).scalars()
            )
            content_changed = existing.content_hash != content_hash
            strategy_changed = existing_strategies != {expected_strategy}

            if content_changed or strategy_changed:
                plans.append(_DocPlan(doc=doc, action="update", content_hash=content_hash))
            else:
                skipped += 1

        all_db_doc_ids = set(session.execute(select(DocumentRecord.doc_id)).scalars())

    orphaned_doc_ids = sorted(all_db_doc_ids - full_manifest_doc_ids)

    doc_chunks: dict[str, list[Chunk]] = {}
    all_chunks: list[Chunk] = []
    for plan in plans:
        chunks = chunk_document(plan.doc, settings)
        doc_chunks[plan.doc.doc_id] = chunks
        all_chunks.extend(chunks)

    tokens_to_embed = sum(chunk.token_count for chunk in all_chunks)
    estimated_cost = estimate_cost(tokens_to_embed, settings)

    report = IndexReport(
        documents_inserted=sum(1 for p in plans if p.action == "insert"),
        documents_updated=sum(1 for p in plans if p.action == "update"),
        documents_skipped=skipped,
        documents_pruned=len(orphaned_doc_ids) if prune else 0,
        orphaned_doc_ids=orphaned_doc_ids,
        estimated_cost_usd=estimated_cost,
    )

    logger.info(
        "Plan: %d insert, %d update, %d skip (unchanged); %d chunks / %d tokens to embed "
        "(~$%.4f); %d orphaned doc(s) in DB not in manifest",
        report.documents_inserted,
        report.documents_updated,
        report.documents_skipped,
        len(all_chunks),
        tokens_to_embed,
        estimated_cost,
        len(orphaned_doc_ids),
    )

    if dry_run:
        report.elapsed_seconds = time.monotonic() - start
        return report

    if estimated_cost > settings.embedding_cost_ceiling_usd:
        raise IndexerError(
            f"Estimated cost ${estimated_cost:.4f} exceeds "
            f"Settings.embedding_cost_ceiling_usd (${settings.embedding_cost_ceiling_usd:.4f}); "
            "refusing to run. Raise the ceiling if this is intentional."
        )

    if all_chunks and not confirm:
        answer = input(
            f"About to embed {len(all_chunks)} chunk(s) / {tokens_to_embed} tokens "
            f"(~${estimated_cost:.4f}). Proceed? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            raise IndexerError("Indexing cancelled: not confirmed.")

    embedded_by_doc: dict[str, list[EmbeddedChunk]] = defaultdict(list)
    if all_chunks:
        embedded = await embed_chunks(all_chunks, settings)
        for embedded_chunk in embedded:
            embedded_by_doc[embedded_chunk.chunk.doc_id].append(embedded_chunk)
        report.tokens_embedded = tokens_to_embed
        report.api_calls = math.ceil(len(all_chunks) / settings.embedding_batch_size)

    now = datetime.now(UTC)
    for plan in plans:
        doc_embeddings = embedded_by_doc.get(plan.doc.doc_id, [])
        with session_scope(settings) as session:
            if plan.action == "update":
                existing = session.get(DocumentRecord, plan.doc.doc_id)
                if existing is not None:
                    session.delete(existing)
                    session.flush()

            session.add(
                DocumentRecord(
                    doc_id=plan.doc.doc_id,
                    filename=plan.doc.filename,
                    source_type=plan.doc.source_type,
                    persona_visibility=plan.doc.persona_visibility,
                    title=plan.doc.title,
                    provenance=plan.doc.provenance,
                    notes=plan.doc.notes,
                    char_count=plan.doc.char_count,
                    content_hash=plan.content_hash,
                    indexed_at=now,
                )
            )
            for embedded_chunk in doc_embeddings:
                chunk = embedded_chunk.chunk
                session.add(
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        source_type=chunk.source_type,
                        persona_visibility=chunk.persona_visibility,
                        strategy=chunk.strategy,
                        embedding=embedded_chunk.embedding,
                        embedding_model=embedded_chunk.embedding_model,
                        indexed_at=now,
                    )
                )

        report.chunks_written += len(doc_embeddings)

    if prune and orphaned_doc_ids:
        with session_scope(settings) as session:
            for doc_id in orphaned_doc_ids:
                existing = session.get(DocumentRecord, doc_id)
                if existing is not None:
                    session.delete(existing)

    if report.chunks_written > 0:
        ensure_hnsw_index(settings)

    report.elapsed_seconds = time.monotonic() - start
    return report
