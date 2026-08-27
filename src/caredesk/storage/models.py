"""ORM models.

SQLAlchemy models for the `documents` and `chunks` tables backing the
pgvector index. Table/index DDL lives in `migrations/` (Alembic), not
here — these are the ORM mappings the indexer (and later, retrieval) use
to read and write rows.

`source_type` and `persona_visibility` are denormalised onto `chunks` as
well as `documents` deliberately: retrieval filters on them per-chunk, and
a join back to `documents` on that hot path is exactly what this avoids.

The HNSW index on `chunks.embedding` is intentionally NOT declared here.
It's created by the indexer after the initial bulk load (see
`caredesk.ingestion.indexer.ensure_hnsw_index`) — building it against an
empty table and then bulk-inserting is slower than loading first and
building the index once.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from caredesk.config import get_settings

# The pgvector column type needs a concrete dimension at class-definition
# time. Read it from Settings (rather than hardcoding 1536) so a changed
# Settings.embedding_dimension is reflected here; caredesk.ingestion.embedder
# separately fails loudly at indexing time if embedding_model and
# embedding_dimension disagree.
_EMBEDDING_DIMENSION = get_settings().embedding_dimension


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """One row per indexed source document."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(Text, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    persona_visibility: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkRecord(Base):
    """One row per chunk, with its embedding vector."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunks_doc_id_chunk_index"),
        Index("ix_chunks_persona_visibility", "persona_visibility"),
        Index("ix_chunks_source_type", "source_type"),
        Index("ix_chunks_strategy", "strategy"),
    )

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        Text, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    persona_visibility: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIMENSION), nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")
