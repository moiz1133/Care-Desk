"""create documents and chunks

Revision ID: 0001
Revises:
Create Date: 2026-08-27

Creates the `documents` and `chunks` tables. Does NOT create the HNSW
index on `chunks.embedding` — that's built by the indexer after the
initial bulk load (see caredesk.ingestion.indexer.ensure_hnsw_index),
since building an HNSW index against an empty table and then
bulk-inserting is slower than loading first and indexing once.

Requires the `vector` extension to already be enabled (the docker-compose
Postgres init script does this) — this migration asserts it exists rather
than creating it, so a missing extension fails loudly and clearly instead
of surfacing as a cryptic "type vector does not exist" error.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from caredesk.config import get_settings

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    extension_exists = connection.execute(
        sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    if not extension_exists:
        raise RuntimeError(
            "Postgres extension 'vector' is not enabled. It should be created by "
            "docker/postgres-init/001_enable_pgvector.sql on first container boot — "
            "check that init script ran, or enable it manually with "
            "`CREATE EXTENSION vector;` before re-running this migration."
        )

    embedding_dimension = get_settings().embedding_dimension

    op.create_table(
        "documents",
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("persona_visibility", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("provenance", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("indexed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("doc_id"),
    )

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("persona_visibility", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(embedding_dimension), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("indexed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("doc_id", "chunk_index", name="uq_chunks_doc_id_chunk_index"),
    )

    op.create_index("ix_chunks_persona_visibility", "chunks", ["persona_visibility"])
    op.create_index("ix_chunks_source_type", "chunks", ["source_type"])
    op.create_index("ix_chunks_strategy", "chunks", ["strategy"])


def downgrade() -> None:
    op.drop_index("ix_chunks_strategy", table_name="chunks")
    op.drop_index("ix_chunks_source_type", table_name="chunks")
    op.drop_index("ix_chunks_persona_visibility", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
