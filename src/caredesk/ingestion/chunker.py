"""Fixed-size token chunker.

Deliberately naive: splits document text into fixed-size, overlapping
token windows with no regard for sentence, paragraph, or heading
structure. This is the Week 1 baseline used to measure the damage naive
chunking does before it's replaced by a section-aware strategy in Week 2.
Do not add sentence/heading awareness or per-source-type logic here — that
would defeat the comparison this module exists to enable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from functools import lru_cache

import tiktoken
from pydantic import BaseModel

from caredesk.config import Settings
from caredesk.ingestion.loader import LoadedDocument, PersonaVisibility

logger = logging.getLogger(__name__)


class ChunkerError(ValueError):
    """Raised when a document can't be chunked due to invalid input or configuration."""


class Chunk(BaseModel):
    """One fixed-size, token-bounded slice of a source document."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    source_type: str
    persona_visibility: PersonaVisibility
    title: str
    filename: str
    strategy: str


@lru_cache(maxsize=8)
def _get_encoding(embedding_model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(embedding_model)


def chunk_document(doc: LoadedDocument, settings: Settings) -> list[Chunk]:
    """Split one document into fixed-size, overlapping token windows.

    Each chunk starts `chunk_size - chunk_overlap` tokens after the
    previous chunk's start. A trailing chunk with fewer than
    `chunk_min_tokens` tokens is dropped, unless it is the document's only
    chunk (a document shorter than `chunk_size` still produces one chunk).
    """
    if settings.chunk_overlap >= settings.chunk_size:
        raise ChunkerError(
            f"chunk_overlap ({settings.chunk_overlap}) must be smaller than "
            f"chunk_size ({settings.chunk_size})"
        )

    doc_id = getattr(doc, "doc_id", None)
    persona_visibility = getattr(doc, "persona_visibility", None)
    if not persona_visibility:
        raise ChunkerError(
            f"Document {doc_id!r} has no persona_visibility; refusing to chunk it, "
            "since persona_visibility is a retrieval-time security control."
        )

    encoding = _get_encoding(settings.embedding_model)
    tokens = encoding.encode(doc.text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        logger.warning("Document %s produced zero tokens; zero chunks.", doc_id)
        return []

    strategy = f"fixed_{settings.chunk_size}_{settings.chunk_overlap}"
    step = settings.chunk_size - settings.chunk_overlap

    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0
    while start < total_tokens:
        end = min(start + settings.chunk_size, total_tokens)
        token_slice = tokens[start:end]

        if len(token_slice) < settings.chunk_min_tokens and chunk_index > 0:
            break

        chunk_text = encoding.decode(token_slice)
        char_start = len(encoding.decode(tokens[:start]))
        char_end = char_start + len(chunk_text)

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{chunk_index:04d}",
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(token_slice),
                char_start=char_start,
                char_end=char_end,
                source_type=doc.source_type,
                persona_visibility=persona_visibility,
                title=doc.title,
                filename=doc.filename,
                strategy=strategy,
            )
        )

        chunk_index += 1
        if end >= total_tokens:
            break
        start += step

    if not chunks:
        logger.warning("Document %s produced zero chunks.", doc_id)

    return chunks


def chunk_corpus(docs: Iterable[LoadedDocument], settings: Settings) -> Iterator[Chunk]:
    """Stream chunks for every document in `docs`."""
    for doc in docs:
        yield from chunk_document(doc, settings)
