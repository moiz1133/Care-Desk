"""Document loader.

Reads the corpus manifest at `<corpus_root>/manifest.json`, validates it,
and loads each referenced document into a `LoadedDocument`. Every document
in this corpus is a plain `.txt` file — there is no PDF or markdown parsing
here. Source-type dispatch exists as a real registry (`LOADERS`) so that a
later commit can swap in per-type handling without touching this module's
public interface.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

PersonaVisibility = Literal["patient", "staff", "both"]

MANIFEST_FILENAME = "manifest.json"


class SourceType(StrEnum):
    """The five document categories tracked in the corpus manifest.

    POLICY_PDF keeps its name even though the underlying files are `.txt`
    at this commit: it is the manifest's canonical `source_type` value, and
    renaming it here would desync the enum from `manifest.json`.
    """

    POLICY_PDF = "policy_pdf"
    FAQ_MARKDOWN = "faq_markdown"
    MEDICATION_LEAFLET = "medication_leaflet"
    RESOLVED_TICKET = "resolved_ticket"
    STAFF_RUNBOOK = "staff_runbook"


class ManifestError(ValueError):
    """Raised when the corpus manifest is missing, malformed, or inconsistent."""


class DocumentLoadError(ValueError):
    """Raised when a manifested document fails to load or normalize."""


class ManifestEntry(BaseModel):
    """One validated row of `manifest.json`."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    source_type: SourceType
    persona_visibility: PersonaVisibility
    title: str
    provenance: str
    added_date: str
    notes: str


class LoadedDocument(BaseModel):
    """A fully loaded, normalized document ready for chunking."""

    doc_id: str
    filename: str
    source_type: str
    persona_visibility: PersonaVisibility
    title: str
    text: str
    char_count: int
    provenance: str
    notes: str


def _normalize_text(text: str) -> str:
    """Apply the shared normalisation rules used by every loader route.

    Normalises line endings to `\\n`, strips trailing whitespace per line,
    collapses runs of three or more blank lines to two, and strips leading
    and trailing whitespace from the whole document. Does not touch heading
    markup, casing, or internal structure otherwise.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)

    return "\n".join(collapsed).strip()


def _read_and_normalize(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"File {path} is not valid UTF-8: {exc}") from exc
    return _normalize_text(raw)


def _load_policy_pdf_text(path: Path) -> str:
    return _read_and_normalize(path)


def _load_faq_markdown_text(path: Path) -> str:
    return _read_and_normalize(path)


def _load_medication_leaflet_text(path: Path) -> str:
    return _read_and_normalize(path)


def _load_resolved_ticket_text(path: Path) -> str:
    return _read_and_normalize(path)


def _load_staff_runbook_text(path: Path) -> str:
    return _read_and_normalize(path)


LOADERS: dict[SourceType, Callable[[Path], str]] = {
    SourceType.POLICY_PDF: _load_policy_pdf_text,
    SourceType.FAQ_MARKDOWN: _load_faq_markdown_text,
    SourceType.MEDICATION_LEAFLET: _load_medication_leaflet_text,
    SourceType.RESOLVED_TICKET: _load_resolved_ticket_text,
    SourceType.STAFF_RUNBOOK: _load_staff_runbook_text,
}


def find_unmanifested_files(corpus_root: Path, entries: Iterable[ManifestEntry]) -> list[str]:
    """Return `.txt` files under `corpus_root` that no manifest entry references."""
    manifest_files = {entry.filename for entry in entries}
    on_disk = {path.relative_to(corpus_root).as_posix() for path in corpus_root.rglob("*.txt")}
    return sorted(on_disk - manifest_files)


def load_manifest(corpus_root: Path) -> list[ManifestEntry]:
    """Load, schema-validate, and cross-check every entry in the corpus manifest.

    Raises `ManifestError` for a malformed entry, a duplicate doc_id, or an
    entry whose file is missing on disk. Logs a warning (does not raise) for
    files on disk that aren't referenced by any manifest entry.
    """
    manifest_path = corpus_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest not found at {manifest_path}")

    raw_text = manifest_path.read_text(encoding="utf-8")
    try:
        raw_entries = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest at {manifest_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw_entries, list):
        raise ManifestError(f"Manifest at {manifest_path} must be a JSON array")

    entries: list[ManifestEntry] = []
    seen_doc_ids: set[str] = set()

    for index, raw_entry in enumerate(raw_entries):
        doc_id_hint = raw_entry.get("doc_id") if isinstance(raw_entry, dict) else None
        label = doc_id_hint or f"<entry at index {index}>"

        try:
            entry = ManifestEntry.model_validate(raw_entry)
        except ValidationError as exc:
            fields = ", ".join(".".join(str(part) for part in err["loc"]) for err in exc.errors())
            raise ManifestError(
                f"Manifest entry {label!r} failed schema validation (field(s): {fields}): {exc}"
            ) from exc

        if entry.doc_id in seen_doc_ids:
            raise ManifestError(f"Duplicate doc_id in manifest: {entry.doc_id!r}")
        seen_doc_ids.add(entry.doc_id)

        doc_path = corpus_root / entry.filename
        if not doc_path.is_file():
            raise ManifestError(
                f"Manifest entry {entry.doc_id!r} references a missing file: {entry.filename!r}"
            )

        entries.append(entry)

    unmanifested = find_unmanifested_files(corpus_root, entries)
    if unmanifested:
        logger.warning(
            "%d file(s) under %s are not referenced by manifest.json: %s",
            len(unmanifested),
            corpus_root,
            ", ".join(unmanifested),
        )

    return entries


def load_document(entry: ManifestEntry, corpus_root: Path) -> LoadedDocument:
    """Load and normalize the single document described by `entry`."""
    doc_path = corpus_root / entry.filename
    if not doc_path.is_file():
        raise DocumentLoadError(f"File for doc_id {entry.doc_id!r} not found: {doc_path}")

    loader_fn = LOADERS.get(entry.source_type)
    if loader_fn is None:
        raise DocumentLoadError(
            f"No loader registered for source_type={entry.source_type!r} (doc_id={entry.doc_id!r})"
        )

    text = loader_fn(doc_path)
    if not text:
        raise DocumentLoadError(
            f"Document {entry.doc_id!r} is empty after normalisation ({doc_path})"
        )

    return LoadedDocument(
        doc_id=entry.doc_id,
        filename=entry.filename,
        source_type=entry.source_type.value,
        persona_visibility=entry.persona_visibility,
        title=entry.title,
        text=text,
        char_count=len(text),
        provenance=entry.provenance,
        notes=entry.notes,
    )


def load_corpus(
    corpus_root: Path,
    source_types: Sequence[SourceType] | None = None,
) -> Iterator[LoadedDocument]:
    """Stream every document in the corpus, optionally filtered by source type."""
    entries = load_manifest(corpus_root)
    allowed = set(source_types) if source_types is not None else None
    for entry in entries:
        if allowed is not None and entry.source_type not in allowed:
            continue
        yield load_document(entry, corpus_root)
