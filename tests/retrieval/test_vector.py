"""Tests for caredesk.retrieval.vector.

Requires a real, reachable Postgres+pgvector database — NOT the dev
database — configured via Settings.test_database_url (env TEST_DATABASE_URL
/ .env). See tests/ingestion/test_indexer.py's module docstring for the
one-time setup (CREATE DATABASE caredesk_test; CREATE EXTENSION vector;).
If unset or unreachable, this whole module is skipped.

Chunk embeddings are hand-built, normalised vectors with known cosine
similarity to the (stubbed) query embedding, so every score in these tests
is an exact, independently-computable value — never a real OpenAI call.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import OperationalError

from caredesk.config import Settings
from caredesk.observability import tracing as tracing_module
from caredesk.observability.vocabulary import KNOWN_METADATA_KEYS
from caredesk.retrieval import vector as vector_module
from caredesk.retrieval.vector import RetrievalError, retrieve
from caredesk.storage.models import Base, ChunkRecord, DocumentRecord
from caredesk.storage.session import get_engine, session_scope

EMBEDDING_DIM = 1536


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "langfuse_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


_bootstrap_settings = _make_settings()

if not _bootstrap_settings.test_database_url:
    pytest.skip(
        "Settings.test_database_url (env TEST_DATABASE_URL) is not set; skipping "
        "the DB-backed vector retrieval suite. Point it at a real, non-dev "
        "Postgres+pgvector database to run it.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings() -> Iterator[Settings]:
    # NOT overridden: embedding_dimension — see test_indexer.py for why
    # (caredesk.storage.models reads it once, at import time).
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
    from sqlalchemy import text

    engine = get_engine(test_settings.database_url)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE chunks, documents CASCADE"))
    return test_settings


@pytest.fixture(autouse=True)
def _clear_query_embedding_cache() -> Iterator[None]:
    # The cache is a process-wide lru_cache singleton keyed only by
    # maxsize, so without this, tests sharing the default cache size would
    # see each other's cached query strings.
    vector_module._get_query_cache.cache_clear()
    yield
    vector_module._get_query_cache.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector(active_dims: list[int], dim: int = EMBEDDING_DIM) -> list[float]:
    """A normalised vector with equal weight on `active_dims`.

    Cosine similarity to a pure-dim0 query vector is exactly
    1/sqrt(len(active_dims)) when 0 is among active_dims, else 0 (fully
    orthogonal) when it isn't — giving deterministic, hand-verifiable
    scores without needing a real embedding call.
    """
    vector = [0.0] * dim
    for d in active_dims:
        vector[d] = 1.0
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def _seed_chunk(
    settings: Settings,
    *,
    doc_id: str,
    persona_visibility: str,
    active_dims: list[int],
    source_type: str = "faq_markdown",
    chunk_index: int = 0,
    text_content: str = "chunk text",
) -> None:
    now = datetime.now(UTC)
    with session_scope(settings) as session:
        if session.get(DocumentRecord, doc_id) is None:
            session.add(
                DocumentRecord(
                    doc_id=doc_id,
                    filename=f"{doc_id}.txt",
                    source_type=source_type,
                    persona_visibility=persona_visibility,
                    title=f"Title for {doc_id}",
                    provenance="",
                    notes="",
                    char_count=len(text_content),
                    content_hash=f"hash-{doc_id}",
                    indexed_at=now,
                )
            )
        session.add(
            ChunkRecord(
                chunk_id=f"{doc_id}::{chunk_index:04d}",
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=text_content,
                token_count=3,
                char_start=0,
                char_end=len(text_content),
                source_type=source_type,
                persona_visibility=persona_visibility,
                strategy="fixed_1_0",
                embedding=_make_vector(active_dims),
                embedding_model="test",
                indexed_at=now,
            )
        )


class _RecordingObservation:
    """Minimal Langfuse span/generation stand-in: records every input and
    metadata dict it's given, separately, across both creation and
    .update() calls. Kept separate because they're validated differently:
    metadata keys must stay within the closed vocabulary, input is
    free-form payload -- but input is exactly where the audited persona
    leak lived (`input={"query": ..., "persona": ..., "k": ...}`), so it
    still needs checking for that one invariant."""

    def __init__(self, name: str, kwargs: dict[str, object]) -> None:
        self.name = name
        self.inputs: list[dict[str, object]] = []
        self.metadata_dicts: list[dict[str, object]] = []
        self._collect(kwargs)

    def _collect(self, kwargs: dict[str, object]) -> None:
        if isinstance(kwargs.get("input"), dict):
            self.inputs.append(kwargs["input"])  # type: ignore[arg-type]
        if isinstance(kwargs.get("metadata"), dict):
            self.metadata_dicts.append(kwargs["metadata"])  # type: ignore[arg-type]

    def update(self, **kwargs: object) -> None:
        self._collect(kwargs)

    def __enter__(self) -> _RecordingObservation:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _RecordingLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[_RecordingObservation] = []

    def start_as_current_observation(
        self, *, name: str, as_type: str = "span", **kwargs: object
    ) -> _RecordingObservation:
        obs = _RecordingObservation(name, kwargs)
        self.observations.append(obs)
        return obs


def _patch_query_embedding(monkeypatch: pytest.MonkeyPatch, active_dims: list[int]) -> list[str]:
    """Stub embed_text to always return a vector pointing at `active_dims`.

    Returns the list of query strings actually embedded, so tests can
    assert on call count (e.g. for the cache test).
    """
    calls: list[str] = []

    async def fake_embed_text(text: str, settings: Settings) -> list[float]:
        calls.append(text)
        return _make_vector(active_dims)

    monkeypatch.setattr("caredesk.retrieval.vector.embed_text", fake_embed_text)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_returns_exactly_k_results_when_enough_chunks_exist(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    for i, dims in enumerate([[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]):
        _seed_chunk(settings, doc_id=f"doc_{i}", persona_visibility="patient", active_dims=dims)
    _patch_query_embedding(monkeypatch, [0])

    response = await retrieve("query", "patient", settings, k=3)

    assert len(response.results) == 3


async def test_results_ordered_by_descending_score(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    for i, dims in enumerate([[0, 1, 2, 3], [0], [0, 1, 2], [0, 1]]):
        _seed_chunk(settings, doc_id=f"doc_{i}", persona_visibility="patient", active_dims=dims)
    _patch_query_embedding(monkeypatch, [0])

    response = await retrieve("query", "patient", settings, k=4)

    scores = [result.score for result in response.results]
    assert scores == sorted(scores, reverse=True)
    assert response.results[0].doc_id == "doc_1"  # active_dims=[0]: identical, score 1.0


async def test_scores_are_similarities_not_distances(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    _seed_chunk(settings, doc_id="doc_identical", persona_visibility="patient", active_dims=[0])
    _seed_chunk(settings, doc_id="doc_orthogonal", persona_visibility="patient", active_dims=[1])
    _patch_query_embedding(monkeypatch, [0])

    response = await retrieve("query", "patient", settings, k=2)

    for result in response.results:
        assert 0.0 <= result.score <= 1.0
    by_doc = {result.doc_id: result.score for result in response.results}
    assert by_doc["doc_identical"] == pytest.approx(1.0)
    assert by_doc["doc_orthogonal"] == pytest.approx(0.0)


async def test_patient_persona_never_returns_staff_only_chunk(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    _seed_chunk(settings, doc_id="doc_patient", persona_visibility="patient", active_dims=[0])
    _seed_chunk(settings, doc_id="doc_staff", persona_visibility="staff", active_dims=[0])
    _seed_chunk(settings, doc_id="doc_both", persona_visibility="both", active_dims=[0])
    _patch_query_embedding(monkeypatch, [0])

    response = await retrieve("query", "patient", settings, k=10)

    assert len(response.results) == 2
    visibilities = {result.persona_visibility for result in response.results}
    assert visibilities <= {"patient", "both"}
    assert "staff" not in visibilities


async def test_staff_persona_returns_staff_and_both_never_patient_only(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    _seed_chunk(settings, doc_id="doc_patient", persona_visibility="patient", active_dims=[0])
    _seed_chunk(settings, doc_id="doc_staff", persona_visibility="staff", active_dims=[0])
    _seed_chunk(settings, doc_id="doc_both", persona_visibility="both", active_dims=[0])
    _patch_query_embedding(monkeypatch, [0])

    response = await retrieve("query", "staff", settings, k=10)

    assert len(response.results) == 2
    visibilities = {result.persona_visibility for result in response.results}
    assert visibilities <= {"staff", "both"}
    assert "patient" not in visibilities


async def test_missing_persona_argument_raises_type_error(clean_db: Settings) -> None:
    settings = clean_db
    with pytest.raises(TypeError):
        await retrieve(query="test", settings=settings)  # type: ignore[call-arg]


async def test_invalid_persona_value_raises(clean_db: Settings) -> None:
    settings = clean_db
    with pytest.raises(RetrievalError):
        await retrieve("test", "nurse", settings)


async def test_source_type_filter_constrains_results(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    _seed_chunk(
        settings,
        doc_id="doc_faq",
        persona_visibility="patient",
        active_dims=[0],
        source_type="faq_markdown",
    )
    _seed_chunk(
        settings,
        doc_id="doc_policy",
        persona_visibility="patient",
        active_dims=[0],
        source_type="policy_pdf",
    )
    _patch_query_embedding(monkeypatch, [0])

    from caredesk.ingestion.loader import SourceType

    response = await retrieve(
        "query", "patient", settings, k=10, source_types=[SourceType.FAQ_MARKDOWN]
    )

    assert len(response.results) == 1
    assert response.results[0].doc_id == "doc_faq"
    assert response.results[0].source_type == "faq_markdown"


async def test_fewer_than_k_matching_chunks_returns_what_exists(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    _seed_chunk(settings, doc_id="doc_only", persona_visibility="patient", active_dims=[0])
    _patch_query_embedding(monkeypatch, [0])

    response = await retrieve("query", "patient", settings, k=10)

    assert len(response.results) == 1


async def test_query_embedding_cache_makes_one_embedding_call_for_repeated_query(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = clean_db
    _seed_chunk(settings, doc_id="doc_cache", persona_visibility="patient", active_dims=[0])
    embed_calls = _patch_query_embedding(monkeypatch, [0])

    await retrieve("what is my copay", "patient", settings)
    await retrieve("what is my copay", "patient", settings)

    assert len(embed_calls) == 1


async def test_spans_only_use_known_metadata_keys_and_never_repeat_persona(
    clean_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the real retrieve() -- not a mock of it -- so this catches
    an actual ad-hoc metadata key or persona leak in vector.py, not just a
    typo in a test double. Persona is a request-level invariant (a trace
    tag + trace metadata, set once in start_query_trace); it must never
    reappear in a span's own input/metadata, which is exactly the bug
    commit 9 found and fixed on the `retrieve` span."""
    settings = clean_db
    _seed_chunk(settings, doc_id="doc_a", persona_visibility="patient", active_dims=[0])
    _patch_query_embedding(monkeypatch, [0])

    client = _RecordingLangfuseClient()
    monkeypatch.setattr(tracing_module, "get_langfuse_client", lambda settings: client)

    await retrieve("what is my copay", "patient", settings)

    assert client.observations, "expected retrieve() to emit spans"
    for obs in client.observations:
        for metadata in obs.metadata_dicts:
            unknown = set(metadata) - KNOWN_METADATA_KEYS
            assert not unknown, f"{obs.name} used unknown metadata key(s) {unknown}"
        for payload in [*obs.inputs, *obs.metadata_dicts]:
            assert "persona" not in payload, f"{obs.name} leaked persona into a span"
