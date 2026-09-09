# Architecture decisions

This documents the *why* behind three choices that shape the rest of the
codebase and aren't obvious from reading `src/` alone: the datastore, the
orchestration approach, and the corpus composition. It records reasoning
as of Week 1 (through the eval-harness baseline); it will be revisited,
not silently outdated, when circumstances that motivated a choice change
— see the "revisit if" line under each one.

## Why pgvector

**Decision**: Postgres + pgvector for both relational data (documents,
chunks, conversations) and vector search (HNSW index on
`chunks.embedding`), instead of a dedicated vector database (Pinecone,
Weaviate, Qdrant, etc.).

**Reasoning**:

- **One datastore, one transaction boundary.** A chunk's embedding, its
  source document metadata, its `persona_visibility` flag, and the
  conversation record that cites it are all relational data with
  relational integrity requirements (foreign keys, cascades on
  re-ingestion). Splitting embeddings into a separate vector store means
  either accepting eventual consistency between the two, or building
  and maintaining a synchronization layer — real engineering cost with
  no retrieval-quality benefit at this corpus size.
- **This corpus does not need a dedicated vector database's scale.**
  50 documents, 50 chunks today, low thousands at most as the corpus
  grows through the year. pgvector's HNSW index (`ix_chunks_embedding_hnsw`,
  `vector_cosine_ops`, see `caredesk.ingestion.indexer.ensure_hnsw_index`)
  comfortably serves that; the case for a purpose-built vector database
  is largely about scale and query patterns (billions of vectors,
  sub-millisecond ANN at high QPS) this project isn't near.
- **`persona_visibility` is a security control, not a ranking feature**
  (see `retrieval/vector.py`'s docstring). It has to be enforced as a
  hard filter on every query — a patient must never see a
  staff-only chunk, regardless of similarity score. That's a `WHERE`
  clause against a column in the same row as the embedding. In a
  separate vector store, the same guarantee requires either duplicating
  visibility metadata into the vector store's filter syntax and keeping
  it in sync, or a two-hop query (vector store, then a Postgres lookup
  to re-check visibility) — more moving parts, and an easy place for
  the sync to drift and quietly leak content across personas.
- **Fewer moving parts to run and pay for.** No second managed service
  and its own auth, network path, and failure mode; one connection pool,
  one backup story, one thing to reason about in an incident.
- The index is built once, after bulk load, not incrementally per-insert
  (`ensure_hnsw_index` runs after ingestion, deliberately — building HNSW
  against an empty table then inserting is slower than bulk-inserting
  then indexing). That ingestion-time ordering is easy to control because
  ingestion and indexing are already the same process talking to the
  same database.

**Explicitly not chosen**: a dedicated vector database. Nothing above is
a claim that pgvector outperforms one — at large scale or high-QPS
concurrent load it likely doesn't. The claim is that this corpus and
this load profile don't exercise that difference, and the integration,
consistency, and access-control cost of a second datastore isn't paid
for by a benefit CareDesk doesn't yet need.

**Revisit if**: corpus size moves from low-thousands into the
hundreds-of-thousands-plus range, query concurrency grows enough that
HNSW recall/latency at that scale becomes the bottleneck, or a future
requirement needs vector-database-specific features (e.g. sharded
multi-tenant indexes) that pgvector doesn't offer.

## Why hand-rolled orchestration

**Decision**: the query pipeline (`services/pipeline.py` —
retrieve → decide → generate → verify citations, see
[docs/observability.md](observability.md) for the exact span shape) is
plain Python function calls, not a framework (LangChain, LlamaIndex,
LangGraph). This is stated as a deliberate choice in the root
[README](../README.md), not an oversight — worth expanding on here.

**Reasoning**:

- **The pipeline is currently linear and small enough that a framework
  buys nothing.** Retrieve, decide, generate, verify — four stages,
  each a straightforward function call with typed inputs and outputs.
  A graph/orchestration framework earns its complexity when control
  flow is genuinely non-linear (branching, retries, cycles, parallel
  fan-out/fan-in across many steps) or when the boundary between
  "framework's job" and "app's job" is doing real work. Neither is true
  yet — every routing decision here is one dataclass in, one dataclass
  (`RetrievalResponse`, `CaseResult`, etc.) out.
- **Debuggability.** Every stage of this pipeline handles PHI-adjacent
  healthcare queries and has to be traceable, testable in isolation, and
  auditable by a human reading the code — not by reading framework
  internals to understand what actually ran. A stack trace from a plain
  function call points at CareDesk's own code; a stack trace from inside
  a framework's executor graph often points at the framework first. When
  something like the request_id logging collision surfaced during the
  Week 1 eval run (see `evals/BASELINE.md`), it was diagnosable by
  reading two files and a log grep — that tractability is worth
  preserving deliberately, not an accident of the codebase being young.
- **The observability model is custom-built** (`observability/tracing.py`,
  Langfuse spans keyed to this project's own stage names) and hand-rolled
  orchestration means there's no framework-imposed span/callback model to
  reconcile it against. The trace hierarchy documented in
  [docs/observability.md](observability.md) is exactly the pipeline's
  real call structure, not a framework's abstraction of it.
- **No premature lock-in to a framework's opinions about state,
  memory, or agent loops** before this project has decided it needs
  them. The decision engine (`decision/router.py`) is still a stub —
  committing to LangGraph's state-machine model now would mean guessing
  at the shape of routing logic that doesn't exist yet.

**Explicitly not chosen**: LangChain, LlamaIndex, LangGraph, or similar,
for now. This is not a claim that such frameworks are poorly built or
wrong for other projects — including, plausibly, this one later. It's a
sequencing choice: understand the hand-rolled pipeline's real behavior
first, adopt a framework second, once there's a concrete shape (branching
decision states, multi-turn agent loops, tool-calling fan-out) that
actually needs what a framework provides.

**Revisit if**: the pipeline gains real non-linear control flow — the
Week 4 decision engine, multi-turn conversation state, or agentic
tool use are the likely triggers. The root README already commits to
evaluating a LangGraph refactor in month 4 for exactly this reason —
that date was chosen because it's after the decision engine ships, not
before, so there's real routing logic to migrate rather than a graph
built ahead of its own requirements.

## Why this corpus mix

**Decision**: the Week 1 corpus (`data/corpus/`, 50 documents, indexed
1:1 into 50 chunks at `chunk_size=512`) deliberately mixes five source
types rather than being drawn from one:

| source_type | count | persona_visibility |
|---|---|---|
| faq_markdown | 15 | patient |
| medication_leaflet | 10 | both |
| resolved_ticket | 10 | both |
| policy_pdf | 8 | both / staff |
| staff_runbook | 7 | staff |

(26 documents visible to both personas, 15 patient-only, 9 staff-only —
see `manifest.json`'s `persona_visibility` field.)

**Reasoning**:

- **The system has two personas with different access rights, and the
  corpus has to actually exercise that split.** A corpus that's
  entirely patient-facing FAQs would never test the
  `persona_visibility` filter in `retrieval/vector.py` — the one hard
  security control in the retrieval path. Staff runbooks and internal
  policy detail exist in the corpus specifically so a patient-scoped
  query against staff-only content has something real to be correctly
  refused against, not just an empty table.
- **Document *type* diversity stresses chunking and retrieval
  differently than document *count* alone would.** FAQ markdown is
  short and self-contained (one question, one answer — chunk boundaries
  rarely matter). Medication leaflets are dense and repetitive across
  drugs (dosage, interactions, warnings — the same *shape* of content
  recurring, which is exactly the condition under which embedding
  similarity gets confusable, as seen directly in the Week 1 baseline's
  worst-retrieval-rank table: `leaflet_*` cases dominate it). Policy
  PDFs and staff runbooks carry cross-references and caveats that don't
  hold together if split badly. One corpus that has to serve all of
  these with a single `fixed_512_50` chunking strategy is a more honest
  test of that strategy than a corpus of only one document type would
  be — and it's why chunk strategy is explicitly named as a Week 2
  target in the baseline, rather than assumed fine.
- **`resolved_ticket` documents encode real support-interaction
  phrasing** (how patients actually ask things, informally) rather than
  only reference-document phrasing (how policy is officially written).
  The eval dataset's queries are written in that same informal register
  deliberately — retrieval that only works against
  policy-document-phrased queries would be measuring something
  narrower than what the system will actually receive.
- **The mix is what makes REFUSE/decision-routing evaluable at all.**
  The eval dataset's `unanswerable` slice (see `evals/BASELINE.md`)
  needs genuinely out-of-corpus queries to test the
  `min_relevance_score` refusal path; the `escalate` slice needs
  content (`rb_clinical_escalation`, medication-interaction leaflets)
  that's in-corpus but requires human judgment, not just a lookup. A
  single-source-type corpus would collapse some of these slices into
  each other.
- **Every document is synthetic** ("Synthetic — authored for CareDesk.
  Fictional entities only." — see `manifest.json`'s `provenance` field
  and the root README's data note). This is a hospital-adjacent project
  and the corpus needed to be safe to commit, run through third-party
  LLM APIs, and hand to anyone on the team without a PHI review — real
  clinic documents were never an option, mix or no mix.

**Explicitly not chosen**: a larger corpus, or one drawn from real
(de-identified) clinical documents. Size was kept small deliberately —
at 50 documents, every retrieval miss in the baseline's worst-case table
can be read and understood by a human in minutes; that legibility would
be lost well before a corpus reached a size that meaningfully changed
HNSW's behavior (see "why pgvector," above).

**Revisit if**: the corpus grows enough that manual review of retrieval
failures (as done for the Week 1 baseline) stops being tractable, or a
real production integration requires ingesting actual clinic content —
which would need a PHI/compliance review this corpus was explicitly
designed to avoid needing.
