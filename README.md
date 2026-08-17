# CareDesk

CareDesk is an AI support agent for a healthcare provider. Patients and
clinic staff ask questions in natural language; the system retrieves
relevant material from a document corpus and either resolves the query
directly, guides the user through a process, or escalates to a human.

> **Data note:** CareDesk uses synthetic and public data only. It has
> never held real patient information, and none should be added to
> this repository or its data stores.

## Core loop

Every query moves through the same eight stages:

1. **Ingest** — source documents are loaded, chunked, embedded, and indexed.
2. **Classify** — the incoming query is categorized (intent, persona, urgency).
3. **Retrieve** — vector and keyword search return candidate chunks.
4. **Rerank** — candidates are reranked down to the most relevant set.
5. **Decide** — routing logic chooses to resolve, guide, clarify, escalate, or refuse.
6. **Generate** — a grounded answer is produced from the selected context, where applicable.
7. **Trace** — the full pipeline run is traced end-to-end via Langfuse.
8. **Capture** — the conversation and outcome are persisted for review and evaluation.

## Stack

| Concern                | Choice                                              |
|-------------------------|------------------------------------------------------|
| Language / packaging    | Python 3.11+, managed with [uv](https://docs.astral.sh/uv/) |
| API                      | FastAPI + uvicorn                                    |
| Relational + vector store | Postgres with pgvector                             |
| Cache                    | Redis (semantic cache, added later)                  |
| LLMs                     | OpenAI SDK — `text-embedding-3-small`, `gpt-4o-mini`, `gpt-4o` |
| Tracing                  | Langfuse                                             |
| Operator UI              | Streamlit (ask screen, human review queue)           |
| Testing / quality        | pytest, ruff, mypy, pre-commit                       |

Orchestration is hand-rolled by design — no LangChain, LlamaIndex, or
LangGraph. LangGraph is planned as a deliberate refactor in month 4,
once the hand-rolled pipeline's behavior is well understood.

## Project layout

```
src/caredesk/
  api/            FastAPI app, routers, request/response schemas
  ingestion/      document loaders, chunkers, embedding + indexing
  retrieval/      vector search, keyword search, fusion, reranking
  generation/     prompt templates, grounded answer generation
  decision/       routing logic (resolve/guide/clarify/escalate/refuse)
  observability/  langfuse tracing helpers, metrics
  storage/        db session, models, repositories
  config.py       pydantic-settings configuration
data/corpus/      raw source documents (gitignored except manifest.json)
evals/            eval cases, harness, committed result JSONs
tests/            mirrors src structure
scripts/          one-off CLI utilities
ui/               streamlit apps
```

This is commit 1: repo scaffold and dependency management only. Every
module under `src/caredesk/` is a docstring-only stub — no business
logic is implemented yet.

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
# 1. Install dependencies (creates .venv automatically)
make install

# 2. Configure environment
cp .env.example .env
# then fill in OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# 3. Start Postgres (pgvector) and Redis
make db-up

# 4. Run the API
make dev

# 5. (separate terminal) Run the Streamlit UI
make ui
```

See the [Makefile](Makefile) for the full list of available targets:
`install`, `dev`, `ui`, `test`, `lint`, `typecheck`, `db-up`, `db-down`.
