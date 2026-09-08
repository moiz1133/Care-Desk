# Observability: trace hierarchy and field reference

This is the reference to check when a dashboard panel shows nothing and
the question is "what is this field actually called, and where is it
written." It documents what commits 8 and 9 built (`caredesk.observability`
+ `caredesk.api.middleware`), not aspirational structure. `scripts/verify_traces.py`
(commit 10) generates a batch of real traces to check this against by eye.

## Span hierarchy

```
trace: caredesk.query
  span: retrieve
    span: embed_query      (generation -- it's a model call)
    span: vector_search
  span: generate           (generation)
  span: verify_citations
```

- One trace per `/query` request, opened in `services.pipeline.run_query_pipeline`
  via `observability.tracing.start_query_trace`.
- `retrieve`, `generate`, and `verify_citations` are direct children of the
  trace root, in that order (`generate` and `verify_citations` are
  siblings, not nested -- `verify_citations` only exists as a separate
  span because it's a distinct unit of work, checked *after* the model
  call returns).
- `embed_query` and `vector_search` are the only two spans nested under
  `retrieve`, not under the root -- both live inside `retrieval.vector.retrieve`.
- Pre-model refusals (`no_results`, `low_relevance`) never open a `generate`
  or `verify_citations` span at all -- `generate_answer` returns before
  reaching that code. A `retrieve`-only trace with a refused outcome is
  expected for those two reasons, not a bug.
- `model_insufficient` refusals (the model returns the exact
  `INSUFFICIENT_CONTEXT` sentinel) open `generate` but never
  `verify_citations` -- citation checking only runs once there's a
  non-sentinel answer to check.

### Where each span is opened

| Span | File | Function |
|---|---|---|
| `caredesk.query` (root) | `observability/tracing.py` | `start_query_trace` |
| `retrieve` | `retrieval/vector.py` | `retrieve` |
| `embed_query` | `retrieval/vector.py` | `retrieve` (inside the `retrieve` span) |
| `vector_search` | `retrieval/vector.py` | `retrieve` (inside the `retrieve` span) |
| `generate` | `generation/generator.py` | `generate_answer` |
| `verify_citations` | `generation/generator.py` | `generate_answer` |

Nesting is never passed explicitly -- `start_span`/`start_generation`
(`observability/tracing.py`) create a child of whatever Langfuse
observation is currently active in context, so opening a new span anywhere
in the call graph nests it correctly with zero extra wiring. That's also
why `retrieve()`/`generate_answer()` never take a "trace" argument.

### Where Week 3 and Week 4 slot in

- **Week 3 (`classify`)**: a `classify` span goes in *ahead of* `retrieve`,
  as another direct child of the trace root, once model-tiering decides
  which retrieval/generation path to take before retrieval runs.
- **Week 4 (`decide`)**: a `decide` span goes *between* `generate` and
  `verify_citations` (or after `verify_citations`, depending on whether the
  decision engine needs the verified citation list as an input -- not yet
  settled), once resolve/guide/clarify/escalate routing exists.

Both are additions, not restructuring: the trace root and the
`retrieve`/`generate`/`verify_citations` shape stay exactly as documented
above.

## Trace-level fields

Written once, in exactly one place, per the rule that an invariant-for-the-request
value (persona, conversation, client) belongs on the trace, never repeated
onto every child span.

| Field | Value | Set in | When |
|---|---|---|---|
| `trace_id` | `request_id` with dashes stripped (lossless, reversible -- not a hash) | `tracing._trace_id_for` | trace creation |
| Langfuse `tags` | `[persona, client]` | `tracing.start_query_trace` (`propagate_attributes`) | trace creation |
| Langfuse `session_id` | `conversation_id` | `tracing.start_query_trace` (`propagate_attributes`) | trace creation |
| Langfuse `user_id` | `conversation_id` | `tracing.start_query_trace` (`propagate_attributes`) | trace creation |
| `trace_name` | `"caredesk.query"` | `tracing.start_query_trace` (`propagate_attributes`) | trace creation |
| `input` | the query string | `tracing.start_query_trace` | trace creation |
| `output` | the answer text, or the refusal reason | `tracing.QueryTrace.finalize` | trace finalize |

There is no separate CareDesk "user" identity yet -- `user_id` is set to
`conversation_id` because that's the only stable identity that exists
pre-Week-4. Don't read `user_id` as a real end-user filter until that
changes.

### Trace metadata keys (`observability.vocabulary.TraceMetadataKey`)

Written at trace creation -- invariant for the whole request:

| Key | Meaning |
|---|---|
| `persona` | `"patient"` \| `"staff"`, from the validated request body |
| `client` | `"api"` \| `"eval"` \| `"cli"`, from the `X-Client` header (`observability.vocabulary.ClientType`) |
| `turn_index` | `0` today; Week 4 increments it per conversation turn |
| `environment` | `Settings.environment` (`dev`/`staging`/`prod`) |

Written at trace finalize -- the pipeline's outcome, once known:

| Key | Meaning |
|---|---|
| `k` | resolved `k` actually used (request override, or `Settings.vector_retrieval_k`) |
| `answered` | `not refused` |
| `refused` | whether generation refused |
| `refusal_reason` | one of `no_results` / `low_relevance` / `model_insufficient` / `no_citations` / `hallucinated_citation`, or `None` |
| `retrieval_strategy` | currently always `"vector_only"` |
| `results_returned` | number of chunks retrieval returned |
| `prompt_version` | e.g. `"v1"` (`generation.prompts.PROMPT_VERSION`) |
| `citation_count` | number of verified citations in the answer |
| `total_latency_ms` | end-to-end pipeline latency |

`retrieval_strategy` and `prompt_version` are recorded here, in metadata,
rather than as Langfuse tags: Langfuse's tag/session propagation has to
run *before* any span is created to apply correctly, but both values are
only known after the pipeline has already run. `persona` and `client` are
the two values known early enough to be real tags; these two aren't.

## Span-level fields (`observability.vocabulary.SpanMetadataKey`)

Per-span detail only -- never persona/conversation_id/client/environment,
which every span already inherits automatically via Langfuse's
`propagate_attributes` set once at the trace root.

| Key | Used on | Meaning |
|---|---|---|
| `strategy` | `retrieve` | `"vector_only"` |
| `source_type_filter` | `retrieve` | optional `source_type` filter list, or `None` |
| `cache_hit` | `embed_query` | whether the exact-match query-embedding cache hit |
| `latency_ms` | `embed_query`, `vector_search` | that sub-step's own latency |
| `k` | `vector_search` | resolved top-k passed to the SQL query |
| `persona_filter_applied` | `vector_search` | the `persona_visibility` values allowed through the `WHERE` clause (e.g. `["patient", "both"]`) |
| `persona_filter_excluded_count` | `vector_search` | always `None` today -- computing it needs a second, unfiltered query, which isn't worth doubling every retrieval's DB cost for. The key exists in the vocabulary from day one so a cheap answer (e.g. a cached per-corpus persona/source_type histogram) can fill it in later without a new field. |

Other per-span `input`/`output` worth knowing about, not part of the
metadata vocabulary:

- `retrieve` span `output`: `results_returned`, `top_score`, `bottom_score`.
- `vector_search` span `output`: ranked `(chunk_id, score)` pairs only --
  chunk text (possibly clinical leaflet content) never lands here.
- `generate` span `input`: the fully rendered prompt (system + user
  messages, formatted context included) -- deliberate, so a wrong answer
  can be traced back to exactly what the model saw. This means retrieved
  corpus content lands in the trace; accepted for Week 1's debuggability
  need.
- `generate` span `usage_details`/`cost_details`: input/output token
  counts and cost, rolling up to the trace.
- `verify_citations` span `input`: the `[chunk_id]`s parsed from the
  answer. `output`: `verified_ids`, `hallucinated_ids`, `outcome`
  (`"verified"` / `"no_citations"` / `"hallucinated_citation"`). Marked
  `level="ERROR"` when `hallucinated_ids` is non-empty.

## Closed vocabulary enforcement

Every metadata key written anywhere is checked against
`observability.vocabulary.KNOWN_METADATA_KEYS`
(`TRACE_METADATA_KEYS | SPAN_METADATA_KEYS`) by `tracing._check_metadata_keys`,
called from both `SpanHandle.update` and `_observation`. An unlisted key
doesn't raise -- it logs `unknown_metadata_key` and still gets sent -- but
it's caught at run time, not just by code-review convention. Add a new key
to `TraceMetadataKey` or `SpanMetadataKey` before writing it anywhere, not
after.

## Sampling and forced tracing

`Settings.trace_sample_rate` (default `1.0`) gates whether the *detailed*
child spans (`embed_query`, `vector_search`, `generate`, `verify_citations`)
are recorded for real (`observability/context.py`, `TraceState.record_detail`).
The trace root itself is always recorded regardless of sampling -- that's
what makes "errors and refusals are always traced" true without buffering
every span until the outcome is known. See the docstrings on
`TraceState` and `tracing.start_query_trace` for the exact mechanics and
the honest limitation (a span already skipped before an error/refusal
becomes known isn't retroactively upgraded; only the root's own final
record, and anything created after that point, is unconditional).

## No-op mode

If `Settings.langfuse_enabled` is `False`, or `langfuse_public_key`/
`langfuse_secret_key` are absent, every span/generation helper in
`tracing.py` degrades to a no-op transparently (`get_langfuse_client`
returns `None`). No caller anywhere branches on this. Background jobs and
one-off scripts (`scripts/ask.py`, `scripts/search.py`) that call
`retrieve()`/`generate_answer()` outside of any HTTP request get a
placeholder `RequestContext` (`persona`/`conversation_id` = `"unknown"`,
`client` = `"cli"`) rather than raising -- see
`tracing._placeholder_request_context`.

## Verifying this by hand

`scripts/verify_traces.py` fires a fixed, labeled batch of requests (every
resolved/refused/persona-blocked/cached/multi-turn/eval-client path) at a
running `/query` endpoint and prints the trace URLs plus a manual
checklist covering everything in this document. Run it after touching
anything in `observability/` or the pipeline's tracing calls, before
trusting a dashboard built on top of these fields.
