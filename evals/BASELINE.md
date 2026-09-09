# Week 1 baseline

**What this is, and is not**: this measures retrieval quality and decision
routing against the commit-11 dataset. It does not measure answer quality
-- generated answer text is recorded in `raw.jsonl` for manual review, but
nothing scores it. Automated answer grading (LLM-as-judge) is Month 5,
with DeepEval.

Run directory: `evals/results/20260908T144946Z_82378063/` (also pointed to
by `evals/results/baseline.json`). Full detail: `report.md` in that
directory, `metrics.json` for the raw numbers, `raw.jsonl` for per-case
requests/responses/timings, `config.json` for the full environment record.

## Config

| | |
|---|---|
| chunk strategy | fixed_512_50 |
| chunk_size | 512 |
| chunk_overlap | 50 |
| retrieval | vector_only |
| k | 5 |
| embedding model | text-embedding-3-small |
| generator model | gpt-4o |
| temperature | 0.0 |
| min_relevance_score | 0.3 |
| prompt_version | v1 |
| git sha | `8237806386d8cb39930b6537692879fe115c53a4` |
| git dirty | false |
| corpus | 50 documents, 50 chunks (every document fits in one `fixed_512_50` chunk) |
| index build timestamp | 2026-09-01T11:01:38.110702+00:00 |

## Headline numbers

| Metric | Value |
|---|---|
| Decision accuracy (overall) | 0.460 |
| Decision accuracy (implementable slices) | 0.920 |
| Recall@5 | 0.974 |
| MRR | 0.879 |
| Refusal rate on unanswerable | 1.000 |
| False refusal rate on easy | 0.133 |

## CLARIFY and ESCALATE are not implemented

**Read this before the overall accuracy number above.** The pipeline has
no decision engine until Week 4 -- it only ever resolves or refuses.
CLARIFY and ESCALATE are not producible by any input, so every case
expecting one scores as incorrect by construction, not because retrieval
or generation did anything wrong:

- **CLARIFY**: 12 cases (the `ambiguous` slice, in full)
- **ESCALATE**: 13 cases (the `escalate` slice, in full)

That's 25 of the 50 cases -- half the dataset -- guaranteed wrong today
regardless of pipeline quality. `decision_accuracy_overall` (0.460)
includes them; `decision_accuracy_implementable` (0.920) excludes them,
scoring only the 25 cases (easy + unanswerable) whose expected decision
(RESOLVE / REFUSE) the pipeline can actually produce. Read the
implementable number as "how the pipeline is doing," and the overall
number as "how much of the dataset the pipeline can act on at all" --
they answer different questions.

## Per-slice decision accuracy

| Slice | Accuracy | Note |
|---|---|---|
| easy | 0.867 | 13/15 correctly resolved |
| unanswerable | 1.000 | 10/10 correctly refused |
| ambiguous | 0.000 | not implemented -- see above |
| escalate | 0.000 | not implemented -- see above |

## Unanswerable slice: mean_top_score

**mean_top_score: 0.362** (threshold is 0.3). `false_retrieval_rate: 0.800`
-- 8 of the 10 unanswerable cases had a top retrieval score at or above
the pre-model refusal threshold, meaning the model was called (and still
correctly refused, every time -- see refusal_rate_on_unanswerable above)
rather than being screened out by score alone. This is the number that
shows how close the irrelevant matches sit to the threshold; Week 2's
retrieval work is expected to move it.

## 10 worst cases by retrieval rank

Verbatim from `report.md`. "not found" means no expected doc appeared in
the top 10 retrieved (across both the k=5 primary call and the k=10
retrieval-diagnostic call); a number is the rank the first expected doc
was found at.

| case_id | query | expected docs | retrieved (ranked) | first hit rank |
|---|---|---|---|---|
| esc_011 | a patient just told me they doubled up on their blood pressure pill by accident what do i do | rb_clinical_escalation | leaflet_lisinopril, leaflet_ibuprofen, ticket_0009, leaflet_atorvastatin, leaflet_omeprazole | not found |
| esc_013 | my doctor changed my blood pressure dose last visit but i want to go back to the old one since i felt better on it is that ok | leaflet_lisinopril | ticket_0009, faq_prescription_refills, ticket_0010, faq_second_opinion, leaflet_lisinopril | 5 |
| easy_011 | do i need my regular doctor to approve seeing a high risk pregnancy specialist | policy_maternity | policy_specialist_referral, faq_second_opinion, ticket_0010, policy_maternity, ticket_0002 | 4 |
| amb_009 | is my colonoscopy covered | policy_preventive_care | ticket_0002, policy_imaging, policy_preventive_care, ticket_0005, policy_surgery_preauth | 3 |
| easy_004 | does it cost anything to send my file to my new doctor | faq_records_request | ticket_0007, faq_records_request, faq_new_patient_registration, ticket_0010, ticket_0002 | 2 |
| easy_006 | is there anything i shouldnt drink a lot of with this med | leaflet_atorvastatin | leaflet_ibuprofen, leaflet_atorvastatin, leaflet_lisinopril, leaflet_metformin, leaflet_omeprazole | 2 |
| esc_010 | can i take my allergy meds and my antidepressant on the same day | leaflet_sertraline | leaflet_amoxicillin, leaflet_sertraline, leaflet_lisinopril, leaflet_ibuprofen, leaflet_albuterol | 2 |
| amb_001 | do i need a referral to see a specialist | policy_specialist_referral | policy_specialist_referral, faq_second_opinion, ticket_0006, ticket_0007, ticket_0005 | 1 |
| amb_002 | whats my copay for an mri | policy_imaging | policy_imaging, ticket_0005, ticket_0002, policy_maternity, policy_preventive_care | 1 |
| amb_003 | hows dme cost sharing work for a patient asking about a wheelchair | policy_durable_medical_equipment | policy_durable_medical_equipment, ticket_0004, policy_preventive_care, ticket_0002, ticket_0005 | 1 |

## Other numbers from this run

- Full retrieval table (recall/full_recall/precision @ 1/3/5/10, NDCG@5),
  confusion matrix, citation metrics, and cost/latency breakdown by stage:
  see `report.md` in the run directory.
- hallucinated_citation_rate: 0.000. answers_with_zero_citations: 0.
- total_cost_usd: 0.1569 (primary calls only, 50 cases; the k=10
  retrieval-diagnostic calls used for recall@10 cost separately -- both
  are broken out in `config.json`/`metrics.json`).

## A note on reproducibility

Two full runs were made at this exact config (same git sha, same
concurrency) while producing this baseline. Retrieval numbers were
identical between them (recall@5, MRR, and the worst-10 list did not
change at all). Decision accuracy differed by one case
(false_refusal_rate_on_easy: 1/15 in the first run, 2/15 in the second,
the one recorded above) -- temperature=0.0 reduces but does not fully
guarantee identical output from the generator model across separate API
calls. Worth knowing before reading a one-case difference in a future
comparison run as a real regression.

## Known issue found and fixed while producing this baseline

The eval run's real concurrent load surfaced a bug in `api/main.py`: all
five exception handlers passed `request_id` via `extra=` to their logger
calls, colliding with the `request_id` that `api.middleware`'s global
LogRecordFactory already stamps onto every LogRecord (commit 9). Every
5xx-class failure was crashing the handler trying to log it, escaping to
a bare framework 500 instead of the intended clean `{"detail",
"request_id"}` response. Fixed (see git log on this branch) before this
baseline was run -- unrelated to eval-config tuning, not reverted for the
baseline.
