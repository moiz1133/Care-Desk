# CareDesk eval report

- timestamp: 20260908T144946Z
- git sha: 8237806386d8cb39930b6537692879fe115c53a4  (dirty: False)
- tag: baseline
- config: chunk=fixed_512_50 retrieval=vector_only k=5 embedding_model=text-embedding-3-small generator_model=gpt-4o temperature=0.0 min_relevance_score=0.3 prompt_version=v1
- cases: 50  failed: 0  retried: 1

**Scope**: this measures retrieval and decision routing, not answer quality. Answer text is recorded in raw.jsonl for manual review but is not scored -- automated answer grading (LLM-as-judge) is Month 5, DeepEval.

## Headline numbers

Headline numbers  
--------------------------------------------------  
Decision accuracy (overall)               0.460  
Decision accuracy (implementable slices)  0.920  
Recall@5                                  0.974  
MRR                                       0.879  
Refusal rate on unanswerable              1.000  
False refusal rate on easy                0.133

## Per-slice decision accuracy

- ambiguous: 0.000
- easy: 0.867
- escalate: 0.000
- unanswerable: 1.000

## Not yet implemented

- **CLARIFY** (12 cases, slice(s): ambiguous): not producible by the current pipeline. No decision engine exists until Week 4 -- these cases score as incorrect by construction, not because retrieval or generation failed.
- **ESCALATE** (13 cases, slice(s): escalate): not producible by the current pipeline. No decision engine exists until Week 4 -- these cases score as incorrect by construction, not because retrieval or generation failed.

## Confusion matrix (expected x actual)

| expected \ actual | REFUSE | RESOLVE |
|---|---|---|
| RESOLVE | 2 | 13 |
| CLARIFY | 5 | 7 |
| ESCALATE | 11 | 2 |
| REFUSE | 10 | 0 |

## Retrieval

- recall@1: 0.821   full_recall@1: 0.821   precision@1: 0.821
- recall@3: 0.923   full_recall@3: 0.923   precision@3: 0.308
- recall@5: 0.974   full_recall@5: 0.974   precision@5: 0.195
- recall@10: 0.974   full_recall@10: 0.974   precision@10: 0.097
- mrr: 0.879
- ndcg@5: 0.903

### Unanswerable slice

- false_retrieval_rate: 0.800
- mean_top_score: 0.3617086131902051

## Safety-relevant decision metrics

- refusal_rate_on_unanswerable: 1.000
- false_refusal_rate_on_easy: 0.133
- escalation_recall: 0.000  (0.0 expected -- ESCALATE not implemented)

## Citations

- hallucinated_citation_rate: 0.000
- citation_count_mean: 1.000
- answers_with_zero_citations: 0  (should be 0 by construction)

## Cost and latency

- total_cost_usd: 0.1569   (this run only, primary calls; the k=10 retrieval-diagnostic calls cost separately -- see config.json)
- cost_per_case_usd: 0.0031
- total_input_tokens: 57394   total_output_tokens: 1338
- total_latency_ms: p50=1547.4384003318846  p95=4781.9549003615975
- retrieval_latency_ms: p50=55.952499620616436  p95=62.55930010229349
- generation_latency_ms: p50=1491.0284001380205  p95=4724.521900061518
- embed_cache_hit_rate: 0.000

## 10 worst cases by retrieval rank

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

## Hallucinated citation incidents

None.
