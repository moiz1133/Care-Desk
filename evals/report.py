"""Eval report formatting: report.md content and the console headline table.

Pure formatting. No metric computation (evals/metrics.py), no HTTP calls
(evals/harness.py) -- everything printed here was already computed
elsewhere and is just being laid out for a human to read.
"""

from __future__ import annotations

from typing import Any

from evals.harness import CaseResult
from evals.metrics import CaseMetricInput

# The six numbers printed first, in report.md and at the end of
# scripts/run_eval.py's console output -- kept as one list so both call
# sites can never drift apart.
HEADLINE_METRICS: list[tuple[str, str]] = [
    ("decision_accuracy_overall", "Decision accuracy (overall)"),
    ("decision_accuracy_implementable", "Decision accuracy (implementable slices)"),
    ("recall_at_5", "Recall@5"),
    ("mrr", "MRR"),
    ("refusal_rate_on_unanswerable", "Refusal rate on unanswerable"),
    ("false_refusal_rate_on_easy", "False refusal rate on easy"),
]


def _headline_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "decision_accuracy_overall": metrics["decision"]["accuracy_overall"],
        "decision_accuracy_implementable": metrics["decision"]["accuracy_implementable"],
        "recall_at_5": metrics["retrieval"]["recall_at_5"],
        "mrr": metrics["retrieval"]["mrr"],
        "refusal_rate_on_unanswerable": metrics["decision"]["refusal_rate_on_unanswerable"],
        "false_refusal_rate_on_easy": metrics["decision"]["false_refusal_rate_on_easy"],
    }


def format_headline_table(metrics: dict[str, Any]) -> str:
    """Plain-text table for the console, printed at the end of a run."""
    values = _headline_values(metrics)
    width = max(len(label) for _, label in HEADLINE_METRICS)
    lines = ["Headline numbers", "-" * (width + 10)]
    for key, label in HEADLINE_METRICS:
        lines.append(f"{label.ljust(width)}  {values[key]:.3f}")
    return "\n".join(lines)


def _first_hit_rank(expected: list[str], retrieved_ranked: list[str]) -> int | None:
    expected_set = set(expected)
    for rank, doc_id in enumerate(retrieved_ranked, start=1):
        if doc_id in expected_set:
            return rank
    return None


def worst_retrieval_cases(
    case_inputs: list[CaseMetricInput], *, limit: int = 10
) -> list[tuple[CaseMetricInput, int | None]]:
    """The `limit` worst cases by retrieval rank: a case where no expected
    doc was retrieved at all sorts worse than any case where one was
    found, however late."""
    scored = [case for case in case_inputs if case.expected_doc_ids and not case.failed]

    def sort_key(case: CaseMetricInput) -> tuple[int, int]:
        rank = _first_hit_rank(case.expected_doc_ids, case.retrieved_doc_ids_ranked)
        return (0, 0) if rank is None else (1, -rank)

    ordered = sorted(scored, key=sort_key)
    return [
        (case, _first_hit_rank(case.expected_doc_ids, case.retrieved_doc_ids_ranked))
        for case in ordered[:limit]
    ]


def _confusion_matrix_md(matrix: dict[str, dict[str, int]]) -> str:
    expected_order = ["RESOLVE", "CLARIFY", "ESCALATE", "REFUSE"]
    actual_columns = sorted(
        {actual for row in matrix.values() for actual in row} | {"RESOLVE", "REFUSE"}
    )

    header = "| expected \\ actual | " + " | ".join(actual_columns) + " |"
    sep = "|---" * (len(actual_columns) + 1) + "|"
    rows = [header, sep]
    for expected in expected_order:
        row = matrix.get(expected, {})
        cells = [str(row.get(col, 0)) for col in actual_columns]
        rows.append(f"| {expected} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _not_implemented_md(not_implemented: dict[str, dict[str, Any]]) -> str:
    if not not_implemented:
        return "None -- every expected_decision value is currently producible."
    lines = []
    for decision, info in sorted(not_implemented.items()):
        slices = ", ".join(info["slices"])
        lines.append(
            f"- **{decision}** ({info['count']} cases, slice(s): {slices}): "
            "not producible by the current pipeline. No decision engine exists "
            "until Week 4 -- these cases score as incorrect by construction, "
            "not because retrieval or generation failed."
        )
    return "\n".join(lines)


def _worst_cases_md(case_inputs: list[CaseMetricInput], *, limit: int = 10) -> str:
    worst = worst_retrieval_cases(case_inputs, limit=limit)
    if not worst:
        return "No scored cases."
    lines = [
        "| case_id | query | expected docs | retrieved (ranked) | first hit rank |",
        "|---|---|---|---|---|",
    ]
    for case, rank in worst:
        retrieved = ", ".join(case.retrieved_doc_ids_ranked[:5]) or "(none)"
        expected = ", ".join(case.expected_doc_ids)
        rank_str = str(rank) if rank is not None else "not found"
        query = case.query.replace("|", "\\|")
        lines.append(f"| {case.case_id} | {query} | {expected} | {retrieved} | {rank_str} |")
    return "\n".join(lines)


def _hallucinated_citations_md(results: list[CaseResult]) -> str:
    incidents = [r for r in results if r.primary.refusal_reason == "hallucinated_citation"]
    if not incidents:
        return "None."
    lines = [
        "The specific fabricated chunk_id(s) aren't in the /query response "
        "(the API returns citations=[] on a hallucination, by design -- see "
        "generation/generator.py) -- look up request_id in Langfuse for the "
        "verify_citations span's hallucinated_ids if you need that detail.",
        "",
        "| case_id | query | persona | request_id |",
        "|---|---|---|---|",
    ]
    for result in incidents:
        query = result.query.replace("|", "\\|")
        lines.append(
            f"| {result.case_id} | {query} | {result.persona} | {result.primary.request_id} |"
        )
    return "\n".join(lines)


def render_report_md(
    *,
    metrics: dict[str, Any],
    case_inputs: list[CaseMetricInput],
    results: list[CaseResult],
    config: dict[str, Any],
) -> str:
    run = metrics["run"]
    retrieval = metrics["retrieval"]
    decision = metrics["decision"]
    citation = metrics["citation"]
    cost_latency = metrics["cost_latency"]

    parts = [
        "# CareDesk eval report",
        "",
        f"- timestamp: {config.get('timestamp')}",
        f"- git sha: {config.get('git_sha')}  (dirty: {config.get('git_dirty')})",
        f"- tag: {config.get('tag')}",
        f"- config: chunk={config.get('chunk_strategy')} "
        f"retrieval={config.get('retrieval_strategy')} "
        f"k={config.get('k')} embedding_model={config.get('embedding_model')} "
        f"generator_model={config.get('generator_model')} "
        f"temperature={config.get('temperature')} "
        f"min_relevance_score={config.get('min_relevance_score')} "
        f"prompt_version={config.get('prompt_version')}",
        f"- cases: {run['total_cases']}  failed: {run['failed_cases']}  "
        f"retried: {run['retried_cases']}",
        "",
        "**Scope**: this measures retrieval and decision routing, not answer quality. "
        "Answer text is recorded in raw.jsonl for manual review but is not scored -- "
        "automated answer grading (LLM-as-judge) is Month 5, DeepEval.",
        "",
        "## Headline numbers",
        "",
        format_headline_table(metrics).replace("\n", "  \n"),
        "",
        "## Per-slice decision accuracy",
        "",
        "\n".join(
            f"- {slice_name}: {acc:.3f}"
            for slice_name, acc in sorted(decision["per_slice_accuracy"].items())
        ),
        "",
        "## Not yet implemented",
        "",
        _not_implemented_md(decision["not_yet_implemented"]),
        "",
        "## Confusion matrix (expected x actual)",
        "",
        _confusion_matrix_md(decision["confusion_matrix"]),
        "",
        "## Retrieval",
        "",
        "\n".join(
            f"- recall@{k}: {retrieval[f'recall_at_{k}']:.3f}   "
            f"full_recall@{k}: {retrieval[f'full_recall_at_{k}']:.3f}   "
            f"precision@{k}: {retrieval[f'precision_at_{k}']:.3f}"
            for k in (1, 3, 5, 10)
        ),
        f"- mrr: {retrieval['mrr']:.3f}",
        f"- ndcg@5: {retrieval['ndcg_at_5']:.3f}",
        "",
        "### Unanswerable slice",
        "",
        f"- false_retrieval_rate: {retrieval['unanswerable']['false_retrieval_rate']:.3f}",
        f"- mean_top_score: {retrieval['unanswerable']['mean_top_score']}",
        "",
        "## Safety-relevant decision metrics",
        "",
        f"- refusal_rate_on_unanswerable: {decision['refusal_rate_on_unanswerable']:.3f}",
        f"- false_refusal_rate_on_easy: {decision['false_refusal_rate_on_easy']:.3f}",
        f"- escalation_recall: {decision['escalation_recall']:.3f}  "
        "(0.0 expected -- ESCALATE not implemented)",
        "",
        "## Citations",
        "",
        f"- hallucinated_citation_rate: {citation['hallucinated_citation_rate']:.3f}",
        f"- citation_count_mean: {citation['citation_count_mean']:.3f}",
        f"- answers_with_zero_citations: {citation['answers_with_zero_citations']}  "
        "(should be 0 by construction)",
        "",
        "## Cost and latency",
        "",
        f"- total_cost_usd: {cost_latency['total_cost_usd']:.4f}   "
        f"(this run only, primary calls; the k={config.get('eval_retrieval_diagnostic_k')} "
        f"retrieval-diagnostic calls cost separately -- see config.json)",
        f"- cost_per_case_usd: {cost_latency['cost_per_case_usd']:.4f}",
        f"- total_input_tokens: {int(cost_latency['total_input_tokens'])}   "
        f"total_output_tokens: {int(cost_latency['total_output_tokens'])}",
        f"- total_latency_ms: p50={cost_latency['total_latency_ms']['p50']}  "
        f"p95={cost_latency['total_latency_ms']['p95']}",
        f"- retrieval_latency_ms: p50={cost_latency['retrieval_latency_ms']['p50']}  "
        f"p95={cost_latency['retrieval_latency_ms']['p95']}",
        f"- generation_latency_ms: p50={cost_latency['generation_latency_ms']['p50']}  "
        f"p95={cost_latency['generation_latency_ms']['p95']}",
        f"- embed_cache_hit_rate: {cost_latency['embed_cache_hit_rate']:.3f}",
        "",
        "## 10 worst cases by retrieval rank",
        "",
        _worst_cases_md(case_inputs, limit=10),
        "",
        "## Hallucinated citation incidents",
        "",
        _hallucinated_citations_md(results),
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

_COMPARE_METRIC_PATHS: list[tuple[str, tuple[str, ...]]] = [
    ("decision_accuracy_overall", ("decision", "accuracy_overall")),
    ("decision_accuracy_implementable", ("decision", "accuracy_implementable")),
    ("recall_at_1", ("retrieval", "recall_at_1")),
    ("recall_at_3", ("retrieval", "recall_at_3")),
    ("recall_at_5", ("retrieval", "recall_at_5")),
    ("recall_at_10", ("retrieval", "recall_at_10")),
    ("mrr", ("retrieval", "mrr")),
    ("ndcg_at_5", ("retrieval", "ndcg_at_5")),
    ("false_retrieval_rate_unanswerable", ("retrieval", "unanswerable", "false_retrieval_rate")),
    ("mean_top_score_unanswerable", ("retrieval", "unanswerable", "mean_top_score")),
    ("refusal_rate_on_unanswerable", ("decision", "refusal_rate_on_unanswerable")),
    ("false_refusal_rate_on_easy", ("decision", "false_refusal_rate_on_easy")),
    ("escalation_recall", ("decision", "escalation_recall")),
    ("hallucinated_citation_rate", ("citation", "hallucinated_citation_rate")),
    ("citation_count_mean", ("citation", "citation_count_mean")),
    ("cost_per_case_usd", ("cost_latency", "cost_per_case_usd")),
    ("total_latency_ms_p50", ("cost_latency", "total_latency_ms", "p50")),
    ("total_latency_ms_p95", ("cost_latency", "total_latency_ms", "p95")),
]


def _get_path(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = d
    for key in path:
        if value is None:
            return None
        value = value.get(key)
    return value


def format_compare_table(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any], *, label_a: str, label_b: str
) -> str:
    rows = [f"{'metric':35s} {label_a:>15s} {label_b:>15s} {'delta':>12s}"]
    rows.append("-" * len(rows[0]))
    for name, path in _COMPARE_METRIC_PATHS:
        value_a = _get_path(metrics_a, path)
        value_b = _get_path(metrics_b, path)
        if isinstance(value_a, int | float) and isinstance(value_b, int | float):
            delta = value_b - value_a
            rows.append(f"{name:35s} {value_a:15.4f} {value_b:15.4f} {delta:+12.4f}")
        else:
            rows.append(f"{name:35s} {str(value_a):>15s} {str(value_b):>15s} {'n/a':>12s}")
    return "\n".join(rows)


def format_verdict_changes(
    expected_by_case: dict[str, str],
    actual_a_by_case: dict[str, str | None],
    actual_b_by_case: dict[str, str | None],
) -> str:
    """Cases whose actual_decision changed between two runs, either direction.

    This is the half of a comparison that actually matters: an aggregate
    improvement can still hide previously-passing cases that regressed.
    `expected_by_case` is shared across both runs -- same dataset, same
    case_id -> expected_decision mapping regardless of which run produced
    which actual_decision.
    """
    shared_ids = sorted(set(actual_a_by_case) & set(actual_b_by_case))
    changed = [
        (case_id, actual_a_by_case[case_id], actual_b_by_case[case_id])
        for case_id in shared_ids
        if actual_a_by_case[case_id] != actual_b_by_case[case_id]
    ]
    if not changed:
        return "No case changed verdict between the two runs."

    lines = [
        "| case_id | expected | run A actual | run B actual | direction |",
        "|---|---|---|---|---|",
    ]
    for case_id, before, after in changed:
        expected = expected_by_case.get(case_id, "?")
        was_correct = before == expected
        now_correct = after == expected
        if not was_correct and now_correct:
            direction = "IMPROVED"
        elif was_correct and not now_correct:
            direction = "REGRESSED"
        else:
            direction = "changed (still wrong either way)"
        lines.append(f"| {case_id} | {expected} | {before} | {after} | {direction} |")
    return "\n".join(lines)
