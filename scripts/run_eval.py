"""CLI: run the eval dataset against a live /query endpoint, or compare two
past runs.

Usage:
  uv run python scripts/run_eval.py --tag baseline
  uv run python scripts/run_eval.py --slice easy --slice unanswerable
  uv run python scripts/run_eval.py --case-id esc_001 --case-id esc_008
  uv run python scripts/run_eval.py --no-write
  uv run python scripts/run_eval.py --compare evals/results/RUN_A evals/results/RUN_B

Requires a running server (this drives real HTTP requests against
/query, the same as scripts/verify_traces.py) -- see that script's
docstring for how to start one.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

# evals/ is a sibling of src/, not part of the installed caredesk package
# (see evals/__init__.py) -- unlike `caredesk`, which is pip-installed
# editable and importable from anywhere, running this file directly only
# puts scripts/ on sys.path, so evals needs an explicit assist. pytest
# gets this for free via pyproject.toml's `pythonpath = ["src", "."]`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caredesk.config import Settings, get_settings  # noqa: E402
from caredesk.generation.prompts import PROMPT_VERSION  # noqa: E402
from caredesk.ingestion.loader import load_manifest  # noqa: E402
from caredesk.storage.models import ChunkRecord  # noqa: E402
from caredesk.storage.session import session_scope  # noqa: E402
from evals import harness, report  # noqa: E402
from evals.metrics import compute_all_metrics  # noqa: E402

RESULTS_DIR = Path("evals/results")


# ---------------------------------------------------------------------------
# Environment recording
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Whether the *code* the run will execute is reproducible from git_sha.

    evals/results/ is excluded from this check: this run's own output
    directory is inherently untracked at the moment this check runs (it
    hasn't been written yet), so including it would make git_dirty report
    True for every eval run, forever, regardless of whether the actual
    source is clean -- a flag that's always true carries no signal.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":!evals/results"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        # Unknown git state is treated as dirty -- a baseline whose
        # reproducibility can't be confirmed shouldn't claim to be clean.
        return True


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _corpus_stats(settings: Settings) -> dict[str, Any]:
    manifest = load_manifest(settings.corpus_root)
    with session_scope(settings) as session:
        chunk_count = session.execute(select(func.count()).select_from(ChunkRecord)).scalar_one()
        index_build_timestamp = session.execute(
            select(func.max(ChunkRecord.indexed_at))
        ).scalar_one()
    return {
        "corpus_document_count": len(manifest),
        "corpus_chunk_count": chunk_count,
        "index_build_timestamp": index_build_timestamp.isoformat()
        if index_build_timestamp
        else None,
    }


def _build_config(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    effective_k = args.k if args.k is not None else settings.vector_retrieval_k
    config: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "tag": args.tag,
        "base_url": args.base_url,
        "concurrency": args.concurrency
        if args.concurrency is not None
        else settings.eval_concurrency,
        "slices": args.slices,
        "case_ids": args.case_ids,
        "chunk_strategy": f"fixed_{settings.chunk_size}_{settings.chunk_overlap}",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_strategy": "vector_only",
        "k": effective_k,
        "eval_retrieval_diagnostic_k": max(effective_k, settings.eval_retrieval_diagnostic_k),
        "embedding_model": settings.embedding_model,
        "generator_model": settings.generator_model,
        "temperature": settings.generation_temperature,
        "min_relevance_score": settings.min_relevance_score,
        "prompt_version": PROMPT_VERSION,
        "python_version": platform.python_version(),
        "package_versions": {
            "openai": _package_version("openai"),
            "pgvector": _package_version("pgvector"),
            "sqlalchemy": _package_version("sqlalchemy"),
            "tiktoken": _package_version("tiktoken"),
        },
    }
    config.update(_corpus_stats(settings))
    return config


def _run_dir_path(config: dict[str, Any]) -> Path:
    short_sha = config["git_sha"][:8] if config["git_sha"] != "unknown" else "nogit"
    return RESULTS_DIR / f"{config['timestamp']}_{short_sha}"


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = _build_config(args, settings)

    print("Config for this run:")
    print(json.dumps(config, indent=2, default=str))
    print()

    if args.no_write:
        results = harness.run_eval(
            base_url=args.base_url,
            settings=settings,
            slices=args.slices,
            case_ids=args.case_ids,
            k=args.k,
            concurrency=args.concurrency,
        )
        case_inputs = [harness.to_metric_input(result) for result in results]
        metrics = compute_all_metrics(
            case_inputs,
            relevance_threshold=settings.min_relevance_score,
            implemented_decisions=set(settings.eval_implemented_decisions),
        )
        print(report.format_headline_table(metrics))
        return 0

    run_dir = _run_dir_path(config)
    results = harness.run_and_write_raw(
        base_url=args.base_url,
        settings=settings,
        run_dir=run_dir,
        slices=args.slices,
        case_ids=args.case_ids,
        k=args.k,
        concurrency=args.concurrency,
    )

    unrecovered_5xx = [
        result
        for result in results
        if result.failed and result.primary.error and "5xx after" in result.primary.error
    ]
    if unrecovered_5xx:
        print(f"STOPPING: {len(unrecovered_5xx)} case(s) failed with a 5xx after their one retry:")
        for result in unrecovered_5xx:
            print(f"  {result.case_id}: {result.primary.error}")
        print(
            f"\nRaw results were still written to {run_dir / 'raw.jsonl'} for inspection, "
            "but metrics.json/report.md/config.json were NOT written and baseline.json was "
            "NOT updated -- a baseline with holes in it is not usable as a comparison point."
        )
        return 1

    case_inputs = [harness.to_metric_input(result) for result in results]
    metrics = compute_all_metrics(
        case_inputs,
        relevance_threshold=settings.min_relevance_score,
        implemented_decisions=set(settings.eval_implemented_decisions),
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        report.render_report_md(
            metrics=metrics, case_inputs=case_inputs, results=results, config=config
        ),
        encoding="utf-8",
    )

    if args.tag == "baseline":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pointer = {
            "tag": args.tag,
            "run_dir": run_dir.as_posix(),
            "timestamp": config["timestamp"],
            "git_sha": config["git_sha"],
        }
        (RESULTS_DIR / "baseline.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    print(report.format_headline_table(metrics))
    print()
    print(f"Run directory: {run_dir}")
    return 0


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def _run_compare(dir_a: Path, dir_b: Path) -> int:
    metrics_a = json.loads((dir_a / "metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads((dir_b / "metrics.json").read_text(encoding="utf-8"))

    print(report.format_compare_table(metrics_a, metrics_b, label_a=dir_a.name, label_b=dir_b.name))
    print()

    raw_a = harness.read_raw_jsonl(dir_a / "raw.jsonl")
    raw_b = harness.read_raw_jsonl(dir_b / "raw.jsonl")
    expected_by_case = {row["case_id"]: row["expected_decision"] for row in raw_a}
    expected_by_case.update({row["case_id"]: row["expected_decision"] for row in raw_b})
    actual_a = {row["case_id"]: row["actual_decision"] for row in raw_a}
    actual_b = {row["case_id"]: row["actual_decision"] for row in raw_b}

    print("Cases that changed verdict:")
    print(report.format_verdict_changes(expected_by_case, actual_a, actual_b))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slice", action="append", dest="slices", default=None, help="Repeatable.")
    parser.add_argument(
        "--case-id", action="append", dest="case_ids", default=None, help="Repeatable."
    )
    parser.add_argument("--k", type=int, default=None, help="Override the primary call's k.")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--tag", default=None, help='"baseline" also updates evals/results/baseline.json.'
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_DIR_A", "RUN_DIR_B"),
        default=None,
        help="Compare two existing run directories instead of running anything.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Run and print, but write nothing to disk."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.compare:
        return _run_compare(Path(args.compare[0]), Path(args.compare[1]))
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
