#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.io_utils import dump_jsonl, load_json, load_jsonl
from benchmarks.common.judge_utils import metrics_with_custom_denominator
from benchmarks.ancient_char_exegesis.metrics import aggregate_numeric_rows, build_component_metrics


def enrich_component_metrics(row: dict[str, Any]) -> bool:
    agg_keys = [
        "__agg_component_gt_total",
        "__agg_component_pred_total",
        "__agg_component_match_total",
        "__agg_component_function_correct",
        "__agg_component_evolution_correct",
        "__agg_component_explanation_bertscore_sum",
    ]
    if not any(key in row for key in agg_keys):
        return False

    gold_total = float(row.get("__agg_component_gt_total", 0.0) or 0.0)
    pred_total = float(row.get("__agg_component_pred_total", 0.0) or 0.0)
    match_total = float(row.get("__agg_component_match_total", 0.0) or 0.0)
    function_correct = float(row.get("__agg_component_function_correct", 0.0) or 0.0)
    evolution_correct = float(row.get("__agg_component_evolution_correct", 0.0) or 0.0)
    explanation_sum = float(row.get("__agg_component_explanation_bertscore_sum", 0.0) or 0.0)

    updates = build_component_metrics(
        gold_total=gold_total,
        pred_total=pred_total,
        match_total=match_total,
        function_correct=function_correct,
        evolution_correct=evolution_correct,
        explanation_sum=explanation_sum,
    )

    changed = False
    for key, value in updates.items():
        old = row.get(key)
        old_val = float(old) if isinstance(old, (int, float)) else None
        if old_val is None or abs(old_val - value) > 1e-12:
            row[key] = value
            changed = True
    return changed


def enrich_component_metrics_rows(rows: list[dict[str, Any]]) -> bool:
    changed = False
    for row in rows:
        if enrich_component_metrics(row):
            changed = True
    return changed


def recompute_one_run(run_dir: Path) -> dict[str, Any] | None:
    metrics_dir = run_dir / "metrics"
    summary_path = metrics_dir / "summary.json"
    per_sample_path = metrics_dir / "per_sample.jsonl"
    if not summary_path.exists() or not per_sample_path.exists():
        return None

    summary = load_json(summary_path)
    metric_rows = load_jsonl(per_sample_path)
    failures_bundle = load_json(run_dir / "failures.json")
    failures = failures_bundle.get("failures") or []
    if not isinstance(failures, list):
        failures = []

    rows_changed = enrich_component_metrics_rows(metric_rows)
    if rows_changed:
        dump_jsonl(per_sample_path, metric_rows)

    success_count = len(metric_rows)
    failure_count = len(failures)
    considered_count = success_count + failure_count
    success_only_metrics = aggregate_numeric_rows(metric_rows)
    failure_denominator_metrics = metrics_with_custom_denominator(
        success_only_metrics,
        success_count=success_count,
        denominator_count=considered_count,
    )

    sample_count = int(summary.get("sample_count") or considered_count)
    summary["sample_count"] = sample_count
    summary["success_count"] = success_count
    summary["failure_count"] = failure_count
    summary["considered_count"] = considered_count
    summary["metrics"] = success_only_metrics
    summary["metrics_success_only"] = success_only_metrics
    summary["metrics_with_failures_in_denominator"] = failure_denominator_metrics

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "sample_count": sample_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "considered_count": considered_count,
        "metrics_success_only": success_only_metrics,
        "metrics_with_failures_in_denominator": failure_denominator_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute JieZi-bench benchmark summary metrics with failures included in denominator."
    )
    parser.add_argument(
        "--runs-root",
        default="runs/ancient_char_exegesis",
        help="Run root directory containing per-run subfolders.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help=(
            "Optional output JSON path for merged report. "
            "Defaults to <runs-root>/failure_denominator_metrics_report.json"
        ),
    )
    args = parser.parse_args()

    runs_root = (REPO_ROOT / args.runs_root).resolve()
    if not runs_root.exists() or not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    results: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        result = recompute_one_run(run_dir)
        if result is not None:
            results.append(result)
        else:
            skipped_runs.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "reason": "missing metrics/summary.json or metrics/per_sample.jsonl",
                }
            )

    report_path = Path(args.report_path).resolve() if args.report_path else (runs_root / "failure_denominator_metrics_report.json")
    report = {
        "runs_root": str(runs_root),
        "run_count": len(results),
        "skipped_count": len(skipped_runs),
        "skipped_runs": skipped_runs,
        "runs": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"updated_runs={len(results)}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
