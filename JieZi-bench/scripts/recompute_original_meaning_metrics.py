#!/usr/bin/env python3
"""Recompute original-meaning metrics for all existing run directories.

Currently only BERTScore is supported; all other text-similarity recomputers
(edit distance, LCS, ROUGE, BLEU) have been removed.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.io_utils import dump_json, dump_jsonl, load_json, load_jsonl, safe_mean
from benchmarks.ancient_char_exegesis.metrics import BertScorer


def infer_original_meaning(bundle: dict[str, Any], key: str) -> str:
    text = ""
    container = bundle.get(key)
    if isinstance(container, dict):
        text = str(container.get("本义", "") or "")
    return text.strip()


def update_summary_metric(summary: dict[str, Any], metric_key: str, metric_value: float) -> None:
    success_count = int(summary.get("success_count") or 0)
    failure_count = int(summary.get("failure_count") or 0)
    considered_count = int(summary.get("considered_count") or (success_count + failure_count))
    scale = (success_count / considered_count) if considered_count > 0 else 0.0

    for section in ("metrics", "metrics_success_only"):
        payload = summary.get(section)
        if not isinstance(payload, dict):
            payload = {}
            summary[section] = payload
        payload[metric_key] = metric_value

    payload = summary.get("metrics_with_failures_in_denominator")
    if not isinstance(payload, dict):
        payload = {}
        summary["metrics_with_failures_in_denominator"] = payload
    payload[metric_key] = metric_value * scale


def update_bucket_metric(
    bucket_path: Path,
    metric_rows: list[dict[str, Any]],
    metric_key: str,
    bucket_field: str,
) -> None:
    if not bucket_path.exists():
        return
    payload = load_json(bucket_path)
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in metric_rows:
        key = row.get(bucket_field)
        value = row.get(metric_key)
        if key is None or not isinstance(value, (int, float)):
            continue
        buckets[str(key)].append(float(value))

    if not isinstance(payload, dict):
        return

    for bucket_name, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        values = buckets.get(str(bucket_name), [])
        if values:
            metrics[metric_key] = safe_mean(values)
        elif metric_key in metrics:
            del metrics[metric_key]
    dump_json(bucket_path, payload)


def iter_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def process_run(
    run_dir: Path,
    *,
    scorer: BertScorer,
    metric_key: str,
    batch_size: int,
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    per_sample_path = run_dir / "metrics" / "per_sample.jsonl"
    summary_path = run_dir / "metrics" / "summary.json"
    if not predictions_path.exists() or not per_sample_path.exists() or not summary_path.exists():
        return {
            "run": run_dir.name,
            "status": "skipped",
            "reason": "missing predictions.jsonl or metrics/per_sample.jsonl or metrics/summary.json",
        }

    prediction_rows = load_jsonl(predictions_path)
    metric_rows = load_jsonl(per_sample_path)
    if not prediction_rows or not metric_rows:
        return {
            "run": run_dir.name,
            "status": "skipped",
            "reason": "empty predictions or per_sample",
        }

    prediction_index: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            prediction_index[sample_id] = row

    pending_indices: list[int] = []
    batch_predictions: list[str] = []
    batch_references: list[str] = []
    missing_prediction_count = 0
    empty_both_count = 0
    empty_one_side_count = 0

    for idx, row in enumerate(metric_rows):
        sample_id = str(row.get("sample_id", "")).strip()
        pred_bundle = prediction_index.get(sample_id)
        if pred_bundle is None:
            missing_prediction_count += 1
            row.pop(metric_key, None)
            continue

        pred_text = infer_original_meaning(pred_bundle, "prediction")
        ref_text = infer_original_meaning(pred_bundle, "ground_truth")

        if not pred_text and not ref_text:
            empty_both_count += 1
            row.pop(metric_key, None)
            continue
        if not pred_text or not ref_text:
            empty_one_side_count += 1
            row[metric_key] = 0.0
            continue

        pending_indices.append(idx)
        batch_predictions.append(pred_text)
        batch_references.append(ref_text)

    if pending_indices:
        scores = scorer.score_batch(batch_predictions, batch_references, batch_size=batch_size)
        for row_idx, score in zip(pending_indices, scores, strict=True):
            metric_rows[row_idx][metric_key] = score

    dump_jsonl(per_sample_path, metric_rows)

    summary = load_json(summary_path)
    metric_values = [float(row[metric_key]) for row in metric_rows if isinstance(row.get(metric_key), (int, float))]
    metric_mean = safe_mean(metric_values)
    update_summary_metric(summary, metric_key, metric_mean)
    dump_json(summary_path, summary)

    update_bucket_metric(run_dir / "metrics" / "by_group.json", metric_rows, metric_key, "group")
    update_bucket_metric(run_dir / "metrics" / "by_glyph.json", metric_rows, metric_key, "glyph_category")
    update_bucket_metric(run_dir / "metrics" / "by_prompt.json", metric_rows, metric_key, "prompt_name")

    meta_path = run_dir / "metrics" / f"{metric_key}.meta.json"
    dump_json(
        meta_path,
        {
            "metric_key": metric_key,
            "model_type": scorer.model_type,
            "num_layers": scorer.num_layers,
            "all_layers": scorer.all_layers,
            "layer_agg": scorer.layer_agg,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "scored_pair_count": len(pending_indices),
            "zero_score_due_to_one_side_empty_count": empty_one_side_count,
            "both_empty_omitted_count": empty_both_count,
            "missing_prediction_count": missing_prediction_count,
            "per_sample_count": len(metric_rows),
            "prediction_count": len(prediction_index),
        },
    )

    return {
        "run": run_dir.name,
        "status": "updated",
        "per_sample_count": len(metric_rows),
        "scored_pair_count": len(pending_indices),
        "zero_score_due_to_one_side_empty_count": empty_one_side_count,
        "both_empty_omitted_count": empty_both_count,
        "missing_prediction_count": missing_prediction_count,
        "metric_mean": metric_mean,
        "meta_path": str(meta_path.relative_to(REPO_ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute original meaning metrics for all existing run directories."
    )
    parser.add_argument("--runs-root", default="runs/ancient_char_exegesis")
    parser.add_argument("--model-path", required=True, help="Model path for BERTScore (local dir or ModelScope ID).")
    parser.add_argument(
        "--metric-key",
        default="original_meaning_bertscore",
        help="New per-sample metric field to write.",
    )
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--all-layers", action="store_true", help="Use all layers from bert_score scorer output.")
    parser.add_argument(
        "--layer-agg",
        default="mean",
        choices=["first", "last", "mean", "max", "min"],
        help="Aggregation strategy when --all-layers is enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runs", nargs="*", default=[], help="Optional run directory names under --runs-root.")
    parser.add_argument("--limit-runs", type=int, default=0, help="For smoke test, process first N runs only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = (REPO_ROOT / args.runs_root).resolve()

    scorer = BertScorer(
        model_type=str(Path(args.model_path).resolve()),
        lang=args.lang,
        num_layers=args.num_layers if args.num_layers > 0 else None,
        all_layers=bool(args.all_layers),
        layer_agg=args.layer_agg,
    )

    run_dirs = iter_run_dirs(runs_root)
    if args.runs:
        picked = set(args.runs)
        run_dirs = [path for path in run_dirs if path.name in picked]
    if args.limit_runs > 0:
        run_dirs = run_dirs[: args.limit_runs]
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {runs_root}")

    report: dict[str, Any] = {
        "runs_root": str(runs_root),
        "model_type": scorer.model_type,
        "metric_key": args.metric_key,
        "num_layers": scorer.num_layers,
        "batch_size": args.batch_size,
        "all_layers": scorer.all_layers,
        "layer_agg": scorer.layer_agg,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "results": [],
    }

    updated = 0
    skipped = 0
    for idx, run_dir in enumerate(run_dirs, start=1):
        result = process_run(
            run_dir,
            scorer=scorer,
            metric_key=args.metric_key,
            batch_size=args.batch_size,
        )
        report["results"].append(result)
        if result["status"] == "updated":
            updated += 1
            print(
                f"[{idx}/{len(run_dirs)}] updated {run_dir.name}: "
                f"mean={result['metric_mean']:.6f} pairs={result['scored_pair_count']}",
                flush=True,
            )
        else:
            skipped += 1
            print(f"[{idx}/{len(run_dirs)}] skipped {run_dir.name}: {result['reason']}", flush=True)

    report["updated_runs"] = updated
    report["skipped_runs"] = skipped
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = runs_root / f"{args.metric_key}_report_{stamp}.json"
    dump_json(report_path, report)
    print(f"[done] updated_runs={updated} skipped_runs={skipped}", flush=True)
    print(f"[report] {report_path}", flush=True)


if __name__ == "__main__":
    main()
