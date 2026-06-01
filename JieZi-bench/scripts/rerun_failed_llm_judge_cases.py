#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.io_utils import dump_json, dump_jsonl, load_json, load_jsonl
from benchmarks.common.judge_utils import (
    ensure_canonical_llm_judge_layout,
    find_latest_subdir,
    inject_llm_judge_metrics,
    summarize_judge_rows,
)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def copy_retry_cases(sample_ids: list[str], source_dir: Path, retry_dir: Path) -> int:
    retry_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sample_id in sample_ids:
        source_path = source_dir / f"{sample_id}.json"
        if not source_path.exists():
            continue
        shutil.copy2(source_path, retry_dir / source_path.name)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run only failed llm-judge cases and merge the refreshed scores back into summary metrics")
    parser.add_argument("--runs-root", default="runs/ancient_char_exegesis")
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--model-config", default="llm_judge/model_config.json")
    parser.add_argument("--dimension-prompts-dir", default="llm_judge/prompts/dreambenchpp_v1")
    parser.add_argument("--dimensions", default="fact_alignment,scholarly_expression")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cases-root", default="llm_judge/_batch_cases")
    parser.add_argument("--results-root", default="llm_judge/results/by_run")
    parser.add_argument("--retry-cases-root", default="llm_judge/_retry_cases")
    parser.add_argument("--retry-results-root", default="llm_judge/results/retry_failed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ensure_canonical_llm_judge_layout(REPO_ROOT)

    runs_root = (REPO_ROOT / args.runs_root).resolve()
    result_root = (REPO_ROOT / args.results_root).resolve()
    run_dirs = sorted([p for p in runs_root.glob(args.run_glob) if p.is_dir()])
    if not run_dirs:
        raise RuntimeError("no matching run directories")

    for run_dir in run_dirs:
        latest_result_dir = find_latest_subdir(result_root / run_dir.name)
        latest_summary = load_json(latest_result_dir / "summary.json")
        latest_rows = load_jsonl(latest_result_dir / "results.jsonl")
        error_rows = [row for row in latest_rows if row.get("status") == "error"]
        if not error_rows:
            print(f"[skip] {run_dir.name} has no failed judge cases", flush=True)
            continue

        error_ids = [str(row.get("sample_id")) for row in error_rows if row.get("sample_id")]
        retry_case_dir = (REPO_ROOT / args.retry_cases_root / run_dir.name).resolve()
        if retry_case_dir.exists() and not args.dry_run:
            shutil.rmtree(retry_case_dir)
        copied = copy_retry_cases(error_ids, (REPO_ROOT / args.cases_root / run_dir.name).resolve(), retry_case_dir)
        print(f"[run] {run_dir.name} retry_cases={copied}", flush=True)
        if copied == 0:
            print(f"[skip] {run_dir.name} no retry cases copied", flush=True)
            continue

        retry_result_parent = Path(args.retry_results_root) / run_dir.name
        cmd = [
            sys.executable,
            "llm_judge/run_llm_judge_test.py",
            "--input-dir",
            str(retry_case_dir.relative_to(REPO_ROOT)),
            "--model-config",
            args.model_config,
            "--dimension-prompts-dir",
            args.dimension_prompts_dir,
            "--dimensions",
            args.dimensions,
            "--output-dir",
            str(retry_result_parent),
            "--concurrency",
            str(args.concurrency),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
        ]
        run_cmd(cmd, dry_run=args.dry_run)
        if args.dry_run:
            continue

        retry_result_dir = find_latest_subdir((REPO_ROOT / retry_result_parent).resolve())
        retry_summary = load_json(retry_result_dir / "summary.json")
        retry_rows = load_jsonl(retry_result_dir / "results.jsonl")
        replacement_map = {str(row.get("sample_id")): row for row in retry_rows if row.get("sample_id")}

        merged_rows: list[dict[str, Any]] = []
        for row in latest_rows:
            sample_id = str(row.get("sample_id") or "")
            if sample_id in replacement_map:
                merged_rows.append(replacement_map[sample_id])
            else:
                merged_rows.append(row)

        dimensions = latest_summary.get("dimensions") if isinstance(latest_summary.get("dimensions"), list) else []
        merged_stats = summarize_judge_rows(merged_rows, dimensions)
        merged_summary = dict(latest_summary)
        merged_summary.update(merged_stats)
        merged_summary["retried_error_cases"] = len(error_ids)
        merged_summary["dimensions"] = dimensions
        merged_summary["model_config"] = latest_summary.get("model_config", {})

        dump_jsonl(retry_result_dir / "merged_results.jsonl", sorted(merged_rows, key=lambda x: x.get("sample_id", "")))
        dump_json(retry_result_dir / "merged_summary.json", merged_summary)

        run_summary_path = run_dir / "metrics" / "summary.json"
        run_summary = load_json(run_summary_path)
        merged_summary["retried_error_cases"] = len(error_ids)
        updated = inject_llm_judge_metrics(run_summary, merged_summary, retry_result_dir, repo_root=REPO_ROOT)
        dump_json(run_summary_path, updated)
        print(
            f"[updated] {run_dir.name} errors {len(error_rows)} -> {merged_stats['error_cases']} using {retry_result_dir.relative_to(REPO_ROOT)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
