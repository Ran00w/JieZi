#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.io_utils import dump_json, load_json
from benchmarks.common.judge_utils import (
    ensure_canonical_llm_judge_layout,
    find_latest_subdir,
    inject_llm_judge_metrics,
)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def discover_run_dirs(runs_root: Path, run_glob: str) -> list[Path]:
    dirs = sorted([p for p in runs_root.glob(run_glob) if p.is_dir()])
    selected: list[Path] = []
    for run_dir in dirs:
        if not (run_dir / "predictions.jsonl").exists():
            continue
        if not (run_dir / "metrics" / "summary.json").exists():
            continue
        selected.append(run_dir)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run llm-judge for ancient_char_exegesis runs and write back metrics/summary.json")
    parser.add_argument("--runs-root", default="runs/ancient_char_exegesis", help="Root dir containing run folders")
    parser.add_argument("--run-glob", default="*", help="Glob pattern for selecting run folders under --runs-root")
    parser.add_argument("--limit-runs", type=int, default=0, help="Optional max run folders to process (0 means all)")
    parser.add_argument("--model-config", default="llm_judge/model_config.json")
    parser.add_argument("--dimension-prompts-dir", default="llm_judge/prompts/dreambenchpp_v1")
    parser.add_argument("--dimensions", default="fact_alignment,scholarly_expression")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge-limit", type=int, default=0, help="Optional max samples per run for judge (0 means all)")
    parser.add_argument("--cases-root", default="llm_judge/_batch_cases")
    parser.add_argument("--results-root", default="llm_judge/results/by_run")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands and targets")
    args = parser.parse_args()
    ensure_canonical_llm_judge_layout(REPO_ROOT)

    runs_root = (REPO_ROOT / args.runs_root).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    run_dirs = discover_run_dirs(runs_root, args.run_glob)
    if args.limit_runs > 0:
        run_dirs = run_dirs[: args.limit_runs]
    if not run_dirs:
        raise RuntimeError("no run directories matched the selection criteria")

    print(f"[info] selected runs: {len(run_dirs)}", flush=True)
    for run_dir in run_dirs:
        run_rel = run_dir.relative_to(REPO_ROOT)
        run_name = run_dir.name
        case_output_rel = Path(args.cases_root) / run_name
        result_parent_rel = Path(args.results_root) / run_name
        summary_path = run_dir / "metrics" / "summary.json"

        print(f"[run] {run_rel}", flush=True)
        run_cmd(
            [
                sys.executable,
                "llm_judge/prepare_gold_test.py",
                "--run-dir",
                str(run_rel),
                "--output-dir",
                str(case_output_rel),
                "--overwrite",
            ],
            dry_run=args.dry_run,
        )
        judge_cmd = [
            sys.executable,
            "llm_judge/run_llm_judge_test.py",
            "--input-dir",
            str(case_output_rel),
            "--model-config",
            args.model_config,
            "--dimension-prompts-dir",
            args.dimension_prompts_dir,
            "--dimensions",
            args.dimensions,
            "--output-dir",
            str(result_parent_rel),
            "--concurrency",
            str(args.concurrency),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
        ]
        if args.judge_limit > 0:
            judge_cmd.extend(["--limit", str(args.judge_limit)])
        run_cmd(judge_cmd, dry_run=args.dry_run)

        if args.dry_run:
            print(f"[dry-run] would update {summary_path.relative_to(REPO_ROOT)}", flush=True)
            continue

        result_dir = find_latest_subdir((REPO_ROOT / result_parent_rel).resolve())
        judge_summary = load_json(result_dir / "summary.json")
        summary = load_json(summary_path)
        updated = inject_llm_judge_metrics(summary, judge_summary, result_dir, repo_root=REPO_ROOT)
        dump_json(summary_path, updated)
        print(f"[updated] {summary_path.relative_to(REPO_ROOT)}", flush=True)

    print("[done] rerun and summary update completed", flush=True)


if __name__ == "__main__":
    main()
