#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.io_utils import dump_json, dump_jsonl, load_json, load_jsonl
from benchmarks.common.judge_utils import (
    ensure_canonical_llm_judge_layout,
    find_latest_subdir,
    parse_numeric_scores,
)
from benchmarks.ancient_char_exegesis.judges import normalize_judge_scores

TARGET_DIMENSIONS = [
    "fact_alignment",
    "scholarly_expression",
]
JUDGE_KEYS = [*(f"evolution_judge_{key}" for key in TARGET_DIMENSIONS), "evolution_judge_overall_score"]
LEGACY_JUDGE_KEYS = [
    "evolution_judge_fact_accuracy",
    "evolution_judge_information_completeness",
    "evolution_judge_no_unsupported_content",
    "evolution_judge_clarity_coherence",
    "evolution_judge_format_compliance",
    "evolution_judge_key_point_coverage",
    "evolution_judge_diachronic_logic",
]


def load_config(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(content)
    else:
        payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError(f"config is not an object: {path}")
    return payload


def _judge_row_from_scores(scores: dict[str, float]) -> dict[str, float]:
    row: dict[str, float] = {}
    for key in TARGET_DIMENSIONS:
        row[f"evolution_judge_{key}"] = float(scores.get(key, 0.0))
    row["evolution_judge_overall_score"] = float(scores.get("overall_score", 0.0))
    return row


def _parse_judge_output_bundle(bundle: dict[str, Any]) -> dict[str, float] | None:
    parsed = bundle.get("parsed")
    if not isinstance(parsed, dict):
        return None
    if any(
        legacy_key in parsed
        for legacy_key in ("fact_accuracy", "information_completeness", "clarity_coherence")
    ):
        normalized = normalize_judge_scores(parsed)
        scores = {
            "fact_alignment": float(normalized.get("fact_alignment", 0.0)),
            "overall_score": float(normalized.get("overall_score", 0.0)),
        }
        return scores

    scores: dict[str, float] = {}
    for key, value in parsed.items():
        score_val: Any
        if isinstance(value, dict):
            score_val = value.get("score")
        else:
            score_val = value
        try:
            scores[str(key)] = float(score_val)
        except (TypeError, ValueError):
            continue
    return parse_numeric_scores(scores) if scores else None


def has_all_judge_keys(row: dict[str, Any]) -> bool:
    return all(key in row for key in JUDGE_KEYS)


def _collect_judge_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {key: [] for key in JUDGE_KEYS}
    for row in rows:
        for key in JUDGE_KEYS:
            val = row.get(key)
            if isinstance(val, (int, float)):
                values[key].append(float(val))
    means: dict[str, float] = {}
    for key in JUDGE_KEYS:
        arr = values[key]
        means[key] = float(sum(arr) / len(arr)) if arr else 0.0
    return means


def _collect_grouped_judge_means(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {key: [] for key in JUDGE_KEYS})
    for row in rows:
        group = str(row.get(field) or "").strip()
        if not group:
            continue
        bucket = grouped[group]
        for key in JUDGE_KEYS:
            val = row.get(key)
            if isinstance(val, (int, float)):
                bucket[key].append(float(val))
    result: dict[str, dict[str, float]] = {}
    for group, values in grouped.items():
        result[group] = {}
        for key in JUDGE_KEYS:
            arr = values[key]
            result[group][key] = float(sum(arr) / len(arr)) if arr else 0.0
    return result


def sync_judge_aggregates(run_dir: Path, metrics_rows: list[dict[str, Any]]) -> None:
    metrics_dir = run_dir / "metrics"
    summary_path = metrics_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        overall = _collect_judge_means(metrics_rows)
        for section_key in ("metrics", "metrics_success_only", "metrics_with_failures_in_denominator"):
            section = summary.get(section_key)
            if isinstance(section, dict):
                section.update(overall)
        dump_json(summary_path, summary)

    grouped_files = [
        ("by_group.json", "group"),
        ("by_glyph.json", "glyph_category"),
        ("by_prompt.json", "prompt_name"),
    ]
    for filename, field in grouped_files:
        path = metrics_dir / filename
        if not path.exists():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        grouped_means = _collect_grouped_judge_means(metrics_rows, field)
        for bucket, score_map in grouped_means.items():
            existing = payload.get(bucket)
            if not isinstance(existing, dict):
                existing = {}
                payload[bucket] = existing
            existing.update(score_map)
        dump_json(path, payload)


def run_llm_judge_for_samples(
    *,
    run_dir: Path,
    sample_ids: list[str],
    judge_cfg: dict[str, Any],
    judge_model_cfg: dict[str, Any],
    concurrency: int,
    force_rerun: bool,
) -> tuple[dict[str, dict[str, float]], int]:
    llm_judge_cfg = dict(judge_cfg.get("llm_judge") or {})
    dimensions = str(llm_judge_cfg.get("dimensions", "fact_alignment,scholarly_expression"))
    dimension_prompts_dir = str(llm_judge_cfg.get("dimension_prompts_dir", "llm_judge/prompts/dreambenchpp_v1"))
    cases_root = Path(str(llm_judge_cfg.get("cases_root", "llm_judge/_batch_cases")))
    retry_cases_root = Path(str(llm_judge_cfg.get("retry_cases_root", "llm_judge/_retry_cases")))
    results_root = Path(str(llm_judge_cfg.get("results_root", "llm_judge/results/by_run")))
    max_tokens = int(llm_judge_cfg.get("max_tokens", judge_cfg.get("max_tokens", 1000)))
    temperature = float(llm_judge_cfg.get("temperature", 0.0))
    effective_concurrency = max(1, int(llm_judge_cfg.get("concurrency", concurrency)))

    run_rel = run_dir.relative_to(REPO_ROOT)
    run_name = run_dir.name
    case_dir_rel = cases_root / run_name
    retry_case_dir_rel = retry_cases_root / run_name
    result_parent_rel = results_root / run_name
    case_dir = (REPO_ROOT / case_dir_rel).resolve()
    retry_case_dir = (REPO_ROOT / retry_case_dir_rel).resolve()
    result_parent_dir = (REPO_ROOT / result_parent_rel).resolve()
    judge_output_dir = run_dir / "judge_outputs"
    judge_output_dir.mkdir(parents=True, exist_ok=True)

    prepare_cmd = [
        sys.executable,
        "llm_judge/prepare_gold_test.py",
        "--run-dir",
        str(run_rel),
        "--output-dir",
        str(case_dir_rel),
        "--overwrite",
    ]
    completed = subprocess.run(prepare_cmd, cwd=REPO_ROOT, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_gold_test failed with returncode={completed.returncode}")

    score_map: dict[str, dict[str, float]] = {}
    pending_ids: list[str] = []
    for sample_id in sample_ids:
        judge_path = judge_output_dir / f"{sample_id}.json"
        if not force_rerun and judge_path.exists():
            try:
                cached = _parse_judge_output_bundle(load_json(judge_path))
            except Exception:
                cached = None
            if cached and "scholarly_expression" in cached:
                score_map[sample_id] = _judge_row_from_scores(cached)
                continue
        pending_ids.append(sample_id)

    if not pending_ids:
        return score_map, 0

    if retry_case_dir.exists():
        shutil.rmtree(retry_case_dir)
    retry_case_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for sample_id in pending_ids:
        src = case_dir / f"{sample_id}.json"
        if not src.exists():
            continue
        shutil.copy2(src, retry_case_dir / src.name)
        copied += 1
    if copied == 0:
        return score_map, len(pending_ids)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as tmp_file:
        json.dump(judge_model_cfg, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.write("\n")
        model_config_path = Path(tmp_file.name).resolve()

    try:
        judge_cmd = [
            sys.executable,
            "llm_judge/run_llm_judge_test.py",
            "--input-dir",
            str(retry_case_dir_rel),
            "--model-config",
            str(model_config_path),
            "--dimension-prompts-dir",
            dimension_prompts_dir,
            "--dimensions",
            dimensions,
            "--output-dir",
            str(result_parent_rel),
            "--concurrency",
            str(effective_concurrency),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
        ]
        completed = subprocess.run(judge_cmd, cwd=REPO_ROOT, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"run_llm_judge_test failed with returncode={completed.returncode}")
    finally:
        model_config_path.unlink(missing_ok=True)

    result_dir = find_latest_subdir(result_parent_dir)
    result_rows = load_jsonl(result_dir / "results.jsonl")
    per_sample_dir = result_dir / "per_sample"

    failed = 0
    for row in result_rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in pending_ids:
            continue
        status = str(row.get("status") or "")
        if status != "ok" or not isinstance(row.get("scores"), dict):
            failed += 1
            continue
        scores = parse_numeric_scores(row["scores"])
        if not scores:
            failed += 1
            continue

        reasons: dict[str, str] = {}
        per_sample_path = per_sample_dir / f"{sample_id}.json"
        if per_sample_path.exists():
            per_sample = load_json(per_sample_path)
            if isinstance(per_sample.get("reasons"), dict):
                reasons = {str(k): str(v) for k, v in per_sample["reasons"].items()}

        parsed_payload: dict[str, Any] = {}
        for key, value in scores.items():
            if key == "overall_score":
                parsed_payload[key] = float(value)
            else:
                parsed_payload[key] = {"score": float(value), "reason": reasons.get(key, "")}
        dump_json(
            judge_output_dir / f"{sample_id}.json",
            {
                "source": "llm_judge",
                "result_dir": str(result_dir.relative_to(REPO_ROOT)),
                "parsed": parsed_payload,
            },
        )
        score_map[sample_id] = _judge_row_from_scores(scores)

    failed += sum(1 for sample_id in pending_ids if sample_id not in score_map)
    return score_map, failed


def backfill_run(
    run_dir: Path,
    concurrency: int,
    judge_api_key: str = "",
    force_rerun: bool = False,
    judge_model_override: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    manifest = load_json(run_dir / "manifest.json")
    run_config = manifest.get("run_config") or {}
    judge_cfg = run_config.get("judge") or {}
    if judge_model_override is not None:
        judge_model_cfg = dict(judge_model_override)
    else:
        judge_model_cfg = judge_cfg.get("model")
        if not isinstance(judge_model_cfg, dict):
            raise ValueError(f"judge.model missing in {run_dir}")

    if judge_api_key:
        judge_model_cfg = dict(judge_model_cfg)
        judge_model_cfg["api_key"] = judge_api_key

    metrics_path = run_dir / "metrics" / "per_sample.jsonl"
    metrics_rows = load_jsonl(metrics_path)
    pending_rows: list[dict[str, Any]] = [row for row in metrics_rows if not has_all_judge_keys(row)]
    total_pending = len(pending_rows)
    if total_pending == 0:
        sync_judge_aggregates(run_dir, metrics_rows)
        return (0, 0, 0)

    pending_sample_ids = sorted(
        {
            str(row.get("sample_id") or "").strip()
            for row in pending_rows
            if str(row.get("sample_id") or "").strip()
        }
    )
    if not pending_sample_ids:
        return (total_pending, 0, total_pending)

    score_map, failed_calls = run_llm_judge_for_samples(
        run_dir=run_dir,
        sample_ids=pending_sample_ids,
        judge_cfg=judge_cfg,
        judge_model_cfg=judge_model_cfg,
        concurrency=concurrency,
        force_rerun=force_rerun,
    )

    updated = 0
    for row in metrics_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id not in score_map:
            continue
        for key in list(row.keys()):
            if key.startswith("evolution_judge_"):
                row.pop(key, None)
        for key in LEGACY_JUDGE_KEYS:
            row.pop(key, None)
        row.update(score_map[sample_id])
        updated += 1

    dump_jsonl(metrics_path, metrics_rows)
    sync_judge_aggregates(run_dir, metrics_rows)
    return (total_pending, updated, failed_calls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing LLM-judge scores into metrics/per_sample.jsonl for run dirs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directory names under runs/ancient_char_exegesis")
    parser.add_argument("--runs-root", default="runs/ancient_char_exegesis")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--judge-api-key", default="", help="Override judge API key for this backfill run")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-call judge API even if judge_outputs already exist.")
    parser.add_argument(
        "--judge-model-config",
        default="",
        help="Optional JSON/YAML model config path to override judge.model in run manifest.",
    )
    args = parser.parse_args()
    ensure_canonical_llm_judge_layout(REPO_ROOT)

    judge_model_override = None
    if args.judge_model_config:
        judge_model_override = load_config((REPO_ROOT / args.judge_model_config).resolve())

    runs_root = (REPO_ROOT / args.runs_root).resolve()
    for run_name in args.runs:
        run_dir = runs_root / run_name
        if not run_dir.exists():
            print(f"[skip] run not found: {run_name}", flush=True)
            continue
        print(f"[start] {run_name}", flush=True)
        pending, updated, failed = backfill_run(
            run_dir,
            concurrency=args.concurrency,
            judge_api_key=str(args.judge_api_key or ""),
            force_rerun=bool(args.force_rerun),
            judge_model_override=judge_model_override,
        )
        print(
            f"[done] {run_name} pending={pending} updated={updated} failed_calls={failed}",
            flush=True,
        )


if __name__ == "__main__":
    main()
