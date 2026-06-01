"""Shared utilities for LLM-judge orchestration across run_eval and scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_canonical_llm_judge_layout(repo_root: Path) -> None:
    canonical_dir = repo_root / "llm_judge"
    legacy_dir = repo_root / "test-llm-judge"
    if not canonical_dir.exists():
        raise FileNotFoundError(f"llm_judge directory not found: {canonical_dir}")
    if legacy_dir.exists():
        raise RuntimeError(
            f"legacy llm-judge directory detected: {legacy_dir}. "
            "Only llm_judge/ is supported. Remove test-llm-judge/ before running."
        )


def find_latest_subdir(parent_dir: Path) -> Path:
    candidates = sorted(path for path in parent_dir.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no result dir under {parent_dir}")
    return candidates[-1]


def parse_numeric_scores(raw_scores: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, value in raw_scores.items():
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if scores and "overall_score" not in scores:
        dims = [v for k, v in scores.items() if k != "overall_score"]
        if dims:
            scores["overall_score"] = sum(dims) / len(dims)
    return scores


def metrics_with_custom_denominator(
    success_only_metrics: dict[str, float],
    *,
    success_count: int,
    denominator_count: int,
) -> dict[str, float]:
    if denominator_count <= 0 or success_count <= 0 or not success_only_metrics:
        return {key: 0.0 for key in sorted(success_only_metrics)}
    scale = success_count / denominator_count
    return {key: float(value) * scale for key, value in sorted(success_only_metrics.items())}


def inject_llm_judge_metrics(
    summary: dict[str, Any],
    judge_result: dict[str, Any],
    result_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    sample_count = int(summary.get("sample_count") or summary.get("considered_count") or 0)
    success_count = int(judge_result.get("success_count") or 0)
    means = judge_result.get("mean_scores") if isinstance(judge_result.get("mean_scores"), dict) else {}
    dimensions = judge_result.get("dimensions") if isinstance(judge_result.get("dimensions"), list) else []

    display_dir = str(result_dir.relative_to(repo_root)) if repo_root else str(result_dir)
    retry_count = judge_result.get("retried_error_cases", 0)

    summary["llm_judge"] = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dimensions": dimensions,
        "success_count": success_count,
        "mean_scores": means,
        "result_dir": display_dir,
        "judge_model": judge_result.get("model_config", {}).get("model"),
    }
    if retry_count:
        summary["llm_judge"]["retried_error_cases"] = retry_count

    for field in ("metrics", "metrics_success_only", "metrics_with_failures_in_denominator"):
        if not isinstance(summary.get(field), dict):
            summary[field] = {}

    scale = (success_count / sample_count) if sample_count > 0 else 0.0

    for name, value in means.items():
        metric_key = f"evolution_judge_{name}"
        score = float(value)
        summary["metrics"][metric_key] = score
        summary["metrics_success_only"][metric_key] = score
        summary["metrics_with_failures_in_denominator"][metric_key] = score * scale

    return summary


def summarize_judge_rows(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok" and isinstance(row.get("scores"), dict)]
    means: dict[str, float] = {}
    if ok_rows:
        for field in [*dimensions, "overall_score"]:
            values = [float(row["scores"].get(field, 0.0)) for row in ok_rows]
            means[field] = sum(values) / len(values)
    return {
        "total_cases": len(rows),
        "ok_cases": len(ok_rows),
        "skipped_cases": sum(1 for row in rows if row.get("status") == "skip"),
        "error_cases": sum(1 for row in rows if row.get("status") == "error"),
        "success_count": len(ok_rows),
        "mean_scores": means,
    }
