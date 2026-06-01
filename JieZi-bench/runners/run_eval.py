#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.ancient_char_exegesis.extractors import normalize_prediction
from benchmarks.ancient_char_exegesis.judges import normalize_judge_scores
from benchmarks.ancient_char_exegesis.metrics import (
    BertScorer,
    accuracy,
    aggregate_numeric_rows,
    bertscore_score_or_none_when_both_empty,
    build_component_metrics,
    glyph_accuracy,
    match_component_names,
    set_exact_match,
    set_recall_credit,
)
from benchmarks.ancient_char_exegesis.task import Sample, iter_samples
from benchmarks.common.io_utils import dump_json, dump_jsonl, load_jsonl
from benchmarks.common.openai_client import (
    OpenAICompatibleClient,
    extract_json_object,
    extract_message_text,
    redact_sensitive_config,
)
from benchmarks.common.judge_utils import (
    ensure_canonical_llm_judge_layout,
    find_latest_subdir,
    metrics_with_custom_denominator,
    parse_numeric_scores,
)
from benchmarks.common.rag import ImageToHeadwordRetriever
from benchmarks.common.runtime_env import setup_cache_environment
from benchmarks.common.vllm import start_vllm_server, wait_for_server


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def upsert_row_by_sample_id(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    sample_id = str(row.get("sample_id", ""))
    if sample_id:
        rows[:] = [item for item in rows if str(item.get("sample_id", "")) != sample_id]
    rows.append(row)


def load_failed_sample_ids(run_dir: Path) -> set[str]:
    failures_path = run_dir / "failures.json"
    if not failures_path.exists():
        raise FileNotFoundError(f"failures.json not found under {run_dir}")
    bundle = json.loads(failures_path.read_text(encoding="utf-8"))
    failures = bundle.get("failures") or []
    if not isinstance(failures, list):
        return set()
    sample_ids = {str(item.get("sample_id")) for item in failures if item.get("sample_id")}
    return {sample_id for sample_id in sample_ids if sample_id}


def summarize_retrieval_metrics(rows: list[dict[str, Any]], topks: list[int]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "metrics": {f"top{k}_acc": 0.0 for k in topks}}
    summary: dict[str, Any] = {"sample_count": len(rows), "metrics": {}}
    for k in topks:
        hit_key = f"top{k}_hit"
        hits = [float(row.get(hit_key, 0.0)) for row in rows]
        summary["metrics"][f"top{k}_acc"] = sum(hits) / len(hits)
    return summary


def flush_run_outputs(
    *,
    run_dir: Path,
    samples: list[Sample],
    prediction_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]] | None = None,
    retrieval_topks: list[int] | None = None,
) -> None:
    success_count = len(metric_rows)
    failure_count = len(failures)
    considered_count = success_count + failure_count
    success_only_metrics = aggregate_numeric_rows(metric_rows)
    failure_denominator_metrics = metrics_with_custom_denominator(
        success_only_metrics,
        success_count=success_count,
        denominator_count=considered_count,
    )

    dump_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    dump_jsonl(run_dir / "parsed_predictions.jsonl", parsed_rows)
    dump_jsonl(run_dir / "metrics" / "per_sample.jsonl", metric_rows)
    dump_json(
        run_dir / "metrics" / "summary.json",
        {
            "sample_count": len(samples),
            "success_count": success_count,
            "failure_count": failure_count,
            "considered_count": considered_count,
            "metrics": success_only_metrics,
            "metrics_success_only": success_only_metrics,
            "metrics_with_failures_in_denominator": failure_denominator_metrics,
        },
    )
    dump_json(run_dir / "metrics" / "by_group.json", aggregate_by(metric_rows, "group"))
    dump_json(run_dir / "metrics" / "by_glyph.json", aggregate_by(metric_rows, "glyph_category"))
    dump_json(run_dir / "metrics" / "by_prompt.json", aggregate_by(metric_rows, "prompt_name"))
    dump_json(run_dir / "failures.json", {"failure_count": failure_count, "failures": failures})

    if retrieval_rows is not None and retrieval_topks:
        dump_jsonl(run_dir / "metrics" / "retrieval_per_sample.jsonl", retrieval_rows)
        dump_json(
            run_dir / "metrics" / "retrieval_summary.json",
            summarize_retrieval_metrics(retrieval_rows, retrieval_topks),
        )


def score_sample(
    sample: Sample,
    prediction: dict[str, Any],
    bert_scorer: BertScorer | None,
    judge_scores: dict[str, float] | None,
) -> dict[str, Any]:
    gold = sample.ground_truth
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "slug": sample.slug,
        "group": sample.group,
        "difficulty": sample.difficulty,
        "glyph_category": sample.glyph_category,
        "prompt_name": sample.prompt_name,
        "headword_acc": accuracy(prediction["现代字典字头"], gold["现代字典字头"]),
        "glyph_acc": glyph_accuracy(prediction["字形"], gold["字形"]),
        "structure_acc": accuracy(prediction["结构"], gold["结构"]),
        "liushu_strict_acc": set_exact_match(prediction["造字法"], gold["造字法"]),
        "liushu_partial_credit": set_recall_credit(prediction["造字法"], gold["造字法"]),
    }

    pred_component_map = prediction["构件"]
    gold_component_map = gold["构件"]
    pred_component_names = sorted(pred_component_map.keys())
    gold_component_names = sorted(gold_component_map.keys())

    name_matches = match_component_names(gold_component_names, pred_component_names, anls_threshold=0.8)
    matched_total = float(len(name_matches))
    gold_total = float(len(gold_component_names))
    pred_total = float(len(pred_component_names))

    function_correct = 0.0
    evolution_correct = 0.0
    explanation_bertscore_sum = 0.0
    for gold_name, pred_name in name_matches.items():
        pred_component = pred_component_map[pred_name]
        gold_component = gold_component_map[gold_name]
        function_correct += set_exact_match(pred_component["功能"], gold_component["功能"])
        evolution_correct += accuracy(pred_component["演变类型"], gold_component["演变类型"])
        explanation_bertscore = bertscore_score_or_none_when_both_empty(
            pred_component["解释"],
            gold_component["解释"],
            bert_scorer,
        )
        explanation_bertscore_sum += 0.0 if explanation_bertscore is None else explanation_bertscore

    row.update(
        build_component_metrics(
            gold_total=gold_total,
            pred_total=pred_total,
            match_total=matched_total,
            function_correct=function_correct,
            evolution_correct=evolution_correct,
            explanation_sum=explanation_bertscore_sum,
        )
    )

    original_meaning_bertscore = bertscore_score_or_none_when_both_empty(prediction["本义"], gold["本义"], bert_scorer)
    if original_meaning_bertscore is not None:
        row["original_meaning_bertscore"] = original_meaning_bertscore
    # Only BERTScore is retained for original meaning; all other text-similarity metrics removed.

    if judge_scores:
        row.update({f"evolution_judge_{key}": value for key, value in judge_scores.items()})
    return row


def aggregate_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)
    return {key: aggregate_numeric_rows(value) for key, value in sorted(buckets.items())}


def record_failure(failures: list[dict[str, Any]], item: dict[str, Any]) -> None:
    sample_id = str(item.get("sample_id", ""))
    failures[:] = [failure for failure in failures if str(failure.get("sample_id", "")) != sample_id]
    failures.append(item)


def clear_failure(failures: list[dict[str, Any]], sample_id: str) -> None:
    failures[:] = [failure for failure in failures if str(failure.get("sample_id", "")) != sample_id]


def prepare_run_dir(run_root: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or timestamp
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_sample_index(samples: list[Sample]) -> dict[str, Sample]:
    return {sample.sample_id: sample for sample in samples}


def iter_existing_sample_ids(run_dir: Path) -> list[str]:
    parsed_dir = run_dir / "parsed_predictions"
    if parsed_dir.exists():
        return sorted(path.stem for path in parsed_dir.glob("*.json"))

    parsed_jsonl = run_dir / "parsed_predictions.jsonl"
    if parsed_jsonl.exists():
        sample_ids: list[str] = []
        with parsed_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if sample_id:
                    sample_ids.append(str(sample_id))
        return sorted(set(sample_ids))

    return []


def load_existing_parsed_prediction(run_dir: Path, sample_id: str) -> dict[str, Any]:
    parsed_path = run_dir / "parsed_predictions" / f"{sample_id}.json"
    if parsed_path.exists():
        return json.loads(parsed_path.read_text(encoding="utf-8"))

    parsed_jsonl = run_dir / "parsed_predictions.jsonl"
    if parsed_jsonl.exists():
        with parsed_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("sample_id") == sample_id and isinstance(row.get("prediction"), dict):
                    return row["prediction"]

    raise FileNotFoundError(f"parsed prediction missing for sample_id={sample_id}")


def load_existing_judge_scores(run_dir: Path, sample_id: str) -> dict[str, float] | None:
    judge_path = run_dir / "judge_outputs" / f"{sample_id}.json"
    if not judge_path.exists():
        return None
    judge_bundle = json.loads(judge_path.read_text(encoding="utf-8"))
    parsed = judge_bundle.get("parsed")
    if not isinstance(parsed, dict):
        return None
    if any(
        legacy_key in parsed
        for legacy_key in ("fact_accuracy", "information_completeness", "no_unsupported_content", "clarity_coherence")
    ):
        return normalize_judge_scores(parsed)

    scores: dict[str, float] = {}
    for key, value in parsed.items():
        score_value: Any
        if isinstance(value, dict):
            score_value = value.get("score")
        else:
            score_value = value
        try:
            scores[str(key)] = float(score_value)
        except (TypeError, ValueError):
            continue
    if scores and "overall_score" not in scores:
        dims = [v for k, v in scores.items() if k != "overall_score"]
        if dims:
            scores["overall_score"] = sum(dims) / len(dims)
    return scores or None


def run_llm_judge_batch(
    *,
    run_dir: Path,
    judge_config: dict[str, Any],
    judge_model_config: dict[str, Any],
    judge_concurrency: int,
    target_sample_ids: set[str],
) -> tuple[dict[str, dict[str, float]], dict[str, str], Path]:
    llm_judge_cfg = dict(judge_config.get("llm_judge") or {})
    dimensions = str(llm_judge_cfg.get("dimensions", "fact_alignment,scholarly_expression"))
    dimension_prompts_dir = str(llm_judge_cfg.get("dimension_prompts_dir", "llm_judge/prompts/dreambenchpp_v1"))
    cases_root = Path(str(llm_judge_cfg.get("cases_root", "llm_judge/_batch_cases")))
    retry_cases_root = Path(str(llm_judge_cfg.get("retry_cases_root", "llm_judge/_retry_cases")))
    results_root = Path(str(llm_judge_cfg.get("results_root", "llm_judge/results/by_run")))
    max_tokens = int(llm_judge_cfg.get("max_tokens", judge_config.get("max_tokens", 1000)))
    temperature = float(llm_judge_cfg.get("temperature", 0.0))
    concurrency = max(1, int(llm_judge_cfg.get("concurrency", judge_concurrency)))

    run_rel = run_dir.relative_to(REPO_ROOT)
    run_name = run_dir.name
    case_dir_rel = cases_root / run_name
    case_dir = (REPO_ROOT / case_dir_rel).resolve()
    retry_case_dir_rel = retry_cases_root / run_name
    retry_case_dir = (REPO_ROOT / retry_case_dir_rel).resolve()
    result_parent_rel = results_root / run_name
    result_parent_dir = (REPO_ROOT / result_parent_rel).resolve()

    prepare_cmd = [
        sys.executable,
        "llm_judge/prepare_gold_test.py",
        "--run-dir",
        str(run_rel),
        "--output-dir",
        str(case_dir_rel),
        "--overwrite",
    ]
    print("[judge] cmd:", " ".join(prepare_cmd), flush=True)
    completed = subprocess.run(prepare_cmd, cwd=REPO_ROOT, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_gold_test failed with returncode={completed.returncode}")

    if retry_case_dir.exists():
        shutil.rmtree(retry_case_dir)
    retry_case_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sample_id in sorted(target_sample_ids):
        src = case_dir / f"{sample_id}.json"
        if not src.exists():
            continue
        shutil.copy2(src, retry_case_dir / src.name)
        copied += 1
    if copied == 0:
        raise RuntimeError("no llm_judge cases copied for target sample_ids")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as tmp_file:
        json.dump(judge_model_config, tmp_file, ensure_ascii=False, indent=2)
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
            str(concurrency),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
        ]
        print("[judge] cmd:", " ".join(judge_cmd), flush=True)
        completed = subprocess.run(judge_cmd, cwd=REPO_ROOT, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"run_llm_judge_test failed with returncode={completed.returncode}")
    finally:
        model_config_path.unlink(missing_ok=True)

    result_dir = find_latest_subdir(result_parent_dir)
    result_rows = load_jsonl(result_dir / "results.jsonl")
    per_sample_dir = result_dir / "per_sample"

    score_map: dict[str, dict[str, float]] = {}
    error_map: dict[str, str] = {}
    for row in result_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        status = str(row.get("status") or "")
        if status == "ok" and isinstance(row.get("scores"), dict):
            scores = parse_numeric_scores(row["scores"])
            if not scores:
                error_map[sample_id] = "judge returned empty scores"
                continue
            score_map[sample_id] = scores

            reasons: dict[str, str] = {}
            per_sample_path = per_sample_dir / f"{sample_id}.json"
            if per_sample_path.exists():
                per_sample = json.loads(per_sample_path.read_text(encoding="utf-8"))
                if isinstance(per_sample.get("reasons"), dict):
                    reasons = {str(k): str(v) for k, v in per_sample["reasons"].items()}

            parsed_payload: dict[str, Any] = {}
            for key, value in scores.items():
                if key == "overall_score":
                    parsed_payload[key] = float(value)
                else:
                    parsed_payload[key] = {"score": float(value), "reason": reasons.get(key, "")}
            dump_json(
                run_dir / "judge_outputs" / f"{sample_id}.json",
                {
                    "source": "llm_judge",
                    "result_dir": str(result_dir.relative_to(REPO_ROOT)),
                    "parsed": parsed_payload,
                },
            )
            continue

        reason = str(row.get("reason") or row.get("error") or f"status={status}")
        error_map[sample_id] = reason

    return score_map, error_map, result_dir


def build_rag_context(retrieved_items: list[dict[str, Any]]) -> str:
    lines = [
        "以下是从古文字词条库检索得到的候选字头资料（按相关度排序）。",
        "每个候选都提供该字头在词条文件中的完整内容。",
        "请结合图片与这些资料作答，但如果资料与图片冲突，以图片可证据内容为准。",
        "",
    ]
    for idx, item in enumerate(retrieved_items, start=1):
        char = str(item.get("char", ""))
        score = float(item.get("score", 0.0))
        entry_text = str(item.get("entry_text", ""))
        if not char and not entry_text:
            continue
        lines.append(f"[候选{idx}] 字头：{char}；相似度：{score:.4f}")
        lines.append(entry_text)
        lines.append("")
    return "\n".join(lines).strip()


def run_retrieval_for_sample(
    *,
    sample: Sample,
    run_dir: Path,
    retriever: ImageToHeadwordRetriever,
    top_k_context: int,
    retrieval_pool_k: int,
    retrieval_topks: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_path = run_dir / "retrievals" / f"{sample.sample_id}.json"
    if retrieval_path.exists():
        payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        items = payload.get("retrieved_headwords") or []
        if isinstance(items, list):
            item_list = [dict(item) for item in items if isinstance(item, dict)]
        else:
            item_list = []
    else:
        top_k_fetch = max(top_k_context, max(retrieval_topks))
        retrieved = retriever.retrieve(
            image_path=sample.image_path,
            top_k_headwords=top_k_fetch,
            score_pool_k=retrieval_pool_k,
        )
        item_list = [
            {
                "char": item.char,
                "score": item.score,
                "entry_text": item.entry_text,
                "best_chunk_text": item.best_chunk_text,
                "best_chunk_metadata": item.best_chunk_metadata,
            }
            for item in retrieved
        ]
        payload = {
            "sample_id": sample.sample_id,
            "slug": sample.slug,
            "query_char": sample.character,
            "query_image": str(sample.image_path.relative_to(REPO_ROOT)),
            "retrieved_headwords": item_list,
        }
        dump_json(retrieval_path, payload)

    retrieved_chars = [str(item.get("char", "")) for item in item_list]
    retrieval_row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "slug": sample.slug,
        "group": sample.group,
        "difficulty": sample.difficulty,
        "glyph_category": sample.glyph_category,
        "query_char": sample.character,
        "retrieved_chars": retrieved_chars,
    }
    for k in retrieval_topks:
        retrieval_row[f"top{k}_hit"] = 1.0 if sample.character in retrieved_chars[:k] else 0.0
    return item_list, retrieval_row


def process_sample(
    *,
    sample: Sample,
    run_dir: Path,
    client: OpenAICompatibleClient,
    generation_config: dict[str, Any],
    existing_run_only: bool,
    retriever: ImageToHeadwordRetriever | None,
    top_k_context: int,
    retrieval_pool_k: int,
    retrieval_topks: list[int],
    force_regenerate: bool = False,
) -> dict[str, Any]:
    raw_path = run_dir / "raw_responses" / f"{sample.sample_id}.json"
    parsed_path = run_dir / "parsed_predictions" / f"{sample.sample_id}.json"

    retrieval_items: list[dict[str, Any]] = []
    retrieval_row: dict[str, Any] | None = None
    if retriever is not None:
        retrieval_items, retrieval_row = run_retrieval_for_sample(
            sample=sample,
            run_dir=run_dir,
            retriever=retriever,
            top_k_context=top_k_context,
            retrieval_pool_k=retrieval_pool_k,
            retrieval_topks=retrieval_topks,
        )

    if existing_run_only:
        parsed_prediction = load_existing_parsed_prediction(run_dir, sample.sample_id)
    else:
        if not force_regenerate and parsed_path.exists() and raw_path.exists():
            parsed_prediction = json.loads(parsed_path.read_text(encoding="utf-8"))
        else:
            user_text = sample.user_prompt
            if retrieval_items:
                user_text = f"{sample.user_prompt}\n\n[RAG检索上下文]\n{build_rag_context(retrieval_items[:top_k_context])}"

            request_body, response_json = client.chat_completion(
                system_prompt=sample.system_prompt,
                user_text=user_text,
                image_path=sample.image_path,
                few_shot_messages=sample.few_shot_messages,
                temperature=float(generation_config.get("temperature", 0.0)),
                max_tokens=int(generation_config.get("max_tokens", 1200)),
                response_format={"type": "json_object"},
            )
            raw_bundle = {"request": request_body, "response": response_json}
            parsed_prediction = normalize_prediction(extract_json_object(extract_message_text(response_json)))
            dump_json(raw_path, raw_bundle)
            dump_json(parsed_path, parsed_prediction)

    prediction_row = {
        "sample_id": sample.sample_id,
        "slug": sample.slug,
        "image_path": str(sample.image_path.relative_to(REPO_ROOT)),
        "ground_truth": sample.ground_truth,
        "prediction": parsed_prediction,
    }
    if retrieval_items:
        prediction_row["rag_retrieved_headwords"] = [item.get("char", "") for item in retrieval_items[:top_k_context]]

    return {
        "sample": sample,
        "prediction_row": prediction_row,
        "parsed_row": {"sample_id": sample.sample_id, **parsed_prediction},
        "parsed_prediction": parsed_prediction,
        "retrieval_row": retrieval_row,
    }


def sanitize_name(name: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in name.strip())
    normalized = normalized.strip("-_")
    return normalized or "model"


def run_batch_local_models(args: argparse.Namespace) -> int:
    if args.existing_run_dir:
        raise ValueError("--batch-local-model-root cannot be used with --existing-run-dir")
    if args.retry_failures_from_run:
        raise ValueError("--batch-local-model-root cannot be used with --retry-failures-from-run")

    template_model_config_path = Path(args.model_config).resolve()
    model_root = Path(args.batch_local_model_root).resolve()
    if not model_root.exists() or not model_root.is_dir():
        raise FileNotFoundError(f"batch local model root not found: {model_root}")

    model_dirs = sorted(path for path in model_root.iterdir() if path.is_dir() and (args.batch_include_hidden or not path.name.startswith(".")))
    if not model_dirs:
        raise ValueError(f"no model subdirectories found under {model_root}")

    template_model_config = load_yaml(template_model_config_path)
    run_config_for_batch = load_yaml(Path(args.run_config).resolve())
    configured_generation_max_tokens = int((run_config_for_batch.get("generation") or {}).get("max_tokens", 1200))

    model_path_key = args.batch_model_path_key
    model_name_key = args.batch_model_name_key
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    success = 0
    failed = 0
    print(f"[batch] found {len(model_dirs)} model directories under {model_root}", flush=True)
    with tempfile.TemporaryDirectory(prefix="jiezi-batch-model-config-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for index, model_dir in enumerate(model_dirs, start=1):
            model_name = sanitize_name(model_dir.name)
            batch_model_config = copy.deepcopy(template_model_config)
            batch_model_config[model_path_key] = str(model_dir)
            if model_name_key:
                batch_model_config[model_name_key] = model_name
            if args.batch_max_model_len > 0:
                batch_model_config["max_model_len"] = int(args.batch_max_model_len)

            model_max_len = int(batch_model_config.get("max_model_len") or 0)
            if args.generation_max_tokens > 0:
                effective_generation_max_tokens = int(args.generation_max_tokens)
            elif model_max_len > 0:
                effective_generation_max_tokens = min(configured_generation_max_tokens, model_max_len)
            else:
                effective_generation_max_tokens = 0

            temp_model_config_path = tmp_dir_path / f"{index:03d}_{model_name}.yaml"
            temp_model_config_path.write_text(
                yaml.safe_dump(batch_model_config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            if args.run_name:
                run_name = sanitize_name(f"{args.run_name}-{model_name}")
            else:
                run_name = sanitize_name(f"{timestamp}-{model_name}")

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-config",
                str(Path(args.run_config).resolve()),
                "--model-config",
                str(temp_model_config_path),
                "--run-name",
                run_name,
            ]
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            if args.retrieval_only:
                cmd.append("--retrieval-only")
            if args.api_concurrency:
                cmd.extend(["--api-concurrency", str(args.api_concurrency)])
            if args.judge_concurrency:
                cmd.extend(["--judge-concurrency", str(args.judge_concurrency)])
            if effective_generation_max_tokens > 0:
                cmd.extend(["--generation-max-tokens", str(effective_generation_max_tokens)])

            # Batch mode defaults to judge=off unless explicitly set.
            batch_judge = args.judge if args.judge != "auto" else "off"
            cmd.extend(["--judge", batch_judge])

            print(f"[batch] ({index}/{len(model_dirs)}) start model_dir={model_dir}", flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                success += 1
                print(f"[batch] ({index}/{len(model_dirs)}) done model_dir={model_dir}", flush=True)
            else:
                failed += 1
                print(
                    f"[batch] ({index}/{len(model_dirs)}) failed model_dir={model_dir} returncode={result.returncode}",
                    flush=True,
                )

    print(f"[batch] finished: success={success}, failed={failed}, total={len(model_dirs)}", flush=True)
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ancient character exegesis benchmark.")
    parser.add_argument("--run-config", required=True, help="Path to run yaml config.")
    parser.add_argument("--model-config", required=True, help="Path to model yaml config.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", choices=["on", "off", "auto"], default="auto",
                        help="LLM-judge mode: 'on'=force enable, 'off'=force disable, 'auto'=use config setting.")
    parser.add_argument("--retrieval-only", action="store_true", help="Only run retrieval and top-k retrieval metrics.")
    parser.add_argument("--existing-run-dir", default="", help="Score an existing run directory without re-calling the model.")
    parser.add_argument(
        "--retry-failures-from-run",
        default="",
        help="Rerun only failed sample_ids from the given run directory (reads failures.json).",
    )
    parser.add_argument("--api-concurrency", type=int, default=0, help="Concurrent API requests. Defaults to config value or 1.")
    parser.add_argument("--judge-concurrency", type=int, default=0, help="Concurrent judge API requests. Defaults to config value or 8.")
    parser.add_argument("--generation-max-tokens", type=int, default=0, help="Override generation.max_tokens (>0 to enable).")
    parser.add_argument(
        "--batch-local-model-root",
        default="",
        help="Parent directory whose first-level subdirectories are treated as local model directories for sequential eval.",
    )
    parser.add_argument(
        "--batch-model-path-key",
        default="model_path",
        help="Model-config key to override with each local model subdirectory path.",
    )
    parser.add_argument(
        "--batch-model-name-key",
        default="model",
        help="Model-config key to override with each subdirectory name; set empty string to disable.",
    )
    parser.add_argument(
        "--batch-include-hidden",
        action="store_true",
        help="Include hidden subdirectories when scanning --batch-local-model-root.",
    )
    parser.add_argument(
        "--batch-max-model-len",
        type=int,
        default=0,
        help="Override model_config.max_model_len for each batch model (>0 to enable).",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("JIEZI_CACHE_ROOT", os.path.expanduser("~/.cache/jiezi-bench")),
        help="Root directory for huggingface/modelscope/runtime caches.",
    )
    args = parser.parse_args()
    setup_cache_environment(args.cache_root)
    ensure_canonical_llm_judge_layout(REPO_ROOT)

    if args.batch_local_model_root:
        exit_code = run_batch_local_models(args)
        raise SystemExit(exit_code)

    no_proxy_value = os.environ.get("NO_PROXY", "")
    local_hosts = {"127.0.0.1", "localhost"}
    current_hosts = {item.strip() for item in no_proxy_value.split(",") if item.strip()}
    if not local_hosts.issubset(current_hosts):
        os.environ["NO_PROXY"] = ",".join(sorted(current_hosts | local_hosts))

    repo_root = REPO_ROOT
    run_config = load_yaml(Path(args.run_config))
    model_config = load_yaml(Path(args.model_config))
    if args.generation_max_tokens > 0:
        generation_cfg = dict(run_config.get("generation") or {})
        generation_cfg["max_tokens"] = int(args.generation_max_tokens)
        run_config["generation"] = generation_cfg

    task_config = run_config["task"]
    data_config = run_config["data"]
    metric_config = run_config["metrics"]
    rag_config = run_config.get("rag", {})
    rag_enabled = bool(rag_config.get("enabled", False))

    if args.retrieval_only and not rag_enabled:
        raise ValueError("--retrieval-only requires run_config.rag.enabled=true")

    prompt_name = str(model_config.get("prompt_name", "complex"))

    samples = iter_samples(
        repo_root=repo_root,
        data_root=repo_root / data_config["data_root"],
        prompt_name=prompt_name,
        limit=args.limit or int(run_config.get("limit", 0)),
        slug_filters=run_config.get("slug_filters") or [],
        group_filters=run_config.get("group_filters") or [],
    )
    all_sample_ids = {sample.sample_id for sample in samples}

    existing_run_only = bool(args.existing_run_dir)
    retry_failures_from_run = Path(args.retry_failures_from_run).resolve() if args.retry_failures_from_run else None
    retry_failed_sample_ids: set[str] = set()
    retry_mode = retry_failures_from_run is not None
    run_dir_preexisting = False
    if existing_run_only:
        run_dir = Path(args.existing_run_dir).resolve()
    else:
        run_dir = prepare_run_dir(repo_root / run_config.get("run_root", "runs/ancient_char_exegesis"), args.run_name or run_config.get("run_name"))
        run_dir_preexisting = (run_dir / "predictions.jsonl").exists() or (run_dir / "parsed_predictions.jsonl").exists()
        dump_json(
            run_dir / "manifest.json",
            {
                "task": task_config["name"],
                "prompt_name": prompt_name,
                "sample_count": len(samples),
                "run_config": run_config,
                "model_config": redact_sensitive_config(model_config),
            },
        )

    if retry_failures_from_run:
        retry_failed_sample_ids = load_failed_sample_ids(retry_failures_from_run)
        if not retry_failed_sample_ids:
            print(f"[retry-failures] no failed sample_ids found under {retry_failures_from_run}", flush=True)
            return
        missing = sorted(sample_id for sample_id in retry_failed_sample_ids if sample_id not in all_sample_ids)
        if missing:
            print(f"[retry-failures] warning: {len(missing)} sample_ids not found in current dataset", flush=True)
        retry_failed_sample_ids = {sample_id for sample_id in retry_failed_sample_ids if sample_id in all_sample_ids}
        samples = [sample for sample in samples if sample.sample_id in retry_failed_sample_ids]
        print(f"[retry-failures] target {len(samples)} failed samples from {retry_failures_from_run}", flush=True)

    server_process: subprocess.Popen[str] | None = None
    embedding_server_process: subprocess.Popen[str] | None = None

    retriever: ImageToHeadwordRetriever | None = None
    retrieval_topks = sorted({int(k) for k in rag_config.get("eval_topks", [1, 5, 10]) if int(k) > 0})
    top_k_context = int(rag_config.get("top_k_context", 5))
    retrieval_pool_k = int(rag_config.get("score_pool_k", 128))

    if rag_enabled:
        embedding_model_config_path = rag_config.get("embedding_model_config")
        if not embedding_model_config_path:
            raise ValueError("run_config.rag.embedding_model_config is required when rag.enabled=true")
        embedding_model_config = load_yaml(repo_root / str(embedding_model_config_path))
        if embedding_model_config.get("backend") == "vllm_managed" and embedding_model_config.get("manage_server", True):
            embedding_server_process = start_vllm_server(embedding_model_config)
            wait_for_server(
                embedding_model_config,
                timeout=int(embedding_model_config.get("startup_timeout", 600)),
                process=embedding_server_process,
            )
        embedding_client = OpenAICompatibleClient.from_config(embedding_model_config)
        retriever = ImageToHeadwordRetriever(
            embedding_client=embedding_client,
            chunk_jsonl_path=repo_root / str(rag_config["chunk_jsonl_path"]),
            split_entries_txt_dir=repo_root / str(rag_config["split_entries_txt_dir"]),
            cache_dir=repo_root / str(rag_config.get("embedding_cache_dir", "runs/rag_cache")),
            embedding_batch_size=int(rag_config.get("embedding_batch_size", 64)),
        )

    client: OpenAICompatibleClient | None = None
    if not args.retrieval_only and not existing_run_only:
        if model_config.get("backend") == "vllm_managed" and model_config.get("manage_server", True):
            server_process = start_vllm_server(model_config)
            wait_for_server(model_config, timeout=int(model_config.get("startup_timeout", 600)), process=server_process)
        client = OpenAICompatibleClient.from_config(model_config)
    elif not args.retrieval_only:
        client = OpenAICompatibleClient.from_config(model_config)

    bert_scorer = BertScorer(
        model_type=str(metric_config["bertscore"]["model_type"]),
        lang=str(metric_config["bertscore"].get("lang", "zh")),
        num_layers=int(metric_config["bertscore"].get("num_layers", 8)),
        all_layers=bool(metric_config["bertscore"].get("all_layers", True)),
        layer_agg=str(metric_config["bertscore"].get("layer_agg", "mean")),
    ) if metric_config.get("bertscore", {}).get("enabled", True) and not args.retrieval_only else None

    if args.judge == "on":
        judge_enabled = not args.retrieval_only
    elif args.judge == "off":
        judge_enabled = False
    else:  # auto
        judge_enabled = not args.retrieval_only and run_config.get("judge", {}).get("enabled", False)

    if not judge_enabled and run_config.get("judge", {}).get("enabled") and not args.retrieval_only:
        print(f"[judge] disabled (--judge={args.judge})", flush=True)

    judge_model_config = run_config.get("judge", {}).get("model")
    if judge_enabled and not isinstance(judge_model_config, dict):
        raise ValueError("judge.enabled=true requires run_config.judge.model when using llm_judge pipeline")

    api_concurrency = max(1, int(args.api_concurrency or run_config.get("api_concurrency") or model_config.get("api_concurrency") or 1))
    judge_concurrency = max(1, int(args.judge_concurrency or run_config.get("judge", {}).get("api_concurrency") or 8))

    sample_index = build_sample_index(samples)
    if existing_run_only:
        existing_sample_ids = iter_existing_sample_ids(run_dir)
        if not existing_sample_ids and not args.retrieval_only:
            raise FileNotFoundError(f"no parsed predictions found under {run_dir}")
        if existing_sample_ids:
            samples = [sample_index[sample_id] for sample_id in existing_sample_ids if sample_id in sample_index]
        prediction_rows: list[dict[str, Any]] = []
        parsed_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        retrieval_rows = load_jsonl(run_dir / "metrics" / "retrieval_per_sample.jsonl")
    elif run_dir_preexisting:
        prediction_rows = load_jsonl(run_dir / "predictions.jsonl")
        parsed_rows = load_jsonl(run_dir / "parsed_predictions.jsonl")
        metric_rows = load_jsonl(run_dir / "metrics" / "per_sample.jsonl")
        retrieval_rows = load_jsonl(run_dir / "metrics" / "retrieval_per_sample.jsonl")
        failures_bundle = json.loads((run_dir / "failures.json").read_text(encoding="utf-8")) if (run_dir / "failures.json").exists() else {}
        failures = list(failures_bundle.get("failures") or [])
        completed_sample_ids = {str(row.get("sample_id")) for row in parsed_rows if row.get("sample_id")}
        if completed_sample_ids and not args.retrieval_only and not retry_mode:
            samples = [sample for sample in samples if sample.sample_id not in completed_sample_ids]
            print(f"[resume] loaded {len(completed_sample_ids)} completed samples, {len(samples)} remaining", flush=True)
    else:
        prediction_rows = []
        parsed_rows = []
        metric_rows = []
        failures = []
        retrieval_rows = []

    try:
        if args.retrieval_only:
            if retriever is None:
                raise RuntimeError("retriever is not initialized")
            total = len(samples)
            for index, sample in enumerate(samples, start=1):
                try:
                    _, retrieval_row = run_retrieval_for_sample(
                        sample=sample,
                        run_dir=run_dir,
                        retriever=retriever,
                        top_k_context=top_k_context,
                        retrieval_pool_k=retrieval_pool_k,
                        retrieval_topks=retrieval_topks,
                    )
                    retrieval_rows = [row for row in retrieval_rows if row.get("sample_id") != sample.sample_id]
                    retrieval_rows.append(retrieval_row)
                    clear_failure(failures, sample.sample_id)
                    print(f"[retrieval-ok] {index}/{total} {sample.sample_id}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    record_failure(
                        failures,
                        {
                            "sample_id": sample.sample_id,
                            "slug": sample.slug,
                            "image_path": str(sample.image_path.relative_to(repo_root)),
                            "error": f"retrieval_failed: {exc}",
                        },
                    )
                    print(f"[retrieval-error] {index}/{total} {sample.sample_id}: {exc}", flush=True)
                finally:
                    flush_run_outputs(
                        run_dir=run_dir,
                        samples=samples,
                        prediction_rows=prediction_rows,
                        parsed_rows=parsed_rows,
                        metric_rows=metric_rows,
                        failures=failures,
                        retrieval_rows=retrieval_rows,
                        retrieval_topks=retrieval_topks,
                    )
            return

        if existing_run_only:
            if client is None:
                client = OpenAICompatibleClient.from_config(model_config)
            for index, sample in enumerate(samples, start=1):
                try:
                    result = process_sample(
                        sample=sample,
                        run_dir=run_dir,
                        client=client,
                        generation_config=run_config.get("generation", {}),
                        existing_run_only=True,
                        retriever=retriever,
                        top_k_context=top_k_context,
                        retrieval_pool_k=retrieval_pool_k,
                        retrieval_topks=retrieval_topks,
                    )
                    upsert_row_by_sample_id(prediction_rows, result["prediction_row"])
                    upsert_row_by_sample_id(parsed_rows, result["parsed_row"])
                    if result.get("retrieval_row"):
                        upsert_row_by_sample_id(retrieval_rows, result["retrieval_row"])
                    judge_scores = None if args.judge == "off" else load_existing_judge_scores(run_dir, sample.sample_id)
                    upsert_row_by_sample_id(metric_rows, score_sample(sample, result["parsed_prediction"], bert_scorer, judge_scores))
                    clear_failure(failures, sample.sample_id)
                    print(f"[ok] {index}/{len(samples)} {sample.sample_id}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    record_failure(
                        failures,
                        {
                            "sample_id": sample.sample_id,
                            "slug": sample.slug,
                            "image_path": str(sample.image_path.relative_to(repo_root)),
                            "error": str(exc),
                        },
                    )
                    print(f"[error] {index}/{len(samples)} {sample.sample_id}: {exc}", flush=True)
                finally:
                    flush_run_outputs(
                        run_dir=run_dir,
                        samples=samples,
                        prediction_rows=prediction_rows,
                        parsed_rows=parsed_rows,
                        metric_rows=metric_rows,
                        failures=failures,
                        retrieval_rows=retrieval_rows,
                        retrieval_topks=retrieval_topks,
                    )
        else:
            if client is None:
                raise RuntimeError("generation client is not initialized")

            stage1_results: dict[str, dict[str, Any]] = {}
            future_to_sample: dict[Any, Sample] = {}
            with ThreadPoolExecutor(max_workers=api_concurrency) as executor:
                for sample in samples:
                    future = executor.submit(
                        process_sample,
                        sample=sample,
                        run_dir=run_dir,
                        client=client,
                        generation_config=run_config.get("generation", {}),
                        existing_run_only=False,
                        retriever=retriever,
                        top_k_context=top_k_context,
                        retrieval_pool_k=retrieval_pool_k,
                        retrieval_topks=retrieval_topks,
                        force_regenerate=sample.sample_id in retry_failed_sample_ids,
                    )
                    future_to_sample[future] = sample

                completed = 0
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    completed += 1
                    try:
                        result = future.result()
                        upsert_row_by_sample_id(prediction_rows, result["prediction_row"])
                        upsert_row_by_sample_id(parsed_rows, result["parsed_row"])
                        if result.get("retrieval_row"):
                            upsert_row_by_sample_id(retrieval_rows, result["retrieval_row"])
                        stage1_results[sample.sample_id] = result
                        # Always compute base metrics (without judge scores) in stage 1,
                        # so results are preserved even if the judge stage fails later.
                        upsert_row_by_sample_id(
                            metric_rows, score_sample(sample, result["parsed_prediction"], bert_scorer, None)
                        )
                        clear_failure(failures, sample.sample_id)
                        print(f"[ok] {completed}/{len(samples)} {sample.sample_id}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        record_failure(
                            failures,
                            {
                                "sample_id": sample.sample_id,
                                "slug": sample.slug,
                                "image_path": str(sample.image_path.relative_to(repo_root)),
                                "error": str(exc),
                            },
                        )
                        print(f"[error] {completed}/{len(samples)} {sample.sample_id}: {exc}", flush=True)
                    finally:
                        flush_run_outputs(
                            run_dir=run_dir,
                            samples=samples,
                            prediction_rows=prediction_rows,
                            parsed_rows=parsed_rows,
                            metric_rows=metric_rows,
                            failures=failures,
                            retrieval_rows=retrieval_rows,
                            retrieval_topks=retrieval_topks,
                        )

            if judge_enabled and stage1_results:
                print(
                    f"[judge] stage-2 start: {len(stage1_results)} samples, "
                    f"judge_concurrency={judge_concurrency}, pipeline=llm_judge",
                    flush=True,
                )
                try:
                    judge_scores_map, judge_error_map, judge_result_dir = run_llm_judge_batch(
                        run_dir=run_dir,
                        judge_config=run_config.get("judge", {}),
                        judge_model_config=dict(judge_model_config or {}),
                        judge_concurrency=judge_concurrency,
                        target_sample_ids=set(stage1_results),
                    )
                    print(f"[judge] llm_judge result_dir={judge_result_dir.relative_to(repo_root)}", flush=True)
                except Exception as exc:
                    print(f"[judge] stage-2 failed: {exc} — base metrics from stage 1 are preserved", flush=True)
                else:
                    for index, sample_id in enumerate(sorted(stage1_results), start=1):
                        result = stage1_results[sample_id]
                        sample = result["sample"]
                        judge_scores = judge_scores_map.get(sample_id)
                        if judge_scores is not None:
                            # Re-score with judge scores to enhance the base metrics.
                            upsert_row_by_sample_id(
                                metric_rows, score_sample(sample, result["parsed_prediction"], bert_scorer, judge_scores)
                            )
                            print(f"[judge-ok] {index}/{len(stage1_results)} {sample.sample_id}", flush=True)
                        else:
                            reason = judge_error_map.get(sample_id, "missing judge result")
                            print(f"[judge-warn] {index}/{len(stage1_results)} {sample.sample_id}: {reason} (base metrics preserved)", flush=True)
                finally:
                    flush_run_outputs(
                        run_dir=run_dir,
                        samples=samples,
                        prediction_rows=prediction_rows,
                        parsed_rows=parsed_rows,
                        metric_rows=metric_rows,
                        failures=failures,
                        retrieval_rows=retrieval_rows,
                        retrieval_topks=retrieval_topks,
                    )
    finally:
        if server_process is not None:
            server_process.send_signal(signal.SIGTERM)
        if embedding_server_process is not None:
            embedding_server_process.send_signal(signal.SIGTERM)


if __name__ == "__main__":
    main()
