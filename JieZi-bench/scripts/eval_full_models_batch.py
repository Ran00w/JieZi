#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def list_model_dirs(root: Path) -> list[Path]:
    items: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and (entry / "config.json").exists():
            items.append(entry)
    return items


def build_compat_stage_dir(src_model_dir: Path, stage_root: Path, ref_model_dir: Path) -> Path:
    stage_dir = stage_root / src_model_dir.name
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for item in src_model_dir.iterdir():
        if item.name == "processor_config.json":
            continue
        os.symlink(item.resolve(), stage_dir / item.name)

    for name in ("video_preprocessor_config.json", "configuration.json", "merges.txt", "vocab.json"):
        src = ref_model_dir / name
        dst = stage_dir / name
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
    return stage_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch eval all local models under a models directory.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--models-root",
        required=True,
        help="Parent directory containing per-model subdirectories.",
    )
    parser.add_argument("--run-config", default="configs/runs/ancient_char_exegesis.yaml")
    parser.add_argument("--base-model-config", required=True)
    parser.add_argument(
        "--reference-model-path",
        required=True,
        help="Path to reference model for compatibility symlinks.",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--api-concurrency", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--staging-root", default="/tmp/full_models_compat")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-models", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    models_root = Path(args.models_root).resolve()
    run_config = (repo_root / args.run_config).resolve()
    base_model_config = (repo_root / args.base_model_config).resolve()
    python_exec = args.python_exec
    reference_model_path = Path(args.reference_model_path).resolve()
    stage_root = Path(args.staging_root).resolve()

    if not models_root.exists():
        print(f"[fatal] models root not found: {models_root}")
        return 2
    if not reference_model_path.exists():
        print(f"[fatal] reference model path not found: {reference_model_path}")
        return 2

    model_dirs = list_model_dirs(models_root)
    if not model_dirs:
        print(f"[fatal] no model dirs found under: {models_root}")
        return 2
    if args.max_models > 0:
        model_dirs = model_dirs[: args.max_models]

    base_cfg = load_yaml(base_model_config)
    batch_tag = now_tag()
    generated_cfg_dir = repo_root / "configs" / "models" / "generated_full_models"
    report_dir = repo_root / "runs" / "batch_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cache_root = os.environ.get("JIEZI_CACHE_ROOT", os.path.expanduser("~/.cache/jiezi-bench"))
    env = os.environ.copy()
    env["HF_HOME"] = os.environ.get("HF_HOME", f"{cache_root}/huggingface")
    env["HUGGINGFACE_HUB_CACHE"] = os.environ.get("HUGGINGFACE_HUB_CACHE", f"{cache_root}/huggingface/hub")
    env["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", f"{cache_root}/huggingface/transformers")
    env["MODELSCOPE_CACHE"] = os.environ.get("MODELSCOPE_CACHE", f"{cache_root}/modelscope")

    results: list[dict[str, Any]] = []
    for idx, model_dir in enumerate(model_dirs, start=1):
        model_name = model_dir.name
        run_name = f"batch-full-{batch_tag}-{model_name}"
        cfg_path = generated_cfg_dir / f"{model_name}.yaml"
        compat_model_dir = build_compat_stage_dir(model_dir, stage_root, reference_model_path)

        cfg = dict(base_cfg)
        cfg["model"] = model_name
        cfg["model_path"] = str(compat_model_dir)
        cfg["tensor_parallel_size"] = int(args.tp)
        cfg["pipeline_parallel_size"] = int(args.pp)
        cfg["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
        cfg["max_model_len"] = int(args.max_model_len)
        cfg["max_num_seqs"] = int(args.max_num_seqs)
        cfg["timeout"] = max(int(cfg.get("timeout", 120)), 600)
        cfg["max_retries"] = int(cfg.get("max_retries", 1))
        cfg["port"] = int(cfg.get("port", 9988))
        cfg["cuda_visible_devices"] = str(cfg.get("cuda_visible_devices", "0,1"))

        dump_yaml(cfg_path, cfg)
        print(f"\n=== [{idx}/{len(model_dirs)}] {model_name} ===", flush=True)
        print(f"[info] config: {cfg_path}", flush=True)
        print(f"[info] run_name: {run_name}", flush=True)

        cmd = [
            python_exec,
            "runners/run_eval.py",
            "--run-config",
            str(run_config),
            "--model-config",
            str(cfg_path),
            "--run-name",
            run_name,
            "--judge", "off",
            "--api-concurrency",
            str(args.api_concurrency),
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        started = datetime.now().isoformat(timespec="seconds")
        rc = subprocess.run(cmd, cwd=repo_root, env=env).returncode
        finished = datetime.now().isoformat(timespec="seconds")
        run_dir = repo_root / "runs" / "ancient_char_exegesis" / run_name
        summary_path = run_dir / "metrics" / "summary.json"

        item: dict[str, Any] = {
            "model_name": model_name,
            "model_path": str(model_dir),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "return_code": rc,
            "started_at": started,
            "finished_at": finished,
            "summary_path": str(summary_path),
        }
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                item["summary"] = summary
                item["success_count"] = summary.get("success_count")
                item["failure_count"] = summary.get("failure_count")
                item["sample_count"] = summary.get("sample_count")
                metrics = summary.get("metrics") or {}
                item["headword_acc"] = metrics.get("headword_acc")
                item["glyph_acc"] = metrics.get("glyph_acc")
                item["structure_acc"] = metrics.get("structure_acc")
                item["liushu_partial_credit"] = metrics.get("liushu_partial_credit")
                item["original_meaning_bertscore"] = metrics.get("original_meaning_bertscore")
            except Exception as exc:  # noqa: BLE001
                item["summary_parse_error"] = str(exc)
        else:
            item["summary_missing"] = True

        results.append(item)

    report_json = report_dir / f"full_models_eval_report_{batch_tag}.json"
    report_csv = report_dir / f"full_models_eval_report_{batch_tag}.csv"
    report_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "model_name",
        "return_code",
        "sample_count",
        "success_count",
        "failure_count",
        "headword_acc",
        "glyph_acc",
        "structure_acc",
        "liushu_partial_credit",
        "original_meaning_bertscore",
        "run_name",
        "run_dir",
        "summary_path",
    ]
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print("\n=== batch done ===", flush=True)
    print(f"[report] {report_json}", flush=True)
    print(f"[report] {report_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
