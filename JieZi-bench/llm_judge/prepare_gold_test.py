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

from benchmarks.common.io_utils import as_text, dump_json, load_json, load_jsonl
from benchmarks.common.openai_client import extract_json_object, extract_message_text


def extract_prediction_from_raw(raw_path: Path) -> dict[str, Any]:
    raw = load_json(raw_path)
    response = raw.get("response")
    if not isinstance(response, dict):
        return {}
    try:
        content = extract_message_text(response)
        parsed = extract_json_object(content)
    except Exception:  # noqa: BLE001
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_case(
    *,
    sample_id: str,
    run_name: str,
    task_name: str,
    format_instruction: str,
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    raw_path: Path | None,
) -> dict[str, Any] | None:
    gt_evolution = as_text(ground_truth.get("历代字形演变"))
    pred_evolution = as_text(prediction.get("历代字形演变"))
    if not gt_evolution:
        return None

    return {
        "sample_id": sample_id,
        "run_name": run_name,
        "task_name": task_name,
        "format_instruction": format_instruction,
        "ground_truth_evolution": gt_evolution,
        "prediction_evolution": pred_evolution,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "source": {
            "raw_response_path": str(raw_path.relative_to(REPO_ROOT)) if raw_path and raw_path.exists() else "",
            "prepared_from": "predictions.jsonl",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized llm-judge test cases into llm_judge/gold-test.")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs/ancient_char_exegesis/20260319-160127")
    parser.add_argument(
        "--output-dir",
        default="llm_judge/gold-test",
        help="Output directory where normalized case JSON files are written",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases (0 means all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing case files")
    args = parser.parse_args()

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    raw_dir = run_dir / "raw_responses"

    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    run_cfg = manifest.get("run_config") if isinstance(manifest, dict) else {}
    task_cfg = run_cfg.get("task") if isinstance(run_cfg, dict) else {}
    task_name = as_text(task_cfg.get("name")) or as_text(manifest.get("task")) or "Ancient Chinese Character Exegesis"
    format_instruction = as_text(task_cfg.get("format_instruction"))

    rows = load_jsonl(predictions_path)
    if not rows:
        raise FileNotFoundError(f"predictions.jsonl missing or empty: {predictions_path}")

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for row in rows:
        sample_id = as_text(row.get("sample_id"))
        if not sample_id:
            skipped += 1
            continue

        ground_truth = row.get("ground_truth")
        prediction = row.get("prediction")
        if not isinstance(ground_truth, dict):
            skipped += 1
            continue
        if not isinstance(prediction, dict):
            prediction = {}

        raw_path = raw_dir / f"{sample_id}.json"
        if (not prediction or "历代字形演变" not in prediction) and raw_path.exists():
            prediction_from_raw = extract_prediction_from_raw(raw_path)
            if prediction_from_raw:
                prediction = prediction_from_raw

        case = build_case(
            sample_id=sample_id,
            run_name=run_dir.name,
            task_name=task_name,
            format_instruction=format_instruction,
            ground_truth=ground_truth,
            prediction=prediction,
            raw_path=raw_path,
        )
        if case is None:
            skipped += 1
            continue

        out_path = output_dir / f"{sample_id}.json"
        if out_path.exists() and not args.overwrite:
            continue
        dump_json(out_path, case)
        written += 1

        if args.limit > 0 and written >= args.limit:
            break

    manifest_out = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "written_cases": written,
        "skipped_rows": skipped,
        "total_prediction_rows": len(rows),
        "task_name": task_name,
    }
    dump_json(output_dir / "_manifest.json", manifest_out)
    print(json.dumps(manifest_out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
