#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download default JieZi-bench local models to modelscope cache.")
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope/models")),
        help="Model cache root; each repo is downloaded to <cache-root>/<org>/<name>.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="ModelScope repo id, e.g. Qwen/Qwen3.5-4B. Can be provided multiple times.",
    )
    args = parser.parse_args()

    default_models = [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-VL-Embedding-8B",
        "OpenGVLab/InternVL3_5-4B",
        "OpenGVLab/InternVL3_5-8B",
    ]
    models = args.model or default_models
    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    for repo_id in models:
        target_dir = cache_root / repo_id
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {repo_id} -> {target_dir}", flush=True)
        snapshot_download(repo_id, local_dir=str(target_dir))
    print("[done] all requested models downloaded", flush=True)


if __name__ == "__main__":
    main()
