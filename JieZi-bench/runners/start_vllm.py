#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.vllm import build_vllm_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the vLLM launch command for a model config.")
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    print(" ".join(build_vllm_command(config)))


if __name__ == "__main__":
    main()
