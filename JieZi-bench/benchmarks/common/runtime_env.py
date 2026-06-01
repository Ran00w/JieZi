from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

def _resolve_cache_root(cache_root: str | Path | None = None) -> Path:
    if cache_root:
        return Path(cache_root).expanduser().resolve()
    env_root = os.environ.get("JIEZI_CACHE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.home() / ".cache" / "jiezi-bench"


def build_cache_env(cache_root: str | Path | None = None) -> dict[str, str]:
    root = _resolve_cache_root(cache_root)
    root = root.expanduser().resolve()
    hf_home = root / "huggingface"
    hf_hub = hf_home / "hub"
    transformers = hf_home / "transformers"
    ms_cache = root / "modelscope"
    ms_dataset = ms_cache / "datasets"
    xdg_cache = root / "xdg"
    mpl_cache = root / "matplotlib"
    for path in (root, hf_home, hf_hub, transformers, ms_cache, ms_dataset, xdg_cache, mpl_cache):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "JIEZI_CACHE_ROOT": str(root),
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub),
        "TRANSFORMERS_CACHE": str(transformers),
        "MODELSCOPE_CACHE": str(ms_cache),
        "MS_CACHE_HOME": str(ms_cache),
        "MODELSCOPE_HOME": str(ms_cache),
        "MODELSCOPE_DATASETS_CACHE": str(ms_dataset),
        "XDG_CACHE_HOME": str(xdg_cache),
        "MPLCONFIGDIR": str(mpl_cache),
    }


def setup_cache_environment(cache_root: str | Path | None = None, *, override: bool = False) -> Mapping[str, str]:
    env_values = build_cache_env(cache_root)
    for key, value in env_values.items():
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return env_values
