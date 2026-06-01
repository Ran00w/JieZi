from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from benchmarks.common.openai_client import OpenAICompatibleClient
from benchmarks.common.runtime_env import build_cache_env


def build_vllm_command(config: dict[str, Any]) -> list[str]:
    vllm_exec = config.get("vllm_exec")
    python_exec = config.get("python_exec")
    if vllm_exec:
        command = [str(vllm_exec), "serve", str(config["model_path"])]
    elif python_exec:
        command = [str(python_exec), "-m", "vllm.entrypoints.openai.api_server"]
    else:
        command = ["python", "-m", "vllm.entrypoints.openai.api_server"]
    if not vllm_exec:
        command.extend(["--model", str(config["model_path"])])
    command.extend(["--host", str(config.get("host", "127.0.0.1"))])
    command.extend(["--port", str(config.get("port", 8000))])
    command.extend(["--served-model-name", str(config["model"])])
    if config.get("tensor_parallel_size"):
        command.extend(["--tensor-parallel-size", str(config["tensor_parallel_size"])])
    if config.get("pipeline_parallel_size"):
        command.extend(["--pipeline-parallel-size", str(config["pipeline_parallel_size"])])
    if config.get("dtype"):
        command.extend(["--dtype", str(config["dtype"])])
    if config.get("gpu_memory_utilization"):
        command.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])
    if config.get("max_model_len"):
        command.extend(["--max-model-len", str(config["max_model_len"])])
    if config.get("max_num_seqs"):
        command.extend(["--max-num-seqs", str(config["max_num_seqs"])])
    if config.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if config.get("limit_mm_per_prompt"):
        command.extend(["--limit-mm-per-prompt", str(config["limit_mm_per_prompt"])])
    extra_args = config.get("extra_args") or []
    command.extend(str(item) for item in extra_args)
    return command


def start_vllm_server(config: dict[str, Any]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    cache_root = config.get("cache_root") or env.get("JIEZI_CACHE_ROOT")
    env.update(build_cache_env(cache_root))
    cuda_visible_devices = config.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    python_exec = config.get("python_exec")
    if python_exec:
        env_root = os.path.dirname(os.path.dirname(str(python_exec)))
        env_lib = os.path.join(env_root, "lib")
        env["LD_LIBRARY_PATH"] = f"{env_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    process = subprocess.Popen(
        build_vllm_command(config),
        # Do not pipe server logs without draining them continuously.
        # A full pipe buffer can block the vLLM process and stall inference.
        stdout=None,
        stderr=None,
        env=env,
    )
    return process


def wait_for_server(config: dict[str, Any], timeout: int = 600, process: subprocess.Popen[str] | None = None) -> None:
    client = OpenAICompatibleClient.from_config(config)
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {process.returncode}. "
                "Check vLLM console logs for details."
            )
        try:
            client.health()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(5)
    raise TimeoutError(f"vLLM server did not become healthy in {timeout}s: {last_error}")
