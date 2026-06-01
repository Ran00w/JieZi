#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import uvicorn


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    max_tokens: int = 512


def decode_data_url_to_pil(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only data URL images are supported")
    _, payload = url.split(",", 1)
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    out_messages: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        norm_content: list[dict[str, Any]] = []
        if isinstance(content, str):
            norm_content.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    norm_content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, str):
                        url = image_url
                    else:
                        url = str(image_url.get("url", ""))
                    if not url:
                        continue
                    images.append(decode_data_url_to_pil(url))
                    norm_content.append({"type": "image", "image": "<image>"})
        else:
            norm_content.append({"type": "text", "text": str(content)})
        out_messages.append({"role": role, "content": norm_content})
    return out_messages, images


def coerce_answer_to_json_content(answer: str) -> str:
    text = answer.strip()
    if text:
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1 and right > left:
            candidate = text[left : right + 1]
            try:
                obj = json.loads(candidate)
                return json.dumps(obj, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                pass
    fallback = {
        "现代字典字头": "",
        "字形": "",
        "造字法": [],
        "结构": "",
        "特殊结构": "",
        "构件": {},
        "本义": "",
        "历代字形演变": text,
    }
    return json.dumps(fallback, ensure_ascii=False)


def build_app(model_path: str, served_model_name: str, device: str, max_new_tokens_cap: int) -> FastAPI:
    app = FastAPI()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    lock = threading.Lock()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionsRequest) -> dict[str, Any]:
        if req.model != served_model_name:
            raise HTTPException(status_code=400, detail=f"unsupported model: {req.model}")
        try:
            chat_messages, images = normalize_messages(req.messages)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid request messages: {exc}") from exc

        prompt = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        proc_inputs = processor(text=[prompt], images=images if images else None, return_tensors="pt")
        proc_inputs = {k: v.to(device) for k, v in proc_inputs.items()}

        max_new_tokens = max(1, min(int(req.max_tokens), max_new_tokens_cap))
        temperature = float(req.temperature)
        do_sample = temperature > 0.0

        with lock:
            with torch.inference_mode():
                outputs = model.generate(
                    **proc_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                )

        input_len = int(proc_inputs["input_ids"].shape[1])
        answer_ids = outputs[:, input_len:]
        answer = processor.batch_decode(answer_ids, skip_special_tokens=True)[0]
        answer = coerce_answer_to_json_content(answer)

        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
            "object": "chat.completion",
            "created": now,
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": int(answer_ids.shape[1]),
                "total_tokens": int(input_len + answer_ids.shape[1]),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible local server for TongGu-VL-2B-Instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="TongGu-VL-2B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens-cap", type=int, default=256)
    args = parser.parse_args()

    app = build_app(args.model_path, args.served_model_name, args.device, args.max_new_tokens_cap)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
