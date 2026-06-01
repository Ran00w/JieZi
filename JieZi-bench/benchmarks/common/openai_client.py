from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from urllib3.util import Timeout as Urllib3Timeout


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 300,
        max_retries: int = 0,
        retry_backoff_base: float = 2.0,
        retry_backoff_max: float = 30.0,
        retry_http_statuses: set[int] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_base = max(0.0, float(retry_backoff_base))
        self.retry_backoff_max = max(self.retry_backoff_base, float(retry_backoff_max))
        self.retry_http_statuses = retry_http_statuses or {408, 409, 429, 500, 502, 503, 504}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleClient":
        api_key = resolve_api_key(config)
        return cls(
            base_url=str(config["base_url"]),
            model=str(config["model"]),
            api_key=api_key,
            timeout=int(config.get("timeout", 300)),
            max_retries=int(config.get("max_retries", 0)),
            retry_backoff_base=float(config.get("retry_backoff_base", 2.0)),
            retry_backoff_max=float(config.get("retry_backoff_max", 30.0)),
            retry_http_statuses=set(int(code) for code in config.get("retry_http_statuses", [408, 409, 429, 500, 502, 503, 504])),
        )

    def _sleep_before_retry(self, attempt_index: int) -> None:
        if self.retry_backoff_base <= 0:
            return
        delay = min(self.retry_backoff_max, self.retry_backoff_base * (2 ** (attempt_index - 1)))
        jitter = random.uniform(0.8, 1.2)
        time.sleep(delay * jitter)

    def _request_with_retry(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, Any] | None,
        timeout: float,
    ) -> requests.Response:
        attempt = 0
        request_timeout = Urllib3Timeout(
            total=float(timeout),
            connect=min(30.0, float(timeout)),
            read=float(timeout),
        )
        parsed_url = urlparse(url)
        is_local_target = parsed_url.hostname in {"127.0.0.1", "localhost"}
        proxies = {"http": None, "https": None} if is_local_target else None
        while True:
            attempt += 1
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=request_timeout,
                    proxies=proxies,
                )
                if response.status_code in self.retry_http_statuses and attempt <= self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                is_retryable_error = isinstance(
                    exc,
                    (requests.Timeout, requests.ConnectionError, requests.exceptions.SSLError),
                )
                if not is_retryable_error or attempt > self.max_retries:
                    raise
                self._sleep_before_retry(attempt)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat_completion(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_path: Path | None = None,
        few_shot_messages: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_content: list[dict[str, Any]] | str
        if image_path is None:
            user_content = user_text
        else:
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": encode_image_as_data_url(image_path)}},
            ]

        request_body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                *(few_shot_messages or []),
                {"role": "user", "content": user_content},
            ],
        }
        if response_format is not None:
            request_body["response_format"] = response_format
        if extra_body:
            request_body.update(extra_body)

        headers = self._build_headers()

        try:
            response = self._request_with_retry(
                method="POST",
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                json_body=request_body,
                timeout=self.timeout,
            )
        except requests.HTTPError as exc:
            if (
                response_format is not None
                and _is_response_format_unsupported(exc)
            ):
                fallback_body = dict(request_body)
                fallback_body.pop("response_format", None)
                response = self._request_with_retry(
                    method="POST",
                    url=f"{self.base_url}/chat/completions",
                    headers=headers,
                    json_body=fallback_body,
                    timeout=self.timeout,
                )
                request_body = fallback_body
            else:
                raise
        return request_body, response.json()

    def _parse_embeddings_response(self, response_json: dict[str, Any]) -> list[list[float]]:
        data = response_json.get("data") or []
        if not isinstance(data, list) or not data:
            raise ValueError("empty embeddings response")
        ordered = sorted(
            [item for item in data if isinstance(item, dict)],
            key=lambda item: int(item.get("index", 0)),
        )
        vectors: list[list[float]] = []
        for item in ordered:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("invalid embedding format in response")
            vectors.append([float(x) for x in emb])
        return vectors

    def create_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        clean_inputs = [str(text) for text in texts]
        if not clean_inputs:
            return []
        request_body = {"model": self.model, "input": clean_inputs}
        response = self._request_with_retry(
            method="POST",
            url=f"{self.base_url}/embeddings",
            headers=self._build_headers(),
            json_body=request_body,
            timeout=self.timeout,
        )
        vectors = self._parse_embeddings_response(response.json())
        if len(vectors) != len(clean_inputs):
            raise RuntimeError(f"embedding output mismatch: expected {len(clean_inputs)} vectors, got {len(vectors)}")
        return vectors

    def create_image_embedding(self, image_path: Path) -> list[float]:
        data_url = encode_image_as_data_url(image_path)
        candidate_inputs: list[Any] = [
            [{"type": "input_image", "image_url": data_url}],
            [{"type": "input_image", "image_url": {"url": data_url}}],
            [{"type": "image_url", "image_url": data_url}],
            [{"type": "image_url", "image_url": {"url": data_url}}],
            [data_url],
        ]
        last_exc: Exception | None = None
        for candidate_input in candidate_inputs:
            try:
                request_body = {"model": self.model, "input": candidate_input}
                response = self._request_with_retry(
                    method="POST",
                    url=f"{self.base_url}/embeddings",
                    headers=self._build_headers(),
                    json_body=request_body,
                    timeout=self.timeout,
                )
                vectors = self._parse_embeddings_response(response.json())
                if not vectors:
                    raise ValueError("no embedding vector returned")
                return vectors[0]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        raise RuntimeError(f"failed to create image embedding after trying multiple payload formats: {last_exc}")

    def health(self) -> dict[str, Any]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._request_with_retry(
            method="GET",
            url=f"{self.base_url}/models",
            headers=headers,
            json_body=None,
            timeout=min(self.timeout, 30),
        )
        return response.json()


def extract_message_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("empty choices in model response")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(texts).strip()
    raise ValueError(f"unsupported message content format: {type(content).__name__}")


def _is_response_format_unsupported(exc: requests.HTTPError) -> bool:
    response = exc.response
    if response is None:
        return False
    if response.status_code != 400:
        return False
    body = response.text.lower()
    return (
        "response_format" in body
        and ("json_object" in body or "not supported" in body or "invalidparameter" in body)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("model output does not contain a JSON object")
        return json.loads(text[start : end + 1])


def resolve_api_key(config: dict[str, Any]) -> str | None:
    direct_key = config.get("api_key")
    if direct_key:
        return str(direct_key)

    api_key_env = config.get("api_key_env")
    if not api_key_env:
        return None

    api_key_env = str(api_key_env)
    env_value = os.environ.get(api_key_env)
    if env_value:
        return env_value

    if api_key_env.startswith("sk-") or api_key_env.startswith("Bearer ") or not api_key_env.replace("_", "").isalnum():
        return api_key_env

    return None


def redact_sensitive_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(config)
    for key in ("api_key", "api_key_env"):
        value = redacted.get(key)
        if not value:
            continue
        value_str = str(value)
        if key == "api_key_env" and value_str.isupper() and "_" in value_str and not value_str.startswith("sk-"):
            continue
        redacted[key] = mask_secret(value_str)
    return redacted


def mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}***{secret[-4:]}"
