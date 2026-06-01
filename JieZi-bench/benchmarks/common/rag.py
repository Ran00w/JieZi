from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common.openai_client import OpenAICompatibleClient


@dataclass
class RetrievedHeadword:
    char: str
    score: float
    entry_text: str
    best_chunk_text: str
    best_chunk_metadata: dict[str, Any]


class ImageToHeadwordRetriever:
    def __init__(
        self,
        *,
        embedding_client: OpenAICompatibleClient,
        chunk_jsonl_path: Path,
        split_entries_txt_dir: Path,
        cache_dir: Path,
        embedding_batch_size: int = 64,
    ) -> None:
        self.embedding_client = embedding_client
        self.chunk_jsonl_path = chunk_jsonl_path
        self.split_entries_txt_dir = split_entries_txt_dir
        self.cache_dir = cache_dir
        self.embedding_batch_size = max(1, int(embedding_batch_size))

        self.chunk_rows = self._load_chunk_rows(chunk_jsonl_path)
        if not self.chunk_rows:
            raise ValueError(f"no chunks loaded from {chunk_jsonl_path}")
        self.chunk_texts = [str(row.get("text", "")) for row in self.chunk_rows]
        self.chunk_chars = [str((row.get("metadata") or {}).get("char", "")).strip() for row in self.chunk_rows]
        self.chunk_embeddings = self._load_or_build_chunk_embeddings()

    @staticmethod
    def _load_chunk_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                rows.append(row)
        return rows

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return matrix / norms

    def _cache_key(self) -> str:
        stat = self.chunk_jsonl_path.stat()
        key_raw = (
            f"{self.embedding_client.model}|{self.chunk_jsonl_path.resolve()}|"
            f"{stat.st_size}|{int(stat.st_mtime)}"
        )
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self.cache_dir / f"chunk_embeddings_{self._cache_key()}.npz"

    def _load_or_build_chunk_embeddings(self) -> np.ndarray:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path()
        if cache_path.exists():
            loaded = np.load(cache_path)
            matrix = loaded["embeddings"].astype(np.float32)
            if matrix.shape[0] == len(self.chunk_rows):
                return matrix

        vectors: list[np.ndarray] = []
        for start in range(0, len(self.chunk_texts), self.embedding_batch_size):
            batch = self.chunk_texts[start : start + self.embedding_batch_size]
            batch_embeddings = self.embedding_client.create_text_embeddings(batch)
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"embedding batch size mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
                )
            vectors.extend(np.asarray(item, dtype=np.float32) for item in batch_embeddings)

        matrix = np.vstack(vectors).astype(np.float32)
        matrix = self._l2_normalize(matrix)
        np.savez_compressed(cache_path, embeddings=matrix)
        return matrix

    def retrieve(
        self,
        *,
        image_path: Path,
        top_k_headwords: int = 5,
        score_pool_k: int = 128,
    ) -> list[RetrievedHeadword]:
        image_vector = np.asarray(self.embedding_client.create_image_embedding(image_path), dtype=np.float32)
        image_vector = image_vector / max(float(np.linalg.norm(image_vector)), 1e-12)
        similarities = np.matmul(self.chunk_embeddings, image_vector)
        candidate_count = min(max(top_k_headwords * 8, score_pool_k), len(similarities))
        top_chunk_indices = np.argpartition(similarities, -candidate_count)[-candidate_count:]
        top_chunk_indices = top_chunk_indices[np.argsort(similarities[top_chunk_indices])[::-1]]

        best_by_char: dict[str, tuple[float, int]] = {}
        for idx in top_chunk_indices:
            char = self.chunk_chars[idx]
            if not char:
                continue
            score = float(similarities[idx])
            if char not in best_by_char or score > best_by_char[char][0]:
                best_by_char[char] = (score, int(idx))

        ranked_chars = sorted(best_by_char.items(), key=lambda item: item[1][0], reverse=True)[:top_k_headwords]
        results: list[RetrievedHeadword] = []
        for char, (score, idx) in ranked_chars:
            entry_path = self.split_entries_txt_dir / f"{char}.txt"
            if entry_path.exists():
                entry_text = entry_path.read_text(encoding="utf-8")
            else:
                entry_text = self.chunk_texts[idx]
            results.append(
                RetrievedHeadword(
                    char=char,
                    score=score,
                    entry_text=entry_text,
                    best_chunk_text=self.chunk_texts[idx],
                    best_chunk_metadata=dict(self.chunk_rows[idx].get("metadata") or {}),
                )
            )
        return results
