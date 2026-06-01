from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def as_text(value: Any) -> str:
    """Coerce a value to a stripped string, treating None as empty."""
    if value is None:
        return ""
    return str(value).strip()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a file path."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[warn] invalid json skipped: {path} ({exc})", flush=True)
        return {}


def dump_json(path: Path, data: Any) -> None:
    """Write a JSON object to a file path, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] invalid jsonl line skipped: {path}:{idx} ({exc})", flush=True)
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append a single JSON object as a new line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean of values, or 0.0 if the list is empty."""
    return float(mean(values)) if values else 0.0
