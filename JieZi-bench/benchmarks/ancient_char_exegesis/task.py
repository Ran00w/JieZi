from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.ancient_char_exegesis.extractors import extract_reference_label
from benchmarks.ancient_char_exegesis.prompts import load_prompt_bundle


@dataclass
class Sample:
    sample_id: str
    slug: str
    character: str
    group: str
    difficulty: str
    glyph_category: str
    image_path: Path
    prompt_name: str
    system_prompt: str
    few_shot_messages: list[dict[str, Any]]
    user_prompt: str
    ground_truth: dict[str, Any]


def iter_samples(
    *,
    repo_root: Path,
    data_root: Path,
    prompt_name: str,
    limit: int = 0,
    slug_filters: list[str] | None = None,
    group_filters: list[str] | None = None,
) -> list[Sample]:
    slug_filter_set = set(slug_filters or [])
    group_filter_set = set(group_filters or [])
    samples: list[Sample] = []

    for entry_path in sorted(data_root.glob("*/entry.json")):
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        slug = entry_path.parent.name
        group = str(entry.get("group", ""))
        if slug_filter_set and slug not in slug_filter_set:
            continue
        if group_filter_set and group not in group_filter_set:
            continue

        for image_record in entry.get("image_records", []):
            image_rel_path = Path(image_record["local_path"])
            image_path = entry_path.parent / image_rel_path
            gt_path = data_root / slug / image_rel_path.with_suffix(".json")
            if not gt_path.exists():
                raise FileNotFoundError(f"ground-truth json missing: {gt_path}")
            gt_json = json.loads(gt_path.read_text(encoding="utf-8"))
            system_prompt, few_shot_messages, user_prompt = load_prompt_bundle(prompt_name, entry, image_record)
            sample = Sample(
                sample_id=f"{slug}__{image_record['category']}__{image_path.stem}",
                slug=slug,
                character=str(entry["character"]),
                group=group,
                difficulty=str(entry.get("difficulty", "")),
                glyph_category=str(image_record.get("category", "")),
                image_path=image_path,
                prompt_name=prompt_name,
                system_prompt=system_prompt,
                few_shot_messages=few_shot_messages,
                user_prompt=user_prompt,
                ground_truth=extract_reference_label(gt_json, modern_headword=str(entry["character"])),
            )
            samples.append(sample)
            if limit > 0 and len(samples) >= limit:
                return samples
    return samples
