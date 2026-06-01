from __future__ import annotations

from typing import Any

JUDGE_DIMENSIONS = [
    "fact_alignment",
    "key_point_coverage",
    "diachronic_logic",
]

LEGACY_TO_NEW = {
    "fact_accuracy": "fact_alignment",
    "information_completeness": "key_point_coverage",
    "clarity_coherence": "diachronic_logic",
}

NEW_TO_LEGACY = {
    "fact_alignment": "fact_accuracy",
    "key_point_coverage": "information_completeness",
    "diachronic_logic": "clarity_coherence",
}


def _clamp_score(value: float, lo: float = 0.0, hi: float = 4.0) -> float:
    return max(lo, min(hi, value))


def _extract_score(data: dict[str, Any], key: str, legacy_mode: bool) -> float:
    names = [key]
    legacy_name = NEW_TO_LEGACY.get(key)
    if legacy_name:
        names.append(legacy_name)

    raw_score: float | None = None
    for name in names:
        value = data.get(name)
        if isinstance(value, dict):
            score = value.get("score")
        else:
            score = value
        try:
            raw_score = float(score)
            break
        except (TypeError, ValueError):
            continue

    if raw_score is None:
        return 0.0

    if legacy_mode:
        # Legacy outputs are 0-5; scale to 0-4.
        raw_score = _clamp_score(max(0.0, min(5.0, raw_score)) * (4.0 / 5.0))
        return raw_score
    return _clamp_score(raw_score)


def normalize_judge_scores(data: dict[str, Any]) -> dict[str, float]:
    """Normalize judge score payloads to the current dimension names and 0-4 scale.

    Keeps backward compatibility with legacy dimension names and 0-5 scale.
    """
    legacy_mode = any(key in data for key in LEGACY_TO_NEW)

    normalized: dict[str, float] = {}
    for field in JUDGE_DIMENSIONS:
        normalized[field] = _extract_score(data, field, legacy_mode)

    overall = data.get("overall_score")
    try:
        overall_value = float(overall)
        if legacy_mode:
            overall_value = max(0.0, min(5.0, overall_value)) * (4.0 / 5.0)
        normalized["overall_score"] = _clamp_score(overall_value)
    except (TypeError, ValueError):
        normalized["overall_score"] = _clamp_score(
            sum(normalized[field] for field in JUDGE_DIMENSIONS) / len(JUDGE_DIMENSIONS)
        )
    return normalized
