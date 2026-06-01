from __future__ import annotations

from typing import Any

from benchmarks.ancient_char_exegesis.constants import (
    ALLOWED_EVOLUTION_TYPES,
    ALLOWED_FUNCTIONS,
    ALLOWED_GLYPHS,
    ALLOWED_IDS,
)


GLYPH_ALIASES = {
    "甲骨": "甲骨文",
    "金": "金文",
    "古": "古文",
    "篆": "篆书",
    "小篆": "篆书",
    "大篆": "篆书",
    "隶": "隶书",
    "楷": "楷书",
    "草": "草书",
}

LIUSHU_ALIASES = {
    "象形": "象形字",
    "指事": "指事字",
    "会意": "会意字",
    "形声": "形声字",
    "转注": "转注字",
    "假借": "假借字",
}

STRUCTURE_ALIASES = {
    "左右结构": "⿰",
    "左中右结构": "⿲",
    "上下结构": "⿱",
    "上中下结构": "⿳",
    "全包围结构": "⿴",
    "上三包围结构": "⿵",
    "下三包围结构": "⿶",
    "左三包围结构": "⿷",
    "左上包围结构": "⿸",
    "右上包围结构": "⿹",
    "左下包围结构": "⿺",
    "镶嵌结构": "⿻",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", "\n").split()).strip()


def normalize_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    deduped: list[str] = []
    for item in values:
        item_text = normalize_text(item)
        if item_text and item_text not in deduped:
            deduped.append(item_text)
    return deduped


def normalize_liushu(value: Any) -> list[str]:
    items = normalize_list(value)
    normalized: list[str] = []
    for item in items:
        candidate = LIUSHU_ALIASES.get(item, item)
        if candidate in LIUSHU_ALIASES:
            candidate = LIUSHU_ALIASES[candidate]
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def normalize_glyph(value: Any) -> str:
    glyph = normalize_text(value)
    if glyph in ALLOWED_GLYPHS:
        return glyph
    return GLYPH_ALIASES.get(glyph, "")


def normalize_structure(value: Any) -> str:
    structure = normalize_text(value)
    if structure in ALLOWED_IDS:
        return structure
    return STRUCTURE_ALIASES.get(structure, "")


def normalize_component_function(value: Any) -> list[str]:
    items = normalize_list(value)
    valid = [item for item in items if item in ALLOWED_FUNCTIONS]
    return valid or ["符号"]


def normalize_components(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_data in value.items():
        name = normalize_text(raw_name)
        if not name or not isinstance(raw_data, dict):
            continue
        evolution_type = normalize_text(raw_data.get("演变类型", "保留"))
        if evolution_type not in ALLOWED_EVOLUTION_TYPES:
            evolution_type = "保留"
        normalized[name] = {
            "功能": normalize_component_function(raw_data.get("功能", "符号")),
            "演变类型": evolution_type,
            "解释": normalize_text(raw_data.get("解释", "")),
        }
    return normalized


def normalize_prediction(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "现代字典字头": normalize_text(raw.get("现代字典字头", "")),
        "字形": normalize_glyph(raw.get("字形", "")),
        "造字法": normalize_liushu(raw.get("造字法", [])),
        "结构": normalize_structure(raw.get("结构", "")),
        "特殊结构": normalize_text(raw.get("特殊结构", "")),
        "构件": normalize_components(raw.get("构件", {})),
        "本义": normalize_text(raw.get("本义", "")),
        "历代字形演变": normalize_text(raw.get("历代字形演变", "")),
    }


def extract_reference_label(reference_json: dict[str, Any], modern_headword: str) -> dict[str, Any]:
    normalized = normalize_prediction(reference_json)
    normalized["现代字典字头"] = modern_headword
    return normalized
