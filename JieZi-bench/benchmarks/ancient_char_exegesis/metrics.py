from __future__ import annotations

import os
import unicodedata
from collections import defaultdict
from statistics import mean
from typing import Any

from benchmarks.common.runtime_env import setup_cache_environment

RADICAL_MERGE_TABLE: dict[str, list[str]] = {
    "人": ["人", "亻", "人字旁", "单人旁", "单立人"],
    "手": ["手", "扌", "龵", "提手旁"],
    "心": ["心", "忄", "㣺", "竖心旁", "心字底"],
    "言": ["言", "讠", "言字旁"],
    "食": ["食", "饣", "食字旁"],
    "金": ["金", "钅", "釒", "金字旁"],
    "示": ["示", "礻", "示字旁", "示补旁"],
    "糸": ["糸", "纟", "糹", "绞丝旁"],
    "水": ["水", "氵", "氺", "三点水"],
    "火": ["火", "灬", "四点底"],
    "犬": ["犬", "犭", "反犬旁"],
    "衣": ["衣", "衤", "衣字旁", "衣补旁"],
    "玉": ["玉", "王", "王字旁", "斜玉旁"],
    "月": ["月", "月字旁"],
    "肉": ["肉", "⺼", "肉月旁"],
    "邑": ["邑", "阝(右)", "右耳旁"],
    "阜": ["阜", "阝(左)", "左耳旁"],
    "艸": ["艸", "艹", "草字头"],
    "竹": ["竹", "⺮", "竹字头"],
    "网": ["网", "罒", "四字头"],
    "刀": ["刀", "刂", "立刀旁"],
    "力": ["力", "力字旁"],
    "口": ["口", "口字旁"],
    "木": ["木", "木字旁"],
    "土": ["土", "土字旁"],
    "女": ["女", "女字旁"],
    "子": ["子", "子字旁"],
    "目": ["目", "目字旁"],
    "页": ["頁", "页", "页字旁"],
    "贝": ["貝", "贝", "贝字旁"],
}

RADICAL_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical for canonical, aliases in RADICAL_MERGE_TABLE.items() for alias in aliases
}

GLYPH_EQUIVALENCE_TO_CANONICAL: dict[str, str] = {
    "楷书": "楷书",
    "简体": "楷书",
}


def accuracy(pred: str, gold: str) -> float:
    return 1.0 if pred and pred == gold else 0.0


def normalize_glyph_category(label: str) -> str:
    normalized = (label or "").strip()
    return GLYPH_EQUIVALENCE_TO_CANONICAL.get(normalized, normalized)


def glyph_accuracy(pred: str, gold: str) -> float:
    pred_norm = normalize_glyph_category(pred)
    gold_norm = normalize_glyph_category(gold)
    return 1.0 if pred_norm and pred_norm == gold_norm else 0.0


def set_exact_match(pred: list[str], gold: list[str]) -> float:
    return 1.0 if set(pred) == set(gold) else 0.0


def set_recall_credit(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    return len(set(pred) & set(gold)) / len(set(gold))


def build_component_metrics(
    *,
    gold_total: float,
    pred_total: float,
    match_total: float,
    function_correct: float,
    evolution_correct: float,
    explanation_sum: float,
) -> dict[str, Any]:
    """Build component-level metric fields from aggregate counts.

    This centralises the logic so that both the online scorer
    (run_eval.py::score_sample) and the offline recomputer
    (recompute_failure_denominator_metrics.py) share one definition.
    """
    precision = (match_total / pred_total) if pred_total > 0 else 0.0
    recall = (match_total / gold_total) if gold_total > 0 else 0.0
    f1 = f1_from_pr(precision, recall)

    function_precision = (function_correct / pred_total) if pred_total > 0 else 0.0
    function_recall = (function_correct / gold_total) if gold_total > 0 else 0.0
    function_f1 = f1_from_pr(function_precision, function_recall)

    evolution_precision = (evolution_correct / pred_total) if pred_total > 0 else 0.0
    evolution_recall = (evolution_correct / gold_total) if gold_total > 0 else 0.0
    evolution_f1 = f1_from_pr(evolution_precision, evolution_recall)

    explanation_precision = (explanation_sum / pred_total) if pred_total > 0 else 0.0
    explanation_recall = (explanation_sum / gold_total) if gold_total > 0 else 0.0
    explanation_f1 = f1_from_pr(explanation_precision, explanation_recall)

    return {
        "construction_precision": precision,
        "construction_recall": recall,
        "construction_f1": f1,
        "component_match_recall": recall,
        "component_function_acc": function_recall,
        "component_function_precision": function_precision,
        "component_function_recall": function_recall,
        "component_function_f1": function_f1,
        "component_evolution_acc": evolution_recall,
        "component_evolution_precision": evolution_precision,
        "component_evolution_recall": evolution_recall,
        "component_evolution_f1": evolution_f1,
        "component_explanation_bertscore": explanation_recall,
        "component_explanation_acc": explanation_recall,
        "component_explanation_precision": explanation_precision,
        "component_explanation_recall": explanation_recall,
        "component_explanation_f1": explanation_f1,
        "__agg_component_gt_total": gold_total,
        "__agg_component_pred_total": pred_total,
        "__agg_component_match_total": match_total,
        "__agg_component_function_correct": function_correct,
        "__agg_component_evolution_correct": evolution_correct,
        "__agg_component_explanation_bertscore_sum": explanation_sum,
    }


def _normalize_component_name(name: str) -> str:
    return " ".join(name.replace("\r", "\n").split()).strip()


def _is_single_char_or_radical(name: str) -> bool:
    if not name:
        return False
    if name in RADICAL_ALIAS_TO_CANONICAL:
        return True
    return len(name) == 1


def normalize_component_name_for_matching(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", _normalize_component_name(name))
    if _is_single_char_or_radical(normalized):
        return RADICAL_ALIAS_TO_CANONICAL.get(normalized, normalized)
    return normalized


def text_similarity(prediction: str, reference: str) -> float:
    pred = _normalize_for_anls(prediction or "")
    gold = _normalize_for_anls(reference or "")
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    max_len = max(len(pred), len(gold))
    if max_len == 0:
        return 1.0
    nl = _levenshtein_distance(pred, gold) / max_len
    return max(0.0, 1.0 - nl)


def match_component_names(
    gold_names: list[str],
    pred_names: list[str],
    *,
    anls_threshold: float = 0.8,
) -> dict[str, str]:
    """Match GT component names to prediction component names using exact + fuzzy one-to-one matching."""
    gold_norm = {name: normalize_component_name_for_matching(name) for name in gold_names}
    pred_norm = {name: normalize_component_name_for_matching(name) for name in pred_names}

    pred_buckets: dict[str, list[str]] = defaultdict(list)
    for pred_name in pred_names:
        pred_buckets[pred_norm[pred_name]].append(pred_name)

    matches: dict[str, str] = {}
    unmatched_gold: list[str] = []
    for gold_name in gold_names:
        bucket = pred_buckets.get(gold_norm[gold_name], [])
        if bucket:
            matches[gold_name] = bucket.pop(0)
        else:
            unmatched_gold.append(gold_name)

    matched_pred_names = set(matches.values())
    unmatched_pred = [name for name in pred_names if name not in matched_pred_names]

    candidates: dict[str, list[tuple[str, float]]] = {}
    for gold_name in unmatched_gold:
        options: list[tuple[str, float]] = []
        for pred_name in unmatched_pred:
            score = text_similarity(pred_norm[pred_name], gold_norm[gold_name])
            if score >= anls_threshold:
                options.append((pred_name, score))
        if options:
            candidates[gold_name] = sorted(options, key=lambda item: (-item[1], item[0]))

    best_count = -1
    best_score = -1.0
    best_key: tuple[tuple[str, str], ...] = ()
    best_map: dict[str, str] = {}

    def _dfs(index: int, used_preds: set[str], current: dict[str, str], count: int, score: float) -> None:
        nonlocal best_count, best_score, best_key, best_map
        if index == len(unmatched_gold):
            current_key = tuple(sorted(current.items()))
            if (
                count > best_count
                or (count == best_count and score > best_score + 1e-12)
                or (count == best_count and abs(score - best_score) <= 1e-12 and (not best_key or current_key < best_key))
            ):
                best_count = count
                best_score = score
                best_key = current_key
                best_map = dict(current)
            return

        gold_name = unmatched_gold[index]
        _dfs(index + 1, used_preds, current, count, score)
        for pred_name, edge_score in candidates.get(gold_name, []):
            if pred_name in used_preds:
                continue
            used_preds.add(pred_name)
            current[gold_name] = pred_name
            _dfs(index + 1, used_preds, current, count + 1, score + edge_score)
            del current[gold_name]
            used_preds.remove(pred_name)

    _dfs(0, set(), {}, 0, 0.0)
    matches.update(best_map)
    return matches


class BertScorer:
    def __init__(
        self,
        model_type: str,
        lang: str = "zh",
        num_layers: int | None = None,
        all_layers: bool = True,
        layer_agg: str = "mean",
    ) -> None:
        self.model_type = model_type
        self.lang = lang
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.layer_agg = str(layer_agg or "last").strip().lower()
        if self.layer_agg not in {"last", "first", "mean", "max", "min"}:
            raise ValueError(f"Unsupported layer_agg: {layer_agg}")
        self._scorer = None

    def _slice_front_layers(self, values):
        if self.num_layers is None:
            return values
        limit = int(self.num_layers)
        if limit <= 0:
            return values
        if int(values.shape[0]) <= limit:
            return values
        return values[:limit]

    def _resolve_model_path(self) -> str:
        if os.path.isdir(self.model_type):
            return self.model_type
        if not self.model_type.startswith("AI-ModelScope/"):
            return self.model_type
        try:
            from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
        except ModuleNotFoundError:
            return self.model_type
        setup_cache_environment(os.environ.get("JIEZI_CACHE_ROOT"))
        cache_dir = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
        try:
            return str(snapshot_download(self.model_type, cache_dir=cache_dir))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download ModelScope model '{self.model_type}'. "
                "If network is unavailable, pre-download the model and set metrics.bertscore.model_type "
                "to the local directory path."
            ) from exc

    def _ensure(self) -> None:
        if self._scorer is not None:
            return
        try:
            from bert_score import BERTScorer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("BERTScore requires bert_score and transformers. Please install them first.") from exc
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        model_path = self._resolve_model_path()
        kwargs = {
            "model_type": model_path,
            "lang": self.lang,
            "rescale_with_baseline": False,
            "all_layers": self.all_layers,
        }
        if self.num_layers is not None:
            kwargs["num_layers"] = self.num_layers
        self._scorer = BERTScorer(**kwargs)
        tokenizer = getattr(self._scorer, "_tokenizer", None)
        if tokenizer is not None:
            model_max_length = getattr(tokenizer, "model_max_length", None)
            # Some tokenizers expose an effectively "infinite" sentinel value here.
            # Fast tokenizers may overflow when this value is forwarded to truncation.
            if isinstance(model_max_length, int) and model_max_length > 1_000_000:
                tokenizer.model_max_length = 8192

    def score(self, prediction: str, reference: str) -> float:
        if not prediction or not reference:
            return 0.0
        self._ensure()
        _, _, f1 = self._scorer.score([prediction], [reference])
        if not self.all_layers:
            return float(f1[0].item())
        return self._aggregate_layers(f1, batch_size=1)

    def score_batch(self, predictions: list[str], references: list[str], batch_size: int = 32) -> list[float]:
        self._ensure()
        all_scores: list[float] = []
        for start in range(0, len(predictions), batch_size):
            end = start + batch_size
            pred_batch = predictions[start:end]
            ref_batch = references[start:end]
            _, _, f1 = self._scorer.score(pred_batch, ref_batch, batch_size=batch_size)
            if self.all_layers:
                batch_scores = self._aggregate_layer_batch(f1, batch=len(pred_batch))
            else:
                batch_scores = [float(v.item()) for v in f1]
            all_scores.extend(batch_scores)
        return all_scores

    def _aggregate_layers(self, f1: Any, batch_size: int) -> float:
        if f1.ndim == 1:
            values = f1
        elif f1.ndim == 2:
            if int(f1.shape[0]) == 1:
                values = f1[0, :]
            elif int(f1.shape[1]) == 1:
                values = f1[:, 0]
            else:
                values = f1[:, 0]
        else:
            raise RuntimeError(f"Unexpected all-layer BERTScore tensor shape: {tuple(f1.shape)}")
        values = self._slice_front_layers(values)
        return self._reduce_layers(values)

    def _aggregate_layer_batch(self, f1: Any, batch: int) -> list[float]:
        if f1.ndim != 2:
            raise RuntimeError(f"expected all-layer score tensor with ndim=2, got shape={tuple(f1.shape)}")
        if int(f1.shape[0]) == batch:
            matrix = f1
        elif int(f1.shape[1]) == batch:
            matrix = f1.transpose(0, 1)
        elif batch == 1:
            matrix = f1.reshape(1, -1)
        else:
            raise RuntimeError(f"unexpected all-layer score shape={tuple(f1.shape)} for batch={batch}")
        matrix = self._slice_front_layers(matrix.transpose(0, 1)).transpose(0, 1)
        return [self._reduce_layers(matrix[i, :]) for i in range(int(matrix.shape[0]))]

    def _reduce_layers(self, values: Any) -> float:
        if self.layer_agg == "first":
            return float(values[0].item())
        if self.layer_agg == "last":
            return float(values[-1].item())
        if self.layer_agg == "mean":
            return float(values.mean().item())
        if self.layer_agg == "max":
            return float(values.max().item())
        if self.layer_agg == "min":
            return float(values.min().item())
        raise RuntimeError(f"Unexpected layer_agg: {self.layer_agg}")


def bertscore_score_or_none_when_both_empty(
    prediction: str, reference: str, bert_scorer: BertScorer | None
) -> float | None:
    pred = (prediction or "").strip()
    gold = (reference or "").strip()
    if not pred and not gold:
        return None
    if not pred or not gold or bert_scorer is None:
        return 0.0
    return bert_scorer.score(pred, gold)


def _normalize_for_anls(text: str) -> str:
    return "".join(text.split())


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            ))
        prev = curr
    return prev[-1]


def f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def aggregate_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key.startswith("__agg_"):
                continue
            if isinstance(value, (int, float)):
                buckets[key].append(float(value))
    return {key: mean(values) for key, values in sorted(buckets.items()) if values}
