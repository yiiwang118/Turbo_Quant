# Scorer functions adapted from autokv/evaluation/benchmarks/longbench/calculate_metrics.py
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Callable


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# ── Per-task metric functions ─────────────────────────────────────────────────

def _token_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    common   = Counter(pred_tokens) & Counter(gt_tokens)
    n_same   = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall    = n_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return _token_f1(_normalize(prediction).split(), _normalize(ground_truth).split())


def rouge_score(prediction: str, ground_truth: str, **_: Any) -> float:
    from rouge import Rouge
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    import jieba
    pred = " ".join(jieba.cut(prediction, cut_all=False))
    gt   = " ".join(jieba.cut(ground_truth, cut_all=False))
    return rouge_score(pred, gt)


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes: list[str] = kwargs.get("all_classes") or []
    matches = [c for c in all_classes if c in prediction]
    # Remove over-matching: if a match is a substring of the ground truth but not equal, drop it
    matches = [m for m in matches if not (m in ground_truth and m != ground_truth)]
    return (1.0 / len(matches)) if ground_truth in matches else 0.0


def count_score(prediction: str, ground_truth: str, **_: Any) -> float:
    nums = re.findall(r"\d+", prediction)
    hits = sum(1 for n in nums if n == str(ground_truth))
    return 0.0 if not nums else hits / len(nums)


def retrieval_score(prediction: str, ground_truth: str, **_: Any) -> float:
    gt_ids = re.findall(r"Paragraph (\d+)", ground_truth)
    if not gt_ids:
        return 0.0
    nums = re.findall(r"\d+", prediction)
    hits = sum(1 for n in nums if n == gt_ids[0])
    return 0.0 if not nums else hits / len(nums)


def retrieval_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    gt_ids = re.findall(r"段落(\d+)", ground_truth)
    if not gt_ids:
        return 0.0
    nums = re.findall(r"\d+", prediction)
    hits = sum(1 for n in nums if n == gt_ids[0])
    return 0.0 if not nums else hits / len(nums)


def code_sim_score(prediction: str, ground_truth: str, **_: Any) -> float:
    from fuzzywuzzy import fuzz
    for line in prediction.lstrip("\n").split("\n"):
        if not any(c in line for c in ("`", "#", "//")):
            return fuzz.ratio(line, ground_truth) / 100
    return 0.0


# ── Task → metric mapping ─────────────────────────────────────────────────────

DATASET2METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa":         qa_f1_score,
    "qasper":              qa_f1_score,
    "multifieldqa_en":     qa_f1_score,
    "hotpotqa":            qa_f1_score,
    "2wikimqa":            qa_f1_score,
    "musique":             qa_f1_score,
    "triviaqa":            qa_f1_score,
    "gov_report":          rouge_score,
    "qmsum":               rouge_score,
    "multi_news":          rouge_score,
    "samsum":              rouge_score,
    "dureader":            rouge_zh_score,
    "vcsum":               rouge_zh_score,
    "multifieldqa_zh":     rouge_zh_score,
    "trec":                classification_score,
    "lsht":                classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "passage_count":        count_score,
    "lcc":                  code_sim_score,
    "repobench-p":          code_sim_score,
}


# ── Aggregation ───────────────────────────────────────────────────────────────

def score_predictions(
    task: str,
    predictions: list[str],
    answers: list[list[str]],
    all_classes: Any,
) -> float:
    """Return mean score (0–100) over all prediction / ground-truth pairs."""
    metric = DATASET2METRIC.get(task, qa_f1_score)
    classes = all_classes or []
    total = 0.0
    for pred, gts in zip(predictions, answers):
        if task in ("trec", "triviaqa", "samsum", "lsht"):
            pred = pred.lstrip().split("\n")[0]
        total += max(metric(pred.lstrip(), gt, all_classes=classes) for gt in gts)
    return round(100 * total / max(len(predictions), 1), 2)
