"""LongBench metric functions copied and adapted from autokv."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

import numpy as np
from rouge import Rouge


def calculate_metrics(df) -> dict[str, float]:
    predictions = df["predicted_answer"].tolist()
    answers = df["answers"].tolist()
    dataset = df["task"].tolist()[0]
    all_classes = df["all_classes"].tolist()[0]
    return {"score": scorer(dataset, predictions, answers, all_classes)}


def calculate_metrics_e(df) -> dict[str, float]:
    predictions = df["predicted_answer"].tolist()
    answers = df["answers"].tolist()
    dataset = df["task"].tolist()[0].removesuffix("-e")
    all_classes = df["all_classes"].tolist()[0]
    lengths = df["length"].tolist()
    return scorer_e(dataset, predictions, answers, lengths, all_classes)


def scorer_e(
    dataset: str,
    predictions: list[str],
    answers: list[list[str]],
    lengths: list[int],
    all_classes: list[str],
) -> dict[str, float]:
    scores: dict[str, list[float] | float] = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)  # type: ignore[attr-defined]
        elif length < 8000:
            scores["4-8k"].append(score)  # type: ignore[attr-defined]
        else:
            scores["8k+"].append(score)  # type: ignore[attr-defined]
    return {
        key: round(100 * float(np.mean(values)), 2) if isinstance(values, list) and values else 0.0
        for key, values in scores.items()
    }


def scorer(dataset: str, predictions: list[str], answers: list[list[str]], all_classes: list[str]) -> float:
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip().split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction.lstrip(), ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / max(len(predictions), 1), 2)


def score_predictions(task: str, predictions: list[str], answers: list[list[str]], all_classes: Any) -> float:
    classes = all_classes or []
    return scorer(task, predictions, answers, classes)


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def normalize_zh_answer(text: str) -> str:
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(value: str) -> str:
        return "".join(value.split())

    def remove_punc(value: str) -> str:
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in value if ch not in all_punctuation)

    return white_space_fix(remove_punc(text.lower()))


def count_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def retrieval_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def code_sim_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    from fuzzywuzzy import fuzz

    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            return fuzz.ratio(line, ground_truth) / 100
    return 0.0


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes = kwargs.get("all_classes", [])
    em_match_list = [class_name for class_name in all_classes if class_name in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def rouge_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    import jieba

    prediction_tokens = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth_tokens = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction_tokens, ground_truth_tokens)


def f1_score(prediction: list[str], ground_truth: list[str], **kwargs: Any) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    import jieba

    prediction_tokens = [normalize_zh_answer(token) for token in jieba.cut(prediction, cut_all=False)]
    ground_truth_tokens = [normalize_zh_answer(token) for token in jieba.cut(ground_truth, cut_all=False)]
    prediction_tokens = [token for token in prediction_tokens if token]
    ground_truth_tokens = [token for token in ground_truth_tokens if token]
    return f1_score(prediction_tokens, ground_truth_tokens)


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
