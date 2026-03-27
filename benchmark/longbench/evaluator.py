"""Reusable LongBench evaluation workflow."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.kvpress.pipeline import KVPressTextGenerationRunner
from benchmark.longbench.metrics import score_predictions
from benchmark.longbench.presses import TurboQuantPress

LONGBENCH_DATASET = "Xnhyacinth/LongBench"
DEFAULT_TASKS = ["qasper", "triviaqa", "hotpotqa"]
DEFAULT_BITS = [0, 8, 4, 2, 1]
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@dataclass
class LongBenchEvalConfig:
    model: str = DEFAULT_MODEL
    tasks: list[str] = field(default_factory=lambda: DEFAULT_TASKS.copy())
    bits: list[int] = field(default_factory=lambda: DEFAULT_BITS.copy())
    fraction: float = 0.1
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    seed: int = 42

    def validate(self) -> None:
        if not self.tasks:
            raise ValueError("`tasks` cannot be empty.")
        if not self.bits:
            raise ValueError("`bits` cannot be empty.")
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError("`fraction` must be in (0, 1].")


def _set_deterministic_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_runner(model_name: str) -> tuple[KVPressTextGenerationRunner, str]:
    device = "auto" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()

    return KVPressTextGenerationRunner(model=model, tokenizer=tokenizer), device


def _make_press(bits: int, seed: int) -> Optional[TurboQuantPress]:
    if bits == 0:
        return None
    return TurboQuantPress(b_key=bits, b_value=bits, seed=seed)


def run_task(
    runner: KVPressTextGenerationRunner,
    task: str,
    bits: int,
    fraction: float,
    max_new_tokens: Optional[int],
    max_context_length: Optional[int],
    seed: int,
) -> float:
    df = load_dataset(
        LONGBENCH_DATASET,
        data_dir=task,
        split="test",
        trust_remote_code=True,
    ).to_pandas()

    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    press = _make_press(bits, seed)
    predictions: list[str] = []

    for context, df_group in tqdm(
        df.groupby("context", sort=False),
        desc=f"{task} b={bits}",
        leave=False,
    ):
        questions = df_group["question"].tolist()
        answer_prefix = df_group["answer_prefix"].iloc[0]
        local_max_new_tokens = max_new_tokens or int(df_group["max_new_tokens"].iloc[0])

        output = runner(
            context=context,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=local_max_new_tokens,
            max_context_length=max_context_length,
        )
        predictions.extend(output["answers"])  # type: ignore[arg-type]

    answers = df["answers"].tolist()
    all_classes = df["all_classes"].iloc[0] if "all_classes" in df.columns else []
    return score_predictions(task, predictions, answers, all_classes)


def evaluate_longbench(
    runner: KVPressTextGenerationRunner,
    config: LongBenchEvalConfig,
) -> dict[str, dict[str, float]]:
    config.validate()
    _set_deterministic_seeds(config.seed)

    results: dict[str, dict[str, float]] = {}
    for task in config.tasks:
        task_results: dict[str, float] = {}
        for bits in config.bits:
            label = f"b={bits}" if bits > 0 else "baseline"
            task_results[label] = run_task(
                runner=runner,
                task=task,
                bits=bits,
                fraction=config.fraction,
                max_new_tokens=config.max_new_tokens,
                max_context_length=config.max_context_length,
                seed=config.seed,
            )
        results[task] = task_results
    return results
