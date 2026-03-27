# LongBench evaluation workflow for TurboQuant
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
DEFAULT_MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_TASKS     = ["qasper", "triviaqa", "hotpotqa"]
DEFAULT_BITS      = [0, 8, 4, 2, 1]   # 0 = no quantisation (baseline)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LongBenchEvalConfig:
    model:              str        = DEFAULT_MODEL
    tasks:              list[str]  = field(default_factory=lambda: list(DEFAULT_TASKS))
    bits:               list[int]  = field(default_factory=lambda: list(DEFAULT_BITS))
    fraction:           float      = 0.1
    max_new_tokens:     Optional[int] = None
    max_context_length: Optional[int] = None
    seed:               int        = 42

    def validate(self) -> None:
        if not self.tasks:       raise ValueError("`tasks` cannot be empty.")
        if not self.bits:        raise ValueError("`bits` cannot be empty.")
        if not 0 < self.fraction <= 1.0:
            raise ValueError(f"`fraction` must be in (0, 1], got {self.fraction}.")


# ── Model loading ─────────────────────────────────────────────────────────────

def build_runner(model_name: str) -> tuple[KVPressTextGenerationRunner, str]:
    """Load model + tokenizer, return (runner, device_str)."""
    device = "auto" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict = {"trust_remote_code": True, "torch_dtype": "auto"}
    if device == "auto":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()

    return KVPressTextGenerationRunner(model=model, tokenizer=tokenizer), device


# ── Per-task evaluation ───────────────────────────────────────────────────────

def run_task(
    runner: KVPressTextGenerationRunner,
    task: str,
    bits: int,
    fraction: float,
    max_new_tokens: Optional[int],
    max_context_length: Optional[int],
    seed: int,
) -> float:
    df = load_dataset(LONGBENCH_DATASET, data_dir=task, split="test",
                      trust_remote_code=True).to_pandas()
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    press = TurboQuantPress(b_key=bits, b_value=bits, seed=seed) if bits > 0 else None
    predictions: list[str] = []

    for context, group in tqdm(df.groupby("context", sort=False),
                               desc=f"{task} b={bits}", leave=False):
        out = runner(
            context=context,
            questions=group["question"].tolist(),
            answer_prefix=group["answer_prefix"].iloc[0],
            press=press,
            max_new_tokens=max_new_tokens or int(group["max_new_tokens"].iloc[0]),
            max_context_length=max_context_length,
        )
        predictions.extend(out["answers"])  # type: ignore[arg-type]

    all_classes = df["all_classes"].iloc[0] if "all_classes" in df.columns else []
    return score_predictions(task, predictions, df["answers"].tolist(), all_classes)


# ── Top-level evaluation loop ─────────────────────────────────────────────────

def evaluate_longbench(
    runner: KVPressTextGenerationRunner,
    config: LongBenchEvalConfig,
) -> dict[str, dict[str, float]]:
    """Run all (task, bits) combinations; return {task: {label: score}}."""
    config.validate()

    # Deterministic seeds for reproducible dataset sampling
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    results: dict[str, dict[str, float]] = {}
    for task in config.tasks:
        results[task] = {}
        for bits in config.bits:
            label = f"b={bits}" if bits > 0 else "baseline"
            results[task][label] = run_task(
                runner=runner, task=task, bits=bits,
                fraction=config.fraction,
                max_new_tokens=config.max_new_tokens,
                max_context_length=config.max_context_length,
                seed=config.seed,
            )
    return results
