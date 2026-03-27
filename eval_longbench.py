#!/usr/bin/env python3
"""Standalone LongBench evaluation for TurboQuant KV quantization.

This script is fully local to Turboquant:
- no AUTOKV_PATH
- no import from sibling autokv repo
- reusable logic lives under benchmark/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from benchmark.longbench import (
    DEFAULT_BITS,
    DEFAULT_MODEL,
    DEFAULT_TASKS,
    LongBenchEvalConfig,
    build_runner,
    evaluate_longbench,
)


def _parse_tasks(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    raise ValueError(f"Unsupported tasks format: {type(raw)}")


def _parse_bits(raw: Any) -> list[int]:
    if isinstance(raw, list):
        return [int(item) for item in raw]
    if isinstance(raw, str):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    raise ValueError(f"Unsupported bits format: {type(raw)}")


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping, got: {type(raw)}")
    return raw


def _pick(cli_value: Any, yaml_data: dict[str, Any], key: str, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in yaml_data and yaml_data[key] is not None:
        return yaml_data[key]
    return default_value


def _print_summary(results: dict[str, dict[str, float]], tasks: list[str], bit_list: list[int]) -> None:
    labels = ["baseline"] + [f"b={bits}" for bits in bit_list if bits > 0]
    col_w = max(len(label) for label in labels) + 2

    header = f"{'task':<22}" + "".join(f"{label:>{col_w}}" for label in labels)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for task in tasks:
        row = f"{task:<22}"
        for label in labels:
            value = results[task].get(label, float("nan"))
            row += f"{value:>{col_w}.2f}"
        print(row)
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="LongBench evaluation for TurboQuant KV quantization")
    parser.add_argument("--config", default=None, help="YAML config path (optional)")
    parser.add_argument("--model", default=None, help="HuggingFace model ID")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated LongBench tasks",
    )
    parser.add_argument(
        "--bits",
        default=None,
        help="Comma-separated bit-widths (0 = baseline without quantization)",
    )
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of each task dataset to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_context_length", type=int, default=None)
    parser.add_argument("--output", default=None, help="JSON file to save results (optional)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    yaml_data = _load_yaml_config(args.config)

    model = _pick(args.model, yaml_data, "model", DEFAULT_MODEL)
    tasks = _parse_tasks(_pick(args.tasks, yaml_data, "tasks", DEFAULT_TASKS))
    bits = _parse_bits(_pick(args.bits, yaml_data, "bits", DEFAULT_BITS))
    fraction = float(_pick(args.fraction, yaml_data, "fraction", 0.1))
    max_new_tokens = _pick(args.max_new_tokens, yaml_data, "max_new_tokens", None)
    max_context_length = _pick(args.max_context_length, yaml_data, "max_context_length", None)
    seed = int(_pick(args.seed, yaml_data, "seed", 42))
    output_path = _pick(args.output, yaml_data, "output", None)

    config = LongBenchEvalConfig(
        model=model,
        tasks=tasks,
        bits=bits,
        fraction=fraction,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        seed=seed,
    )
    config.validate()

    print(f"Loading model: {config.model}")
    runner, device = build_runner(config.model)
    print(f"Model loaded on {device}.\n")

    print("Running evaluation...\n")
    results = evaluate_longbench(runner=runner, config=config)

    _print_summary(results=results, tasks=config.tasks, bit_list=config.bits)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
