#!/usr/bin/env python3
"""LongBench evaluation for TurboQuant KV quantization.

Usage:
    python eval_longbench.py --tasks qasper,triviaqa --bits 0,8,4,2,1 --fraction 0.1
    python eval_longbench.py --config eval_config.yaml

bits=0 is the unquantised baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.longbench import (
    DEFAULT_BITS, DEFAULT_MODEL, DEFAULT_TASKS,
    LongBenchEvalConfig, build_runner, evaluate_longbench,
)


def _csv_str(raw: Any, default: list[str]) -> list[str]:
    if isinstance(raw, list): return [str(x).strip() for x in raw]
    if isinstance(raw, str):  return [x.strip() for x in raw.split(",") if x.strip()]
    return default

def _csv_int(raw: Any, default: list[int]) -> list[int]:
    if isinstance(raw, list): return [int(x) for x in raw]
    if isinstance(raw, str):  return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return default


def _print_table(results: dict[str, dict[str, float]], tasks: list[str], bits: list[int]) -> None:
    labels = ["baseline"] + [f"b={b}" for b in bits if b > 0]
    col    = max(len(l) for l in labels) + 2
    header = f"{'task':<22}" + "".join(f"{l:>{col}}" for l in labels)
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for task in tasks:
        row = f"{task:<22}" + "".join(f"{results[task].get(l, float('nan')):>{col}.2f}" for l in labels)
        print(row)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              default=None)
    parser.add_argument("--tasks",              default=None, help="comma-separated task names")
    parser.add_argument("--bits",               default=None, help="comma-separated bit-widths; 0=baseline")
    parser.add_argument("--fraction",           type=float, default=None)
    parser.add_argument("--max_new_tokens",     type=int,   default=None)
    parser.add_argument("--max_context_length", type=int,   default=None)
    parser.add_argument("--seed",               type=int,   default=None)
    parser.add_argument("--output",             default=None, help="path to save JSON results")

    # Optional YAML config (CLI args take priority)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    yaml_cfg: dict[str, Any] = {}
    if args.config:
        import yaml
        yaml_cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    def pick(cli_val: Any, key: str, default: Any) -> Any:
        return cli_val if cli_val is not None else yaml_cfg.get(key, default)

    config = LongBenchEvalConfig(
        model              = pick(args.model,              "model",              DEFAULT_MODEL),
        tasks              = _csv_str(pick(args.tasks,     "tasks",              DEFAULT_TASKS), DEFAULT_TASKS),
        bits               = _csv_int(pick(args.bits,      "bits",               DEFAULT_BITS),  DEFAULT_BITS),
        fraction           = float(pick(args.fraction,     "fraction",           0.1)),
        max_new_tokens     = pick(args.max_new_tokens,     "max_new_tokens",     None),
        max_context_length = pick(args.max_context_length, "max_context_length", None),
        seed               = int(pick(args.seed,           "seed",               42)),
    )
    config.validate()

    print(f"Model: {config.model}")
    runner, device = build_runner(config.model)
    print(f"Loaded on {device}.\n")

    results = evaluate_longbench(runner=runner, config=config)
    _print_table(results, config.tasks, config.bits)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
