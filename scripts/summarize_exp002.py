#!/usr/bin/env python3
"""Validate and summarize the formal EXP-002 evaluation cells.

This script is intentionally offline: it reads JSON/checkpoint metadata and
does not import ManiSkill, create an environment, or launch a training job.
It accepts either the copied local ``exp002/server_results`` directory or the
server-side ``runs/exp002`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("clean", "noisy")
SEEDS = (1001, 1002, 1003)
EVAL_CONDITIONS = ("clean", "noisy")
REQUIRED_METRICS = ("success_once", "success_at_end", "return", "episode_len")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic error path
        raise ValueError(f"cannot parse JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_formal_results(root: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], list[str]]:
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    errors: list[str] = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            run_name = f"{condition}_seed{seed}"
            run_dir = root / run_name
            config_path = run_dir / "config.json"
            checkpoint_path = run_dir / "final_ckpt.pt"
            if not config_path.exists():
                errors.append(f"missing config: {config_path}")
                continue
            if not checkpoint_path.exists():
                errors.append(f"missing checkpoint: {checkpoint_path}")
                continue

            config = load_json(config_path)
            experiment = config.get("experiment", {})
            overrides = config.get("overrides", {})
            if experiment.get("condition") != condition:
                errors.append(f"{run_name}: config condition mismatch")
            if overrides.get("seed") != seed:
                errors.append(f"{run_name}: override seed mismatch")
            if overrides.get("num_envs") != 2048:
                errors.append(f"{run_name}: num_envs is not 2048")
            if overrides.get("num_steps") != 100:
                errors.append(f"{run_name}: num_steps is not 100")
            if overrides.get("total_timesteps") != 102400000:
                errors.append(f"{run_name}: total_timesteps mismatch")
            if experiment.get("noise_enabled") != (condition == "noisy"):
                errors.append(f"{run_name}: noise_enabled mismatch")

            for eval_condition in EVAL_CONDITIONS:
                path = run_dir / f"eval_{eval_condition}.json"
                if not path.exists():
                    errors.append(f"missing evaluation: {path}")
                    continue
                data = load_json(path)
                mean = data.get("metrics_mean")
                if not isinstance(mean, dict):
                    errors.append(f"{path}: missing metrics_mean")
                    continue
                missing = [key for key in REQUIRED_METRICS if key not in mean]
                if missing:
                    errors.append(f"{path}: missing metrics {missing}")
                if data.get("episodes_collected") != 256:
                    errors.append(f"{path}: episodes_collected is not 256")
                if data.get("eval_noise") != (eval_condition == "noisy"):
                    errors.append(f"{path}: eval_noise flag mismatch")
                if mean.get("episode_len") != 100.0:
                    errors.append(f"{path}: episode_len is not 100")
                cells[(condition, seed, eval_condition)] = data

    return cells, errors


def metric_summary(
    cells: dict[tuple[str, int, str], dict[str, Any]],
    condition: str,
    eval_condition: str,
    metric: str,
) -> dict[str, Any]:
    values = [
        float(cells[(condition, seed, eval_condition)]["metrics_mean"][metric])
        for seed in SEEDS
    ]
    return {
        "mean": sum(values) / len(values),
        "sample_std": sample_std(values),
        "values": dict(zip((str(seed) for seed in SEEDS), values)),
    }


def build_summary(cells: dict[tuple[str, int, str], dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for eval_condition in EVAL_CONDITIONS:
            aggregate[condition][eval_condition] = {
                metric: metric_summary(cells, condition, eval_condition, metric)
                for metric in ("success_once", "success_at_end", "return")
            }

    paired = {}
    for seed in SEEDS:
        clean_training_noisy_eval = float(
            cells[("clean", seed, "noisy")]["metrics_mean"]["success_once"]
        )
        noisy_training_noisy_eval = float(
            cells[("noisy", seed, "noisy")]["metrics_mean"]["success_once"]
        )
        clean_training_clean_eval = float(
            cells[("clean", seed, "clean")]["metrics_mean"]["success_once"]
        )
        noisy_training_clean_eval = float(
            cells[("noisy", seed, "clean")]["metrics_mean"]["success_once"]
        )
        paired[str(seed)] = {
            "robustness_gain": noisy_training_noisy_eval - clean_training_noisy_eval,
            "clean_cost": noisy_training_clean_eval - clean_training_clean_eval,
        }

    robustness_values = [item["robustness_gain"] for item in paired.values()]
    clean_cost_values = [item["clean_cost"] for item in paired.values()]
    paired["aggregate"] = {
        "robustness_gain": {
            "mean": sum(robustness_values) / len(robustness_values),
            "sample_std": sample_std(robustness_values),
        },
        "clean_cost": {
            "mean": sum(clean_cost_values) / len(clean_cost_values),
            "sample_std": sample_std(clean_cost_values),
        },
    }

    return {
        "experiment": "EXP-002",
        "formal_cells": len(cells),
        "seeds": list(SEEDS),
        "aggregate": aggregate,
        "paired_effects": paired,
        "validation": {
            "episodes_per_cell": 256,
            "episode_length": 100,
            "excluded_smoke_runs": True,
        },
    }


def print_report(summary: dict[str, Any]) -> None:
    print(f"formal_cells={summary['formal_cells']}")
    for condition in CONDITIONS:
        for eval_condition in EVAL_CONDITIONS:
            result = summary["aggregate"][condition][eval_condition]["success_once"]
            print(
                f"{condition}_train -> {eval_condition}_eval: "
                f"success_once={result['mean'] * 100:.2f}% "
                f"+- {result['sample_std'] * 100:.2f} pp"
            )
    effects = summary["paired_effects"]["aggregate"]
    print(
        "robustness_gain="
        f"{effects['robustness_gain']['mean'] * 100:+.2f} pp "
        f"+- {effects['robustness_gain']['sample_std'] * 100:.2f} pp"
    )
    print(
        "clean_cost="
        f"{effects['clean_cost']['mean'] * 100:+.2f} pp "
        f"+- {effects['clean_cost']['sample_std'] * 100:.2f} pp"
    )


def main() -> int:
    args = parse_args()
    cells, errors = load_formal_results(args.results_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    expected = len(CONDITIONS) * len(SEEDS) * len(EVAL_CONDITIONS)
    if len(cells) != expected:
        print(f"ERROR: expected {expected} formal cells, found {len(cells)}")
        return 1

    summary = build_summary(cells)
    print_report(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"summary_json={args.output_json}")
    print("EXP002_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
