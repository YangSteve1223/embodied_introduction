#!/usr/bin/env python3
"""Validate and summarize the 36-cell EXP-003 formal evaluation matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp003.config import load_config, load_selection
from exp003.launch_formal_evaluation import formal_result_path


TRAINING_SEEDS = (1001, 1002, 1003)
MODEL_TYPES = ("original", "control", "treatment")
EVAL_CONDITIONS = ("clean", "noisy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    target_multiplier = load_selection(args.selection_json)
    cells: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    errors: list[str] = []

    for training_seed in TRAINING_SEEDS:
        for model_type in MODEL_TYPES:
            for eval_condition in EVAL_CONDITIONS:
                for evaluation_seed in config.evaluation.seeds:
                    path = formal_result_path(
                        args.results_root,
                        training_seed,
                        model_type,
                        eval_condition,
                        evaluation_seed,
                    )
                    if not path.exists():
                        errors.append(f"missing result: {path}")
                        continue
                    try:
                        data = load_json(path)
                    except Exception as exc:
                        errors.append(str(exc))
                        continue
                    expected_component = "none" if eval_condition == "clean" else "combined"
                    expected_multiplier = 0.0 if eval_condition == "clean" else target_multiplier
                    noise = data.get("noise", {})
                    if data.get("training_seed") != training_seed:
                        errors.append(f"{path}: training_seed mismatch")
                    if data.get("evaluation_seed") != evaluation_seed:
                        errors.append(f"{path}: evaluation_seed mismatch")
                    if noise.get("component") != expected_component:
                        errors.append(f"{path}: noise component mismatch")
                    if float(noise.get("multiplier", -1)) != expected_multiplier:
                        errors.append(f"{path}: noise multiplier mismatch")
                    if data.get("episodes_collected") != config.evaluation.episodes:
                        errors.append(f"{path}: episode count mismatch")
                    metrics = data.get("metrics_mean", {})
                    for key in ("success_once", "success_at_end", "return", "episode_len"):
                        if key not in metrics:
                            errors.append(f"{path}: missing metric {key}")
                    if metrics.get("episode_len") != 100.0:
                        errors.append(f"{path}: episode length is not 100")
                    expected_checkpoint_condition = (
                        "exp002_clean" if model_type == "original" else model_type
                    )
                    if data.get("checkpoint_condition") != expected_checkpoint_condition:
                        errors.append(f"{path}: checkpoint condition mismatch")
                    cells[(training_seed, model_type, eval_condition, evaluation_seed)] = data

    expected = (
        len(TRAINING_SEEDS)
        * len(MODEL_TYPES)
        * len(EVAL_CONDITIONS)
        * len(config.evaluation.seeds)
    )
    if errors or len(cells) != expected:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"ERROR: expected {expected} cells, found {len(cells)}")
        return 1

    per_seed: dict[str, Any] = {}
    for training_seed in TRAINING_SEEDS:
        seed_metrics: dict[str, Any] = {}
        for model_type in MODEL_TYPES:
            seed_metrics[model_type] = {}
            for eval_condition in EVAL_CONDITIONS:
                success_values = [
                    float(
                        cells[
                            (training_seed, model_type, eval_condition, evaluation_seed)
                        ]["metrics_mean"]["success_once"]
                    )
                    for evaluation_seed in config.evaluation.seeds
                ]
                return_values = [
                    float(
                        cells[
                            (training_seed, model_type, eval_condition, evaluation_seed)
                        ]["metrics_mean"]["return"]
                    )
                    for evaluation_seed in config.evaluation.seeds
                ]
                clip_values = [
                    float(
                        cells[
                            (training_seed, model_type, eval_condition, evaluation_seed)
                        ]["interface_metrics"]["noise_induced_clip_fraction"]
                    )
                    for evaluation_seed in config.evaluation.seeds
                ]
                seed_metrics[model_type][eval_condition] = {
                    "success_once": mean(success_values),
                    "return": mean(return_values),
                    "noise_induced_clip_fraction": mean(clip_values),
                }
        seed_metrics["paired_effects"] = {
            "robustness_gain": (
                seed_metrics["treatment"]["noisy"]["success_once"]
                - seed_metrics["control"]["noisy"]["success_once"]
            ),
            "clean_cost": (
                seed_metrics["treatment"]["clean"]["success_once"]
                - seed_metrics["control"]["clean"]["success_once"]
            ),
            "control_noisy_continuation_gain": (
                seed_metrics["control"]["noisy"]["success_once"]
                - seed_metrics["original"]["noisy"]["success_once"]
            ),
            "treatment_noisy_total_gain": (
                seed_metrics["treatment"]["noisy"]["success_once"]
                - seed_metrics["original"]["noisy"]["success_once"]
            ),
            "noisy_return_gain": (
                seed_metrics["treatment"]["noisy"]["return"]
                - seed_metrics["control"]["noisy"]["return"]
            ),
        }
        per_seed[str(training_seed)] = seed_metrics

    robustness = [
        per_seed[str(seed)]["paired_effects"]["robustness_gain"]
        for seed in TRAINING_SEEDS
    ]
    clean_costs = [
        per_seed[str(seed)]["paired_effects"]["clean_cost"]
        for seed in TRAINING_SEEDS
    ]
    noisy_return_gains = [
        per_seed[str(seed)]["paired_effects"]["noisy_return_gain"]
        for seed in TRAINING_SEEDS
    ]
    positive_seeds = sum(value > 0 for value in robustness)
    aggregate = {
        "robustness_gain": {
            "mean": mean(robustness),
            "sample_std": sample_std(robustness),
            "values": dict(zip((str(seed) for seed in TRAINING_SEEDS), robustness)),
        },
        "clean_cost": {
            "mean": mean(clean_costs),
            "sample_std": sample_std(clean_costs),
            "values": dict(zip((str(seed) for seed in TRAINING_SEEDS), clean_costs)),
        },
        "noisy_return_gain": {
            "mean": mean(noisy_return_gains),
            "sample_std": sample_std(noisy_return_gains),
        },
        "positive_robustness_seeds": positive_seeds,
    }
    success_gate = {
        "robustness_gain_at_least_5pp": aggregate["robustness_gain"]["mean"] >= 0.05,
        "at_least_two_positive_seeds": positive_seeds >= 2,
        "clean_cost_no_worse_than_minus_5pp": aggregate["clean_cost"]["mean"] >= -0.05,
        "noisy_return_not_lower": aggregate["noisy_return_gain"]["mean"] >= 0,
    }
    summary = {
        "experiment_id": "EXP-003",
        "stage": "formal",
        "formal_cells": len(cells),
        "target_multiplier": target_multiplier,
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(config.evaluation.seeds),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "engineering_success_gate": success_gate,
        "all_primary_gates_pass": all(success_gate.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"robustness_gain={aggregate['robustness_gain']['mean'] * 100:+.2f} pp "
        f"+- {aggregate['robustness_gain']['sample_std'] * 100:.2f} pp"
    )
    print(
        f"clean_cost={aggregate['clean_cost']['mean'] * 100:+.2f} pp "
        f"+- {aggregate['clean_cost']['sample_std'] * 100:.2f} pp"
    )
    print(
        f"positive_robustness_seeds={positive_seeds}/3 "
        f"all_primary_gates_pass={summary['all_primary_gates_pass']}"
    )
    print(f"EXP003_FORMAL_SUMMARY_OK output={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
