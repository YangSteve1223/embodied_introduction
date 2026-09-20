#!/usr/bin/env python3
"""Validate the 60-cell EXP-003 calibration matrix and freeze a severity."""

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

from exp003.config import load_config


TRAINING_SEEDS = (1001, 1002, 1003)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    return parser.parse_args()


def multiplier_token(value: float) -> str:
    return format(value, "g").replace(".", "p")


def result_path(
    root: Path,
    training_seed: int,
    component: str,
    multiplier: float,
    evaluation_seed: int,
) -> Path:
    return (
        root
        / f"clean_seed{training_seed}"
        / f"component-{component}"
        / f"multiplier-{multiplier_token(multiplier)}"
        / f"eval_seed{evaluation_seed}.json"
    )


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


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    evaluation_seeds = config.evaluation.seeds
    conditions = [("none", 0.0)] + [
        (component, multiplier)
        for component in config.evaluation.components
        for multiplier in config.evaluation.multipliers
    ]
    cells: dict[tuple[int, int, str, float], dict[str, Any]] = {}
    errors: list[str] = []

    for training_seed in TRAINING_SEEDS:
        for evaluation_seed in evaluation_seeds:
            for component, multiplier in conditions:
                path = result_path(
                    args.results_root,
                    training_seed,
                    component,
                    multiplier,
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
                noise = data.get("noise", {})
                if data.get("training_seed") != training_seed:
                    errors.append(f"{path}: training_seed mismatch")
                if data.get("evaluation_seed") != evaluation_seed:
                    errors.append(f"{path}: evaluation_seed mismatch")
                if noise.get("component") != component:
                    errors.append(f"{path}: component mismatch")
                if float(noise.get("multiplier", -1)) != multiplier:
                    errors.append(f"{path}: multiplier mismatch")
                if data.get("episodes_collected") != config.evaluation.episodes:
                    errors.append(f"{path}: episode count mismatch")
                metrics = data.get("metrics_mean", {})
                interface = data.get("interface_metrics", {})
                for key in ("success_once", "success_at_end", "return", "episode_len"):
                    if key not in metrics:
                        errors.append(f"{path}: missing metric {key}")
                if metrics.get("episode_len") != 100.0:
                    errors.append(f"{path}: episode length is not 100")
                if "noise_induced_clip_fraction" not in interface:
                    errors.append(f"{path}: missing clip instrumentation")
                cells[(training_seed, evaluation_seed, component, multiplier)] = data

    expected = len(TRAINING_SEEDS) * len(evaluation_seeds) * len(conditions)
    if errors or len(cells) != expected:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"ERROR: expected {expected} cells, found {len(cells)}")
        return 1

    aggregates: dict[str, Any] = {}
    candidates: list[float] = []
    for component, multiplier in conditions:
        successes: list[float] = []
        returns: list[float] = []
        drops: list[float] = []
        clip_fractions: list[float] = []
        per_training_seed: dict[str, Any] = {}
        for training_seed in TRAINING_SEEDS:
            seed_drops = []
            for evaluation_seed in evaluation_seeds:
                cell = cells[(training_seed, evaluation_seed, component, multiplier)]
                baseline = cells[(training_seed, evaluation_seed, "none", 0.0)]
                success = float(cell["metrics_mean"]["success_once"])
                baseline_success = float(baseline["metrics_mean"]["success_once"])
                drop = baseline_success - success
                successes.append(success)
                returns.append(float(cell["metrics_mean"]["return"]))
                drops.append(drop)
                seed_drops.append(drop)
                clip_fractions.append(
                    float(cell["interface_metrics"]["noise_induced_clip_fraction"])
                )
            per_training_seed[str(training_seed)] = {
                "success_drop": sum(seed_drops) / len(seed_drops)
            }

        key = f"{component}_{multiplier_token(multiplier)}x"
        mean_success = sum(successes) / len(successes)
        mean_drop = sum(drops) / len(drops)
        mean_clip = sum(clip_fractions) / len(clip_fractions)
        direction_consistent = all(
            value["success_drop"] > 0 for value in per_training_seed.values()
        )
        eligible = (
            component == "combined"
            and 0.10 <= mean_drop <= 0.25
            and mean_success >= 0.05
            and direction_consistent
            and mean_clip <= 0.05
        )
        if eligible:
            candidates.append(multiplier)
        aggregates[key] = {
            "component": component,
            "multiplier": multiplier,
            "success_once_mean": mean_success,
            "success_once_sample_std_across_cells": sample_std(successes),
            "paired_success_drop_mean": mean_drop,
            "return_mean": sum(returns) / len(returns),
            "noise_induced_clip_fraction_mean": mean_clip,
            "direction_consistent_across_training_seeds": direction_consistent,
            "eligible": eligible,
            "per_training_seed": per_training_seed,
        }

    summary = {
        "experiment_id": "EXP-003",
        "stage": "calibration",
        "formal_cells": len(cells),
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(evaluation_seeds),
        "selection_rule": {
            "component": "combined",
            "paired_success_drop_min": 0.10,
            "paired_success_drop_max": 0.25,
            "success_floor": 0.05,
            "noise_induced_clip_fraction_max": 0.05,
            "direction_consistent_across_training_seeds": True,
            "choose_smallest_eligible_multiplier": True,
        },
        "aggregate": aggregates,
    }
    selection: dict[str, Any]
    if candidates:
        selected = min(candidates)
        selection = {
            "experiment_id": "EXP-003",
            "stage": "calibration",
            "status": "selected",
            "component": "combined",
            "target_multiplier": selected,
            "source_summary": str(args.output_json),
            "selected_aggregate": aggregates[
                f"combined_{multiplier_token(selected)}x"
            ],
        }
        summary["selection"] = selection
    else:
        selection = {
            "experiment_id": "EXP-003",
            "stage": "calibration",
            "status": "no_candidate",
            "component": "combined",
            "source_summary": str(args.output_json),
        }
        summary["selection"] = selection

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.selection_json.write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    if candidates:
        print(f"EXP003_CALIBRATION_OK target_multiplier={min(candidates):g}")
        return 0
    print("EXP003_CALIBRATION_NO_CANDIDATE")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
