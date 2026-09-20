#!/usr/bin/env python3
"""Validate and summarize the complete EXP-004 primary/secondary matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from exp004.config import (
    ARMS,
    TRAINING_SEEDS,
    expected_stage_count,
    load_config,
    primary_matrix,
    secondary_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def avg(values: list[float]) -> float:
    return mean(values)


def sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def load_stage(root: Path, stage: str, config: Any) -> dict[tuple[int, str, int, str, int], dict[str, Any]]:
    expected = expected_stage_count(config, stage)
    expected_rows = primary_matrix(config) if stage == "primary" else secondary_matrix(config)
    expected_keys = {
        (row["training_seed"], row["model"], row["checkpoint_update"], row["condition"], row["evaluation_seed"])
        for row in expected_rows
    }
    files = sorted((root / stage).glob("seed*/model-*/update-*/eval-*/*.json"))
    if len(files) != expected:
        raise ValueError(f"{stage}: expected {expected} JSON cells, found {len(files)}")
    cells: dict[tuple[int, str, int, str, int], dict[str, Any]] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"experiment_id", "stage", "model", "checkpoint_update", "training_seed", "evaluation_condition", "evaluation_multiplier", "evaluation_seed", "episodes_collected", "metrics_mean", "interface_metrics"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        if data["experiment_id"] != "EXP-004" or data["stage"] != stage or data.get("smoke") is not False:
            raise ValueError(f"{path}: wrong experiment/stage/smoke marker")
        if data["training_seed"] not in TRAINING_SEEDS or data["model"] not in ((*ARMS, "C4") if stage == "primary" else ARMS):
            raise ValueError(f"{path}: unexpected seed or model")
        if data["episodes_collected"] != config.evaluation.episodes:
            raise ValueError(f"{path}: episode count mismatch")
        if set(("success_once", "success_at_end", "return", "episode_len")) - set(data["metrics_mean"]):
            raise ValueError(f"{path}: required metric missing")
        if data["metrics_mean"]["episode_len"] != 100.0:
            raise ValueError(f"{path}: episode length is not 100")
        condition = str(data["evaluation_condition"])
        multiplier = float(data["evaluation_multiplier"])
        if condition == "clean" and multiplier != 0.0:
            raise ValueError(f"{path}: clean multiplier mismatch")
        if condition == "combined4x" and multiplier != 4.0:
            raise ValueError(f"{path}: combined4x multiplier mismatch")
        key = (int(data["training_seed"]), str(data["model"]), int(data["checkpoint_update"]), condition, int(data["evaluation_seed"]))
        if key in cells:
            raise ValueError(f"duplicate cell: {key}")
        cells[key] = data
    if set(cells) != expected_keys:
        missing = sorted(expected_keys - set(cells))
        unexpected = sorted(set(cells) - expected_keys)
        raise ValueError(f"{stage}: matrix mismatch missing={missing[:3]} unexpected={unexpected[:3]}")
    return cells


def cell_metrics(cells: dict[tuple[int, str, int, str, int], dict[str, Any]], seed: int, model: str, update: int, condition: str, eval_seeds: tuple[int, ...]) -> dict[str, float]:
    rows = [cells[(seed, model, update, condition, eval_seed)] for eval_seed in eval_seeds]
    return {
        "success_once": avg([float(row["metrics_mean"]["success_once"]) for row in rows]),
        "success_at_end": avg([float(row["metrics_mean"]["success_at_end"]) for row in rows]),
        "return": avg([float(row["metrics_mean"]["return"]) for row in rows]),
        "noise_induced_clip_fraction": avg([float(row["interface_metrics"]["noise_induced_clip_fraction"]) for row in rows]),
    }


def summarize_endpoint(cells: dict[tuple[int, str, int, str, int], dict[str, Any]], config: Any, stage: str, update: int, models: tuple[str, ...]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        by_model: dict[str, Any] = {}
        for model in models:
            by_model[model] = {
                condition: cell_metrics(cells, seed, model, update, condition, config.evaluation.seeds)
                for condition in ("clean", "combined4x")
            }
        base = by_model["C0"]
        for model in models:
            if model == "C0" or model == "C4":
                continue
            by_model[model]["paired_effects"] = {
                "robustness_gain": by_model[model]["combined4x"]["success_once"] - base["combined4x"]["success_once"],
                "clean_cost": by_model[model]["clean"]["success_once"] - base["clean"]["success_once"],
                "noisy_return_gain": by_model[model]["combined4x"]["return"] - base["combined4x"]["return"],
            }
        per_seed[str(seed)] = by_model
    treatments = {arm: [per_seed[str(seed)][arm]["paired_effects"] for seed in TRAINING_SEEDS] for arm in ("C1", "C2", "C3")}
    aggregate: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for arm, effects in treatments.items():
        robustness = [x["robustness_gain"] for x in effects]
        clean_cost = [x["clean_cost"] for x in effects]
        noisy_return = [x["noisy_return_gain"] for x in effects]
        clip = [per_seed[str(seed)][arm]["combined4x"]["noise_induced_clip_fraction"] for seed in TRAINING_SEEDS]
        aggregate[arm] = {
            "robustness_gain": {"mean": avg(robustness), "sample_std": sd(robustness), "values": dict(zip(map(str, TRAINING_SEEDS), robustness))},
            "clean_cost": {"mean": avg(clean_cost), "sample_std": sd(clean_cost), "values": dict(zip(map(str, TRAINING_SEEDS), clean_cost))},
            "noisy_return_gain": {"mean": avg(noisy_return), "sample_std": sd(noisy_return)},
            "noise_induced_clip_fraction": {"mean": avg(clip), "max": max(clip)},
            "positive_robustness_seeds": sum(x > 0 for x in robustness),
        }
        gates[arm] = {
            "robustness_gain_at_least_5pp": aggregate[arm]["robustness_gain"]["mean"] >= 0.05,
            "at_least_two_positive_seeds": aggregate[arm]["positive_robustness_seeds"] >= 2,
            "clean_cost_no_worse_than_minus_5pp": aggregate[arm]["clean_cost"]["mean"] >= -0.05,
            "noisy_return_not_lower": aggregate[arm]["noisy_return_gain"]["mean"] >= 0.0,
            "noise_induced_clip_not_abnormal": aggregate[arm]["noise_induced_clip_fraction"]["max"] <= 0.05,
        }
        gates[arm]["all_gates_pass"] = all(gates[arm].values())
    return {
        "stage": stage, "endpoint_update": update, "per_seed": per_seed,
        "aggregate": aggregate, "engineering_success_gate": gates,
        "any_treatment_passes": any(value["all_gates_pass"] for value in gates.values()),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    primary = load_stage(args.evaluation_root, "primary", config)
    secondary = load_stage(args.evaluation_root, "secondary", config)
    summary = {
        "experiment_id": "EXP-004",
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(config.evaluation.seeds),
        "primary": summarize_endpoint(primary, config, "primary", 100, (*ARMS, "C4")),
        "secondary": summarize_endpoint(secondary, config, "secondary", 150, ARMS),
        "reference": {"C4": "EXP-003 treatment checkpoint at combined 4x, update 100; reference only, not retrained"},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    for stage_name in ("primary", "secondary"):
        stage = summary[stage_name]
        for arm, gate in stage["engineering_success_gate"].items():
            effect = stage["aggregate"][arm]
            print(f"{stage_name} {arm} robustness_gain={effect['robustness_gain']['mean'] * 100:+.2f}pp clean_cost={effect['clean_cost']['mean'] * 100:+.2f}pp positive={effect['positive_robustness_seeds']}/3 pass={gate['all_gates_pass']}")
    print(f"EXP004_SUMMARY_OK output={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
