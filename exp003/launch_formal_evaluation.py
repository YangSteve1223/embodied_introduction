#!/usr/bin/env python3
"""Evaluate original/control/treatment policies in the EXP-003 formal matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp003.config import load_config, load_selection


TRAINING_SEEDS = (1001, 1002, 1003)
MODEL_TYPES = ("original", "control", "treatment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--exp002-runs-root", required=True, type=Path)
    parser.add_argument("--formal-runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--gpus", nargs="+", default=(0, 1, 2, 3), type=int)
    parser.add_argument("--timeout-seconds", default=3600, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def formal_result_path(
    root: Path,
    training_seed: int,
    model_type: str,
    eval_condition: str,
    evaluation_seed: int,
) -> Path:
    return (
        root
        / f"seed{training_seed}"
        / f"model-{model_type}"
        / f"eval-{eval_condition}"
        / f"eval_seed{evaluation_seed}.json"
    )


def run_job(job: dict[str, Any], gpu: int, args: argparse.Namespace) -> dict[str, Any]:
    output_path: Path = job["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    command = [
        str(args.python),
        "-m",
        "exp003.evaluate",
        "--config",
        str(args.config),
        "--checkpoint",
        str(job["checkpoint"]),
        "--output-json",
        str(output_path),
        "--component",
        job["component"],
        "--multiplier",
        str(job["multiplier"]),
        "--seed",
        str(job["evaluation_seed"]),
        "--expected-training-seed",
        str(job["training_seed"]),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    return {
        **job,
        "gpu": gpu,
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    target_multiplier = load_selection(args.selection_json)
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError("GPU ids must be non-empty and unique")

    jobs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for training_seed in TRAINING_SEEDS:
        checkpoints = {
            "original": args.exp002_runs_root
            / f"clean_seed{training_seed}"
            / "final_ckpt.pt",
            "control": args.formal_runs_root
            / f"control_seed{training_seed}"
            / "final_ckpt.pt",
            "treatment": args.formal_runs_root
            / f"treatment_seed{training_seed}"
            / "final_ckpt.pt",
        }
        for checkpoint in checkpoints.values():
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        for model_type in MODEL_TYPES:
            for eval_condition, component, multiplier in (
                ("clean", "none", 0.0),
                ("noisy", "combined", target_multiplier),
            ):
                for evaluation_seed in config.evaluation.seeds:
                    output_path = formal_result_path(
                        args.output_root,
                        training_seed,
                        model_type,
                        eval_condition,
                        evaluation_seed,
                    )
                    if output_path.exists():
                        if args.resume:
                            skipped.append(str(output_path))
                            continue
                        raise FileExistsError(
                            f"output exists; use --resume to skip it: {output_path}"
                        )
                    jobs.append(
                        {
                            "training_seed": training_seed,
                            "model_type": model_type,
                            "eval_condition": eval_condition,
                            "evaluation_seed": evaluation_seed,
                            "component": component,
                            "multiplier": multiplier,
                            "checkpoint": checkpoints[model_type],
                            "output_path": output_path,
                        }
                    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment_id": "EXP-003",
        "stage": "formal_evaluation",
        "target_multiplier": target_multiplier,
        "jobs_planned": len(jobs),
        "jobs_skipped": len(skipped),
        "gpus": args.gpus,
    }
    manifest_path = args.output_root / "evaluation_launcher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    failures: list[dict[str, Any]] = []
    completed_count = 0
    for wave_start in range(0, len(jobs), len(args.gpus)):
        wave = jobs[wave_start : wave_start + len(args.gpus)]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {
                pool.submit(run_job, job, gpu, args): job
                for job, gpu in zip(wave, args.gpus)
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**job, "returncode": -1, "error": repr(exc)}
                completed_count += 1
                if result["returncode"] != 0:
                    failures.append(result)
                print(
                    f"formal eval {completed_count}/{len(jobs)} "
                    f"seed={job['training_seed']} model={job['model_type']} "
                    f"eval={job['eval_condition']} "
                    f"eval_seed={job['evaluation_seed']} "
                    f"returncode={result['returncode']}",
                    flush=True,
                )
        if failures:
            break

    manifest["jobs_completed"] = completed_count
    manifest["failures"] = failures
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if failures:
        print(f"EXP003_FORMAL_EVALUATION_FAILED failures={len(failures)}")
        return 1
    print(
        f"EXP003_FORMAL_EVALUATION_OK completed={completed_count} "
        f"skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
