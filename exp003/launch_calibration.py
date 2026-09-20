#!/usr/bin/env python3
"""Run the 60-cell evaluation-only EXP-003 calibration matrix."""

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

from exp003.config import load_config
from exp003.summarize_calibration import multiplier_token, result_path


TRAINING_SEEDS = (1001, 1002, 1003)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--gpus", nargs="+", default=(0, 1, 2, 3), type=int)
    parser.add_argument("--timeout-seconds", default=3600, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_job(
    job: dict[str, Any],
    *,
    gpu: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path: Path = job["output_path"]
    log_path = output_path.with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    if not args.repo_root.exists():
        raise FileNotFoundError(args.repo_root)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU ids must be unique")
    if not args.gpus:
        raise ValueError("at least one GPU is required")

    conditions = [("none", 0.0)] + [
        (component, multiplier)
        for component in config.evaluation.components
        for multiplier in config.evaluation.multipliers
    ]
    jobs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for training_seed in TRAINING_SEEDS:
        checkpoint = (
            args.runs_root / f"clean_seed{training_seed}" / "final_ckpt.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for evaluation_seed in config.evaluation.seeds:
            for component, multiplier in conditions:
                output_path = result_path(
                    args.output_root,
                    training_seed,
                    component,
                    multiplier,
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
                        "evaluation_seed": evaluation_seed,
                        "component": component,
                        "multiplier": multiplier,
                        "checkpoint": checkpoint,
                        "output_path": output_path,
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": "EXP-003",
        "stage": "calibration",
        "config": str(args.config),
        "runs_root": str(args.runs_root),
        "output_root": str(args.output_root),
        "gpus": args.gpus,
        "jobs_planned": len(jobs),
        "jobs_skipped": len(skipped),
        "conditions": [
            {"component": component, "multiplier": multiplier}
            for component, multiplier in conditions
        ],
        "evaluation_seeds": list(config.evaluation.seeds),
    }
    (args.output_root / "launcher_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    failures: list[dict[str, Any]] = []
    completed_count = 0
    # One worker owns one GPU at a time. Assigning jobs in GPU-sized waves
    # avoids concurrent processes competing for the same device.
    for wave_start in range(0, len(jobs), len(args.gpus)):
        wave = jobs[wave_start : wave_start + len(args.gpus)]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {
                pool.submit(run_job, job, gpu=gpu, args=args): job
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
                    f"calibration {completed_count}/{len(jobs)} "
                    f"seed={job['training_seed']} eval={job['evaluation_seed']} "
                    f"component={job['component']} "
                    f"multiplier={multiplier_token(job['multiplier'])} "
                    f"returncode={result['returncode']}",
                    flush=True,
                )
        if failures:
            break

    manifest["jobs_completed"] = completed_count
    manifest["failures"] = failures
    (args.output_root / "launcher_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if failures:
        print(f"EXP003_CALIBRATION_LAUNCH_FAILED failures={len(failures)}")
        return 1
    print(
        f"EXP003_CALIBRATION_LAUNCH_OK completed={completed_count} "
        f"skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
