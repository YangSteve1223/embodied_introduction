#!/usr/bin/env python3
"""Launch the six paired EXP-003 formal continuation runs."""

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
TRAINING_SEEDS = (1001, 1002, 1003)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--exp002-runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--gpus", nargs="+", default=(0, 1, 2, 3), type=int)
    parser.add_argument("--timeout-seconds", default=28800, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_job(job: dict[str, Any], gpu: int, args: argparse.Namespace) -> dict[str, Any]:
    logs_dir = args.output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"train_{job['condition']}_seed{job['seed']}.log"
    command = [
        str(args.python),
        "-m",
        "exp003.continue_ppo",
        "--config",
        str(args.config),
        "--source-checkpoint",
        str(job["source_checkpoint"]),
        "--output-dir",
        str(job["output_dir"]),
        "--condition",
        job["condition"],
        "--seed",
        str(job["seed"]),
    ]
    if job["condition"] == "treatment":
        command.extend(("--selection-json", str(args.selection_json)))
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
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError("GPU ids must be non-empty and unique")
    for path in (args.config, args.selection_json, args.python, args.repo_root):
        if not path.exists():
            raise FileNotFoundError(path)

    jobs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for seed in TRAINING_SEEDS:
        source = args.exp002_runs_root / f"clean_seed{seed}" / "final_ckpt.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        for condition in ("control", "treatment"):
            output_dir = args.output_root / f"{condition}_seed{seed}"
            final_checkpoint = output_dir / "final_ckpt.pt"
            if final_checkpoint.exists() and args.resume:
                skipped.append(str(final_checkpoint))
                continue
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(
                    f"incomplete/non-empty run directory requires review: {output_dir}"
                )
            jobs.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "source_checkpoint": source,
                    "output_dir": output_dir,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment_id": "EXP-003",
        "stage": "formal_training",
        "config": str(args.config),
        "selection_json": str(args.selection_json),
        "exp002_runs_root": str(args.exp002_runs_root),
        "output_root": str(args.output_root),
        "gpus": args.gpus,
        "jobs_planned": len(jobs),
        "jobs_skipped": len(skipped),
    }
    manifest_path = args.output_root / "training_launcher_manifest.json"
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
                    f"formal training {completed_count}/{len(jobs)} "
                    f"condition={job['condition']} seed={job['seed']} "
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
        print(f"EXP003_FORMAL_TRAINING_FAILED failures={len(failures)}")
        return 1
    print(
        f"EXP003_FORMAL_TRAINING_OK completed={completed_count} "
        f"skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
