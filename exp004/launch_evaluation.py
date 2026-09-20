#!/usr/bin/env python3
"""Run EXP-004 evaluation matrices, primary before secondary."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from exp004.config import cross_matrix, load_config, primary_matrix, secondary_matrix

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--exp002-runs-root", required=True, type=Path)
    parser.add_argument("--exp003-runs-root", required=True, type=Path)
    parser.add_argument("--formal-runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=ROOT, type=Path)
    parser.add_argument("--gpus", nargs=4, default=(0, 1, 2, 3), type=int)
    parser.add_argument("--stages", nargs="+", choices=("primary", "secondary", "cross"), default=("primary", "secondary"))
    parser.add_argument("--timeout-seconds", default=3600, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def checkpoint_for(job: dict[str, Any], args: argparse.Namespace) -> Path:
    seed, model, update = job["training_seed"], job["model"], job["checkpoint_update"]
    if model == "C4":
        return args.exp003_runs_root / "formal_training" / f"treatment_seed{seed}" / "checkpoints" / "checkpoint_update_0100.pt"
    root = args.formal_runs_root / f"{model}_seed{seed}"
    if update == 100:
        return root / "checkpoints" / "checkpoint_update_0100.pt"
    if update == 150:
        return root / "final_ckpt.pt"
    raise ValueError(f"unsupported checkpoint update: {update}")


def output_for(job: dict[str, Any], args: argparse.Namespace) -> Path:
    return (args.output_root / job["stage"] / f"seed{job['training_seed']}"
            / f"model-{job['model']}" / f"update-{job['checkpoint_update']:03d}"
            / f"eval-{job['condition']}" / f"eval_seed{job['evaluation_seed']}.json")


def run_job(job: dict[str, Any], gpu: int, args: argparse.Namespace) -> dict[str, Any]:
    output = job["output_path"]
    log = output.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), "-m", "exp004.evaluate", "--config", str(args.config),
        "--checkpoint", str(job["checkpoint"]), "--output-json", str(output),
        "--stage", job["stage"], "--model", job["model"],
        "--checkpoint-update", str(job["checkpoint_update"]),
        "--condition", job["condition"], "--multiplier", str(job["multiplier"]),
        "--seed", str(job["evaluation_seed"]),
        "--expected-training-seed", str(job["training_seed"]),
    ]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds, check=False)
    return {**job, "gpu": gpu, "returncode": result.returncode, "elapsed_seconds": time.time() - started, "log": str(log)}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if len(set(args.gpus)) != 4:
        raise ValueError("evaluation requires four unique GPU ids")
    for path in (args.config, args.python, args.repo_root):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_failures: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    for stage in args.stages:
        rows = primary_matrix(config) if stage == "primary" else secondary_matrix(config) if stage == "secondary" else cross_matrix(config)
        jobs: list[dict[str, Any]] = []
        skipped: list[str] = []
        for row in rows:
            job = {**row, "stage": stage}
            checkpoint = checkpoint_for(job, args)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            output = output_for(job, args)
            if output.exists() and args.resume:
                skipped.append(str(output))
                continue
            if output.exists():
                raise FileExistsError(f"evaluation output exists; use --resume: {output}")
            job.update({"checkpoint": checkpoint, "output_path": output})
            jobs.append(job)
        manifest_path = args.output_root / f"{stage}_launcher_manifest.json"
        manifest = {"experiment_id": "EXP-004", "stage": stage, "smoke": False, "jobs_planned": len(jobs), "jobs_skipped": len(skipped), "gpus": list(args.gpus)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        failures: list[dict[str, Any]] = []
        completed = 0
        for start in range(0, len(jobs), 4):
            wave = jobs[start:start + 4]
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {pool.submit(run_job, job, args.gpus[i], args): job for i, job in enumerate(wave)}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {**job, "returncode": -1, "error": repr(exc)}
                    completed += 1
                    if result["returncode"] != 0:
                        failures.append(result)
                    print(f"{stage} eval {completed}/{len(jobs)} model={job['model']} seed={job['training_seed']} condition={job['condition']} eval_seed={job['evaluation_seed']} returncode={result['returncode']}", flush=True)
            if failures:
                break
        manifest.update({"jobs_completed": completed, "failures": failures})
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        stage_counts[stage] = completed + len(skipped)
        if failures:
            stage_failures.extend(failures)
            print(f"EXP004_FORMAL_EVALUATION_FAILED stage={stage} failures={len(failures)}")
            break
        print(f"EXP004_{stage.upper()}_EVALUATION_OK completed={completed} skipped={len(skipped)}", flush=True)
    if stage_failures:
        return 1
    print(f"EXP004_FORMAL_EVALUATION_OK counts={stage_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
