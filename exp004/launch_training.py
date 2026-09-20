#!/usr/bin/env python3
"""Launch the 12-run EXP-004 continuation matrix in three GPU waves."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from exp004.config import ARMS, TRAINING_SEEDS, load_config, training_matrix

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--exp002-runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=ROOT, type=Path)
    parser.add_argument("--gpus", nargs=4, default=(0, 1, 2, 3), type=int)
    parser.add_argument("--timeout-seconds", default=14400, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_job(job: dict[str, Any], gpu: int, args: argparse.Namespace) -> dict[str, Any]:
    out: Path = job["output_dir"]
    log = args.output_root / "logs" / f"train_{job['arm']}_seed{job['training_seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), "-m", "exp004.continue_ppo",
        "--config", str(args.config), "--source-checkpoint", str(job["source_checkpoint"]),
        "--output-dir", str(out), "--arm", job["arm"], "--seed", str(job["training_seed"]),
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
    if tuple(args.gpus) != (0, 1, 2, 3) and len(set(args.gpus)) != 4:
        raise ValueError("training requires four unique GPU ids")
    for path in (args.config, args.python, args.repo_root):
        if not path.exists():
            raise FileNotFoundError(path)
    jobs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in training_matrix(config):
        seed, arm = row["training_seed"], row["arm"]
        source = args.exp002_runs_root / f"clean_seed{seed}" / "final_ckpt.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        out = args.output_root / f"{arm}_seed{seed}"
        final = out / "final_ckpt.pt"
        if final.exists() and args.resume:
            skipped.append(str(final))
            continue
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"non-empty incomplete run requires review: {out}")
        jobs.append({"training_seed": seed, "arm": arm, "source_checkpoint": source, "output_dir": out})
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "training_launcher_manifest.json"
    manifest = {
        "experiment_id": "EXP-004", "stage": "formal_training", "smoke": False,
        "config": str(args.config), "exp002_runs_root": str(args.exp002_runs_root),
        "output_root": str(args.output_root), "gpus": list(args.gpus),
        "jobs_planned": len(jobs), "jobs_skipped": len(skipped), "wave_gpu_rotation": [[args.gpus[(i + j) % 4] for i in range(4)] for j in range(3)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    failures: list[dict[str, Any]] = []
    completed = 0
    wave_size = 4
    for wave_start in range(0, len(jobs), wave_size):
        wave = jobs[wave_start:wave_start + wave_size]
        wave_index = wave_start // wave_size
        gpu_order = [args.gpus[(wave_index + i) % 4] for i in range(4)]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(run_job, job, gpu_order[i], args): job for i, job in enumerate(wave)}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**job, "returncode": -1, "error": repr(exc)}
                completed += 1
                if result["returncode"] != 0:
                    failures.append(result)
                print(f"training {completed}/{len(jobs)} arm={job['arm']} seed={job['training_seed']} returncode={result['returncode']}", flush=True)
        if failures:
            break
    manifest.update({"jobs_completed": completed, "failures": failures})
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if failures:
        print(f"EXP004_FORMAL_TRAINING_FAILED failures={len(failures)}")
        return 1
    print(f"EXP004_FORMAL_TRAINING_OK completed={completed} skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
