#!/usr/bin/env python3
"""One combined bounded EXP-004 smoke gate; never produces formal cells."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from exp004.config import ARMS, load_config

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--exp002-runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=ROOT, type=Path)
    parser.add_argument("--gpus", nargs=4, default=(0, 1, 2, 3), type=int)
    parser.add_argument("--timeout-seconds", default=1800, type=int)
    return parser.parse_args()


def run(command: list[str], name: str, gpu: int | None, args: argparse.Namespace) -> dict[str, Any]:
    log = args.output_root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds, check=False)
    return {"name": name, "gpu": gpu, "returncode": result.returncode, "log": str(log)}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if len(set(args.gpus)) != 4:
        raise ValueError("smoke requires four unique GPUs")
    for path in (args.config, args.python, args.repo_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"smoke output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    test = run([str(args.python), "-m", "unittest", "discover", "-s", "exp004/tests", "-v"], "unit_tests", None, args)
    results.append(test)
    if test["returncode"] != 0:
        print("EXP004_SMOKE_FAILED stage=unit_tests")
        return 1

    source_root = args.exp002_runs_root
    jobs = []
    for arm, gpu in zip(ARMS, args.gpus):
        source = source_root / "clean_seed1002" / "final_ckpt.pt"
        out = args.output_root / "training" / f"{arm}_seed1002"
        jobs.append((arm, gpu, [
            str(args.python), "-m", "exp004.continue_ppo",
            "--config", str(args.config), "--source-checkpoint", str(source),
            "--output-dir", str(out), "--arm", arm, "--seed", "1002",
            "--num-envs", "16", "--num-steps", "5", "--additional-timesteps", "1600", "--max-updates", "2",
        ]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run, command, f"train_{arm}_seed1002", gpu, args): (arm, gpu) for arm, gpu, command in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"smoke training arm={futures[future][0]} returncode={result['returncode']}", flush=True)
    if any(result["returncode"] != 0 for result in results):
        print("EXP004_SMOKE_FAILED stage=training")
        return 1

    eval_jobs = []
    for arm, gpu in zip(ARMS, args.gpus):
        checkpoint = args.output_root / "training" / f"{arm}_seed1002" / "final_ckpt.pt"
        for condition, multiplier in (("clean", "0"), ("combined4x", "4")):
            output = args.output_root / "evaluation" / arm / f"eval_{condition}.json"
            eval_jobs.append((arm, condition, gpu, [
                str(args.python), "-m", "exp004.evaluate",
                "--config", str(args.config), "--checkpoint", str(checkpoint),
                "--output-json", str(output), "--stage", "primary", "--model", arm,
                "--checkpoint-update", "2", "--condition", condition,
                "--multiplier", multiplier, "--seed", "20260920",
                "--expected-training-seed", "1002", "--episodes", "8", "--num-envs", "4",
            ]))
    for start in range(0, len(eval_jobs), 4):
        wave = eval_jobs[start:start + 4]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(run, command, f"eval_{arm}_{condition}", gpu, args): (arm, condition) for arm, condition, gpu, command in wave}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"smoke eval arm={futures[future][0]} condition={futures[future][1]} returncode={result['returncode']}", flush=True)
        if any(result["returncode"] != 0 for result in results[-len(wave):]):
            print("EXP004_SMOKE_FAILED stage=evaluation")
            return 1

    manifest = {"experiment_id": "EXP-004", "stage": "combined_smoke", "smoke": True, "source_seed": 1002, "arms": list(ARMS), "executed_updates": 2, "formal_included": False, "results": results}
    (args.output_root / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("EXP004_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
