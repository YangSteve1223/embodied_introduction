#!/usr/bin/env python3
"""One bounded smoke gate covering every EXP-003 execution path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--gpus", nargs=4, default=(0, 1, 2, 3), type=int)
    parser.add_argument("--timeout-seconds", default=900, type=int)
    return parser.parse_args()


def run_command(
    name: str,
    command: list[str],
    *,
    gpu: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    log_path = args.output_root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
        "name": name,
        "gpu": gpu,
        "returncode": completed.returncode,
        "log": str(log_path),
    }


def run_wave(
    jobs: list[tuple[str, list[str], int | None]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(run_command, name, command, gpu=gpu, args=args): name
            for name, command, gpu in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"smoke name={result['name']} returncode={result['returncode']} "
                f"log={result['log']}",
                flush=True,
            )
    return results


def evaluation_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    output: Path,
    component: str,
    multiplier: float,
) -> list[str]:
    return [
        str(args.python),
        "-m",
        "exp003.evaluate",
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--output-json",
        str(output),
        "--component",
        component,
        "--multiplier",
        str(multiplier),
        "--episodes",
        "8",
        "--num-envs",
        "4",
        "--seed",
        "20260920",
        "--expected-training-seed",
        "1002",
    ]


def continuation_command(
    args: argparse.Namespace,
    *,
    condition: str,
    output: Path,
) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "exp003.continue_ppo",
        "--config",
        str(args.config),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--output-dir",
        str(output),
        "--condition",
        condition,
        "--seed",
        "1002",
        "--num-envs",
        "16",
        "--num-steps",
        "5",
        "--additional-timesteps",
        "800",
        "--max-updates",
        "2",
    ]
    if condition == "treatment":
        command.extend(("--target-multiplier", "1"))
    return command


def main() -> int:
    args = parse_args()
    for path in (args.config, args.source_checkpoint, args.python, args.repo_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(set(args.gpus)) != 4:
        raise ValueError("smoke requires four unique GPU ids")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"smoke output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    unit_test = run_command(
        "unit_tests",
        [
            str(args.python),
            "-m",
            "unittest",
            "discover",
            "-s",
            "exp003/tests",
            "-v",
        ],
        gpu=None,
        args=args,
    )
    results = [unit_test]
    if unit_test["returncode"] != 0:
        print("EXP003_SMOKE_FAILED stage=unit_tests")
        return 1

    control_dir = args.output_root / "control_seed1002"
    treatment_dir = args.output_root / "treatment_seed1002"
    source_eval_dir = args.output_root / "source_evaluation"
    wave_one = [
        (
            "continue_control",
            continuation_command(args, condition="control", output=control_dir),
            args.gpus[0],
        ),
        (
            "continue_treatment",
            continuation_command(args, condition="treatment", output=treatment_dir),
            args.gpus[1],
        ),
        (
            "source_eval_clean",
            evaluation_command(
                args,
                checkpoint=args.source_checkpoint,
                output=source_eval_dir / "eval_clean.json",
                component="none",
                multiplier=0.0,
            ),
            args.gpus[2],
        ),
        (
            "source_eval_noisy",
            evaluation_command(
                args,
                checkpoint=args.source_checkpoint,
                output=source_eval_dir / "eval_noisy.json",
                component="combined",
                multiplier=1.0,
            ),
            args.gpus[3],
        ),
    ]
    wave_one_results = run_wave(wave_one, args)
    results.extend(wave_one_results)
    if any(result["returncode"] != 0 for result in wave_one_results):
        print("EXP003_SMOKE_FAILED stage=wave_one")
        return 1

    control_checkpoint = control_dir / "final_ckpt.pt"
    treatment_checkpoint = treatment_dir / "final_ckpt.pt"
    wave_two = []
    for gpu, model, checkpoint, component, multiplier in (
        (args.gpus[0], "control", control_checkpoint, "none", 0.0),
        (args.gpus[1], "control", control_checkpoint, "combined", 1.0),
        (args.gpus[2], "treatment", treatment_checkpoint, "none", 0.0),
        (args.gpus[3], "treatment", treatment_checkpoint, "combined", 1.0),
    ):
        eval_condition = "clean" if component == "none" else "noisy"
        output = args.output_root / f"{model}_evaluation" / f"eval_{eval_condition}.json"
        wave_two.append(
            (
                f"{model}_eval_{eval_condition}",
                evaluation_command(
                    args,
                    checkpoint=checkpoint,
                    output=output,
                    component=component,
                    multiplier=multiplier,
                ),
                gpu,
            )
        )
    wave_two_results = run_wave(wave_two, args)
    results.extend(wave_two_results)
    manifest = {
        "experiment_id": "EXP-003",
        "stage": "combined_smoke",
        "source_checkpoint": str(args.source_checkpoint),
        "results": results,
    }
    (args.output_root / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if any(result["returncode"] != 0 for result in wave_two_results):
        print("EXP003_SMOKE_FAILED stage=wave_two")
        return 1
    print("EXP003_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
