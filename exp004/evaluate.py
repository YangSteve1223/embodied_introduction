#!/usr/bin/env python3
"""Evaluate one EXP-004 matrix cell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp002.noise import PEG_INSERTION_STATE_LAYOUT
from exp002.ppo import Agent
from exp003.noise import (
    apply_action_treatment,
    apply_observation_treatment,
    build_treatment,
    observation_noise_statistics,
    treatment_metadata,
)
from exp004.checkpoint import load_exp004_checkpoint, sha256_file
from exp004.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stage", choices=("primary", "secondary", "cross"), required=True)
    parser.add_argument("--model", choices=("C0", "C1", "C2", "C3", "C4"), required=True)
    parser.add_argument("--checkpoint-update", type=int, required=True)
    parser.add_argument("--condition", choices=("clean", "combined4x", "combined1x", "combined2x", "combined3x"), required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-training-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def at(value: Any, index: int) -> float:
    return float(value[index].detach().float().cpu() if isinstance(value, torch.Tensor) else value[index])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    episodes = args.episodes or config.evaluation.episodes
    num_envs = args.num_envs or config.evaluation.num_envs
    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    if args.model == "C4" and args.checkpoint_update != 100:
        raise ValueError("C4 is an update-100 reference only")
    if args.model == "C4" and args.stage not in ("primary", "cross"):
        raise ValueError("C4 is included only in primary/cross evaluation")
    if args.condition == "clean" and args.multiplier != 0.0:
        raise ValueError("clean evaluation must use multiplier 0")
    if args.condition != "clean" and args.multiplier <= 0:
        raise ValueError("noisy evaluation requires a positive multiplier")
    out = Path(args.output_json)
    if out.exists():
        raise FileExistsError(f"evaluation output already exists: {out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-004 evaluation")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_exp004_checkpoint(
        checkpoint_path, device=device, config=config,
        expected_seed=args.expected_training_seed, expected_arm=args.model,
        expected_update=args.checkpoint_update,
    )
    component = "none" if args.condition == "clean" else "combined"
    treatment = build_treatment(config.noise, component, args.multiplier)
    seed_everything(args.seed)

    import exp002.headless_compat  # noqa: F401
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = None
    try:
        env = gym.make(
            config.environment.task, num_envs=num_envs,
            obs_mode=config.environment.obs_mode,
            control_mode=config.environment.control_mode,
            robot_uids=config.environment.robot_uids,
            sim_backend=config.environment.sim_backend,
            render_backend=config.environment.render_backend, render_mode=None,
            reward_mode=config.environment.reward_mode, reconfiguration_freq=1,
        )
        env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
        if env.single_observation_space.shape != (43,) or env.single_action_space.shape != (7,):
            raise ValueError("unexpected PegInsertionSide state/action shape")
        agent = Agent(43, 7).to(device)
        agent.load_state_dict(checkpoint["agent"])
        agent.eval()
        clean_obs, _ = env.reset(seed=args.seed)
        if not isinstance(clean_obs, torch.Tensor) or clean_obs.device != device:
            raise RuntimeError("environment observations are not on requested device")
        obs_generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
        action_generator = torch.Generator(device=device).manual_seed(args.seed + 200_003)
        action_low = torch.as_tensor(env.single_action_space.low, device=device)
        action_high = torch.as_tensor(env.single_action_space.high, device=device)
        rows: list[dict[str, float]] = []
        interface_sums: dict[str, float] = {}
        interface_steps = 0
        max_steps = max(100, ((episodes + num_envs - 1) // num_envs) * 100 * 4)
        for vector_step in range(max_steps):
            if len(rows) >= episodes:
                break
            policy_obs = apply_observation_treatment(clean_obs, treatment, generator=obs_generator)
            obs_stats = observation_noise_statistics(clean_obs, policy_obs)
            with torch.no_grad():
                raw_action = agent.get_action(policy_obs, deterministic=True)
            trace = apply_action_treatment(raw_action, treatment, action_low, action_high, generator=action_generator)
            stats = {
                **obs_stats,
                "action_noise_abs_mean": trace.noise_delta.abs().mean(),
                "action_applied_delta_abs_mean": (trace.applied - raw_action).abs().mean(),
                "raw_policy_oob_fraction": trace.raw_oob_fraction,
                "noise_induced_clip_fraction": trace.noise_induced_clip_fraction,
                "final_action_clip_fraction": trace.final_clip_fraction,
            }
            for key, value in stats.items():
                interface_sums[key] = interface_sums.get(key, 0.0) + float(value.detach().float().cpu())
            interface_steps += 1
            clean_obs, _, _, _, infos = env.step(trace.applied)
            if "final_info" not in infos:
                continue
            done_mask = infos["_final_info"]
            episode = infos["final_info"].get("episode", {})
            for index in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
                rows.append({key: at(value, index) for key, value in episode.items()})
                if len(rows) >= episodes:
                    break
        if len(rows) < episodes:
            raise RuntimeError(f"only collected {len(rows)} of {episodes} episodes")
        required = {"success_once", "success_at_end", "return", "episode_len"}
        keys = sorted({key for row in rows for key in row})
        if required - set(keys):
            raise RuntimeError(f"evaluation metrics missing: {sorted(required - set(keys))}")
        summary = {
            "experiment_id": "EXP-004", "stage": args.stage, "smoke": False,
            "model": args.model, "checkpoint_update": args.checkpoint_update,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_experiment_id": checkpoint.get("experiment_id"),
            "training_seed": args.expected_training_seed,
            "checkpoint_arm": checkpoint.get("arm", "C4_reference"),
            "global_step": int(checkpoint["global_step"]),
            "iteration": int(checkpoint.get("iteration", -1)),
            "environment": asdict(config.environment),
            "noise": treatment_metadata(treatment),
            "evaluation_condition": args.condition, "evaluation_multiplier": args.multiplier,
            "evaluation_seed": args.seed,
            "observation_noise_seed": args.seed + 100_003,
            "action_noise_seed": args.seed + 200_003,
            "episodes_requested": episodes, "episodes_collected": len(rows),
            "num_envs": num_envs, "vector_steps": vector_step + 1,
            "metric_keys": keys,
            "metrics_mean": {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys},
            "metrics_std": {key: float(np.std([row[key] for row in rows if key in row])) for key in keys},
            "interface_metrics": {key: value / interface_steps for key, value in interface_sums.items()},
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"EXP004_EVAL_OK stage={args.stage} model={args.model} update={args.checkpoint_update} condition={args.condition} seed={args.seed} output={out}", flush=True)
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
