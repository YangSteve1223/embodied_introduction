#!/usr/bin/env python3
"""Evaluate EXP-002/003 checkpoints under a scaled EXP-003 noise condition."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp002.noise import PEG_INSERTION_STATE_LAYOUT
from exp002.ppo import Agent
from exp003.checkpoint import load_policy_checkpoint, sha256_file
from exp003.config import load_config
from exp003.noise import (
    NOISE_COMPONENTS,
    apply_action_treatment,
    apply_observation_treatment,
    build_treatment,
    observation_noise_statistics,
    treatment_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--component", choices=NOISE_COMPONENTS, required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-training-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_at(value: Any, index: int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value[index].detach().float().cpu())
    return float(value[index])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    episodes = args.episodes or config.evaluation.episodes
    num_envs = args.num_envs or config.evaluation.num_envs
    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"evaluation output already exists: {output_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-003 evaluation")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_policy_checkpoint(
        checkpoint_path,
        device=device,
        config=config,
        expected_seed=args.expected_training_seed,
    )
    treatment = build_treatment(config.noise, args.component, args.multiplier)
    seed_everything(args.seed)

    import exp002.headless_compat  # noqa: F401
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = None
    try:
        env = gym.make(
            config.environment.task,
            num_envs=num_envs,
            obs_mode=config.environment.obs_mode,
            control_mode=config.environment.control_mode,
            robot_uids=config.environment.robot_uids,
            sim_backend=config.environment.sim_backend,
            render_backend=config.environment.render_backend,
            render_mode=None,
            reward_mode=config.environment.reward_mode,
            reconfiguration_freq=1,
        )
        env = ManiSkillVectorEnv(
            env,
            num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        if env.single_observation_space.shape != (
            PEG_INSERTION_STATE_LAYOUT.total_dim,
        ):
            raise ValueError(
                f"unexpected observation shape: {env.single_observation_space.shape}"
            )
        if env.single_action_space.shape != (7,):
            raise ValueError(f"unexpected action shape: {env.single_action_space.shape}")

        agent = Agent(43, 7).to(device)
        agent.load_state_dict(checkpoint["agent"])
        agent.eval()
        clean_obs, _ = env.reset(seed=args.seed)
        if not isinstance(clean_obs, torch.Tensor) or clean_obs.device != device:
            raise RuntimeError(
                f"unexpected observation device: {getattr(clean_obs, 'device', None)}"
            )

        obs_seed = args.seed + 100_003
        action_seed = args.seed + 200_003
        obs_generator = torch.Generator(device=device).manual_seed(obs_seed)
        action_generator = torch.Generator(device=device).manual_seed(action_seed)
        action_low = torch.as_tensor(env.single_action_space.low, device=device)
        action_high = torch.as_tensor(env.single_action_space.high, device=device)
        rows: list[dict[str, float]] = []
        interface_sums: dict[str, float] = {}
        interface_steps = 0
        max_steps = max(100, ((episodes + num_envs - 1) // num_envs) * 100 * 4)

        for step in range(max_steps):
            if len(rows) >= episodes:
                break
            policy_obs = apply_observation_treatment(
                clean_obs,
                treatment,
                generator=obs_generator,
            )
            obs_stats = observation_noise_statistics(clean_obs, policy_obs)
            with torch.no_grad():
                raw_action = agent.get_action(policy_obs, deterministic=True)
            action_trace = apply_action_treatment(
                raw_action,
                treatment,
                action_low,
                action_high,
                generator=action_generator,
            )

            step_stats = {
                **obs_stats,
                "action_noise_abs_mean": action_trace.noise_delta.abs().mean(),
                "action_applied_delta_abs_mean": (
                    action_trace.applied - raw_action
                ).abs().mean(),
                "raw_policy_oob_fraction": action_trace.raw_oob_fraction,
                "noise_induced_clip_fraction": (
                    action_trace.noise_induced_clip_fraction
                ),
                "final_action_clip_fraction": action_trace.final_clip_fraction,
            }
            for key, value in step_stats.items():
                interface_sums[key] = interface_sums.get(key, 0.0) + float(
                    value.detach().float().cpu()
                )
            interface_steps += 1
            clean_obs, _, _, _, infos = env.step(action_trace.applied)

            if "final_info" not in infos:
                continue
            episode = infos["final_info"].get("episode", {})
            done_mask = infos["_final_info"]
            indices = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
            for index in indices:
                rows.append(
                    {key: scalar_at(value, index) for key, value in episode.items()}
                )
                if len(rows) >= episodes:
                    break

        if len(rows) < episodes:
            raise RuntimeError(
                f"only collected {len(rows)} episodes within {max_steps} vector steps"
            )
        metric_keys = sorted({key for row in rows for key in row})
        required = {"success_once", "success_at_end", "return", "episode_len"}
        missing = required - set(metric_keys)
        if missing:
            raise RuntimeError(f"evaluation metrics missing: {sorted(missing)}")

        summary = {
            "experiment_id": "EXP-003",
            "mode": "policy_evaluation",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_seed": int(checkpoint["seed"]),
            "checkpoint_condition": checkpoint.get("condition", "exp002_clean"),
            "global_step": int(checkpoint["global_step"]),
            "iteration": int(checkpoint["iteration"]),
            "environment": asdict(config.environment),
            "noise": treatment_metadata(treatment),
            "evaluation_seed": args.seed,
            "observation_noise_seed": obs_seed,
            "action_noise_seed": action_seed,
            "episodes_requested": episodes,
            "episodes_collected": len(rows),
            "num_envs": num_envs,
            "vector_steps": step + 1,
            "metric_keys": metric_keys,
            "metrics_mean": {
                key: float(np.mean([row[key] for row in rows if key in row]))
                for key in metric_keys
            },
            "metrics_std": {
                key: float(np.std([row[key] for row in rows if key in row]))
                for key in metric_keys
            },
            "interface_metrics": {
                key: value / interface_steps for key, value in interface_sums.items()
            },
            "episodes": rows,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"EXP003_EVAL_OK episodes={len(rows)} component={args.component} "
            f"multiplier={args.multiplier} output={output_path}",
            flush=True,
        )
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
