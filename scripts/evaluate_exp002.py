#!/usr/bin/env python3
"""Evaluate an EXP-002 checkpoint under clean or interface-noisy conditions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp002.config import ExperimentConfig, load_experiment_config
from exp002.noise import (
    InterfaceNoiseConfig,
    PEG_INSERTION_STATE_LAYOUT,
    add_action_noise,
    add_peg_insertion_observation_noise,
)
from exp002.ppo import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-noise", action="store_true")
    return parser.parse_args()


def make_noise_config(config: ExperimentConfig) -> InterfaceNoiseConfig:
    if config.state_layout != "peg_insertion_state_v1":
        raise ValueError(f"unsupported EXP-002 state layout: {config.state_layout}")
    return InterfaceNoiseConfig(
        enabled=True,
        obs_sigma=config.obs_sigma,
        obs_clip=config.obs_clip,
        action_sigma=config.action_sigma,
        action_clip=config.action_clip,
        state_layout=config.state_layout,
        qpos_sigma=config.qpos_sigma,
        qpos_clip=config.qpos_clip,
        qvel_sigma=config.qvel_sigma,
        qvel_clip=config.qvel_clip,
        position_sigma=config.position_sigma,
        position_clip=config.position_clip,
        orientation_sigma_deg=config.orientation_sigma_deg,
        orientation_clip_deg=config.orientation_clip_deg,
    )


def scalar_at(value: Any, index: int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value[index].detach().float().cpu())
    return float(value[index])


def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    config = load_experiment_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured EXP-002 evaluation")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["agent"] if isinstance(checkpoint, dict) and "agent" in checkpoint else checkpoint

    import exp002.headless_compat  # noqa: F401
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers ManiSkill tasks
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = None
    try:
        env = gym.make(
            config.task,
            num_envs=args.num_envs,
            obs_mode=config.obs_mode,
            control_mode=config.control_mode,
            robot_uids=config.robot_uids,
            sim_backend=config.sim_backend,
            render_backend=config.render_backend,
            render_mode=None,
            reward_mode=config.reward_mode,
            reconfiguration_freq=1,
        )
        env = ManiSkillVectorEnv(
            env,
            args.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        if env.single_observation_space.shape != (PEG_INSERTION_STATE_LAYOUT.total_dim,):
            raise ValueError(f"unexpected observation shape: {env.single_observation_space.shape}")
        if env.single_action_space.shape != (7,):
            raise ValueError(f"unexpected action shape: {env.single_action_space.shape}")

        agent = Agent(43, 7).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        clean_obs, _ = env.reset(seed=args.seed)
        if not isinstance(clean_obs, torch.Tensor) or clean_obs.device != device:
            raise RuntimeError(f"unexpected observation device: {getattr(clean_obs, 'device', None)}")

        noise = make_noise_config(config)
        obs_generator = None
        action_generator = None
        if args.eval_noise:
            obs_generator = torch.Generator(device=device).manual_seed(args.seed + 101)
            action_generator = torch.Generator(device=device).manual_seed(args.seed + 202)
        action_low = torch.as_tensor(env.single_action_space.low, device=device)
        action_high = torch.as_tensor(env.single_action_space.high, device=device)

        rows: list[dict[str, float]] = []
        max_steps = max(100, ((args.episodes + args.num_envs - 1) // args.num_envs) * 100 * 4)
        for step in range(max_steps):
            if len(rows) >= args.episodes:
                break
            policy_obs = clean_obs
            if args.eval_noise:
                policy_obs = add_peg_insertion_observation_noise(
                    clean_obs,
                    noise,
                    layout=PEG_INSERTION_STATE_LAYOUT,
                    generator=obs_generator,
                )
            with torch.no_grad():
                raw_action = agent.get_action(policy_obs, deterministic=True)
            if args.eval_noise:
                applied_action = add_action_noise(
                    raw_action, noise, generator=action_generator
                )
            else:
                applied_action = raw_action
            applied_action = torch.clamp(applied_action, action_low, action_high)
            clean_obs, _, _, _, infos = env.step(applied_action)

            if "final_info" not in infos:
                continue
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            episode = final_info.get("episode", {})
            indices = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
            for index in indices:
                row = {key: scalar_at(value, index) for key, value in episode.items()}
                rows.append(row)
                if len(rows) >= args.episodes:
                    break

        if len(rows) < args.episodes:
            raise RuntimeError(
                f"only collected {len(rows)} episodes within {max_steps} vector steps"
            )
        metric_keys = sorted({key for row in rows for key in row})
        required = {"success_once", "success_at_end"}
        missing = required - set(metric_keys)
        if missing:
            raise RuntimeError(f"evaluation metrics missing required keys: {sorted(missing)}")
        summary = {
            "experiment": asdict(config),
            "checkpoint": str(args.checkpoint),
            "eval_noise": args.eval_noise,
            "seed": args.seed,
            "episodes_requested": args.episodes,
            "episodes_collected": len(rows),
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
            "episodes": rows,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"episodes={len(rows)} metrics={metric_keys}", flush=True)
        print(f"EXP002_EVAL_OK output={output_path}", flush=True)
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
