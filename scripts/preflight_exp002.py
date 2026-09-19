#!/usr/bin/env python3
"""Read-only interface and resource preflight for EXP-002."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--compat-module",
        default=os.environ.get("EXP002_COMPAT_MODULE"),
        help="optional module that installs the existing headless compatibility patch",
    )
    return parser.parse_args()


def shape_of(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return str(tuple(shape))
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {shape_of(v)}" for k, v in value.items()) + "}"
    return type(value).__name__


def space_summary(space: Any) -> str:
    return (
        f"type={type(space).__name__}, shape={getattr(space, 'shape', None)}, "
        f"dtype={getattr(space, 'dtype', None)}, low={getattr(space, 'low', None)}, "
        f"high={getattr(space, 'high', None)}"
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    args = parse_args()
    if args.num_envs < 1 or args.steps < 1:
        raise SystemExit("--num-envs and --steps must be positive")

    if args.compat_module:
        print(f"compat module: {args.compat_module}")
        __import__(args.compat_module)
    else:
        print("compat module: none (use the existing headless SAPIEN patch if required)")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
    import torch

    print(f"python: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"mani_skill: {package_version('mani_skill')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    env = None
    try:
        env = gym.make(
            args.env_id,
            num_envs=args.num_envs,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            robot_uids="panda_wristcam",
            sim_backend="physx_cuda",
            render_backend="none",
            reward_mode="normalized_dense",
            reconfiguration_freq=0,
        )
        print(f"environment: {args.env_id}")
        print(f"device: {getattr(env, 'device', '<unknown>')}")
        print(f"observation_space: {space_summary(env.observation_space)}")
        print(f"action_space: {space_summary(env.action_space)}")
        print(f"max_episode_steps: {getattr(env, '_max_episode_steps', '<unknown>')}")

        obs, info = env.reset(seed=args.seed)
        print(f"reset observation: {shape_of(obs)}")
        print(f"reset info keys: {sorted(info.keys()) if isinstance(info, dict) else type(info).__name__}")

        for step in range(1, args.steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step} obs={shape_of(obs)} reward={shape_of(reward)} "
                f"terminated={shape_of(terminated)} truncated={shape_of(truncated)}"
            )
            if isinstance(info, dict):
                print(f"info keys: {sorted(info.keys())}")
                if "success" in info:
                    print(f"success shape: {shape_of(info['success'])}")

        print("EXP002_PREFLIGHT_OK")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
