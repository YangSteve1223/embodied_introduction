"""Checkpoint validation and reproducibility metadata for EXP-003."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from exp003.config import Exp003Config


REQUIRED_SOURCE_KEYS = {
    "agent",
    "optimizer",
    "global_step",
    "iteration",
    "seed",
    "config",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    config: Exp003Config,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("EXP-003 requires an EXP-002 checkpoint bundle")
    missing = REQUIRED_SOURCE_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"source checkpoint is missing keys: {sorted(missing)}")

    source = checkpoint["config"]
    expected_values = {
        "task": config.environment.task,
        "robot_uids": config.environment.robot_uids,
        "obs_mode": config.environment.obs_mode,
        "control_mode": config.environment.control_mode,
        "sim_backend": config.environment.sim_backend,
        "render_backend": config.environment.render_backend,
        "reward_mode": config.environment.reward_mode,
        "num_steps": config.ppo.num_steps,
        "learning_rate": config.ppo.learning_rate,
        "entropy_coefficient": config.ppo.entropy_coefficient,
        "num_minibatches": config.ppo.num_minibatches,
        "update_epochs": config.ppo.update_epochs,
        "gamma": config.ppo.gamma,
        "gae_lambda": config.ppo.gae_lambda,
        "clip_coef": config.ppo.clip_coef,
        "vf_coef": config.ppo.vf_coef,
        "max_grad_norm": config.ppo.max_grad_norm,
        "target_kl": config.ppo.target_kl,
    }
    mismatches = {
        key: {"checkpoint": source.get(key), "expected": expected}
        for key, expected in expected_values.items()
        if source.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source checkpoint/config mismatch: {mismatches}")
    if source.get("condition") != "clean" or source.get("noise_enabled") is not False:
        raise ValueError("EXP-003 must start from an EXP-002 clean checkpoint")
    if int(checkpoint["global_step"]) != 102_400_000:
        raise ValueError("source checkpoint must be the formal 102.4M-step checkpoint")
    if int(checkpoint["iteration"]) != 500:
        raise ValueError("source checkpoint must be the formal iteration-500 checkpoint")
    if expected_seed is not None and int(checkpoint["seed"]) != expected_seed:
        raise ValueError(
            f"source seed {checkpoint['seed']} does not match requested seed {expected_seed}"
        )
    return checkpoint


def load_policy_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    config: Exp003Config,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Load either an EXP-002 clean source or an EXP-003 continuation."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("policy checkpoint must be a checkpoint bundle")
    if checkpoint.get("experiment_id") != "EXP-003":
        return load_source_checkpoint(
            path,
            device=device,
            config=config,
            expected_seed=expected_seed,
        )

    required = {
        "agent",
        "optimizer",
        "seed",
        "condition",
        "global_step",
        "continuation_step",
        "continuation_update",
        "config",
        "source_checkpoint_sha256",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"EXP-003 checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["condition"] not in ("control", "treatment"):
        raise ValueError("invalid EXP-003 checkpoint condition")
    if expected_seed is not None and int(checkpoint["seed"]) != expected_seed:
        raise ValueError(
            f"checkpoint seed {checkpoint['seed']} does not match {expected_seed}"
        )
    snapshot = checkpoint["config"]
    if snapshot.get("environment") != asdict(config.environment):
        raise ValueError("EXP-003 checkpoint environment does not match evaluation config")
    if snapshot.get("ppo") != asdict(config.ppo):
        raise ValueError("EXP-003 checkpoint PPO settings do not match evaluation config")
    return checkpoint


def capture_rng_state(
    obs_generator: torch.Generator,
    action_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "observation_noise": obs_generator.get_state(),
        "action_noise": action_generator.get_state(),
    }


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
