"""Small, dependency-light configuration loader for EXP-002."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    task: str
    robot_uids: str
    obs_mode: str
    control_mode: str
    sim_backend: str
    render_backend: str
    reward_mode: str
    num_envs: int
    num_steps: int
    total_timesteps: int
    entropy_coefficient: float
    seed: int
    noise_enabled: bool
    obs_sigma: float
    obs_clip: float
    action_sigma: float
    action_clip: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML only when the server environment provides PyYAML."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised on misconfigured hosts
        raise RuntimeError("PyYAML is required to load EXP-002 configs") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    noise = raw.get("noise", {})
    ppo = raw.get("ppo", {})
    env = raw.get("environment", {})
    return ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        task=str(env["task"]),
        robot_uids=str(env["robot_uids"]),
        obs_mode=str(env["obs_mode"]),
        control_mode=str(env["control_mode"]),
        sim_backend=str(env["sim_backend"]),
        render_backend=str(env["render_backend"]),
        reward_mode=str(env["reward_mode"]),
        num_envs=int(env["num_envs"]),
        num_steps=int(ppo["num_steps"]),
        total_timesteps=int(ppo["total_timesteps"]),
        entropy_coefficient=float(ppo["entropy_coefficient"]),
        seed=int(raw["seed"]),
        noise_enabled=bool(noise["enabled"]),
        obs_sigma=float(noise["obs_sigma"]),
        obs_clip=float(noise["obs_clip"]),
        action_sigma=float(noise["action_sigma"]),
        action_clip=float(noise["action_clip"]),
    )
