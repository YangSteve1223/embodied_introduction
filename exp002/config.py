"""Small, dependency-light configuration loader for EXP-002."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    condition: str
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
    learning_rate: float
    entropy_coefficient: float
    num_minibatches: int
    update_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float | None
    seed: int
    noise_enabled: bool
    obs_sigma: float
    obs_clip: float
    action_sigma: float
    action_clip: float
    state_layout: str
    qpos_sigma: float
    qpos_clip: float
    qvel_sigma: float
    qvel_clip: float
    position_sigma: float
    position_clip: float
    orientation_sigma_deg: float
    orientation_clip_deg: float


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
        condition=str(raw.get("condition", "unspecified")),
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
        learning_rate=float(ppo["learning_rate"]),
        entropy_coefficient=float(ppo["entropy_coefficient"]),
        num_minibatches=int(ppo["num_minibatches"]),
        update_epochs=int(ppo["update_epochs"]),
        gamma=float(ppo.get("gamma", 0.8)),
        gae_lambda=float(ppo.get("gae_lambda", 0.9)),
        clip_coef=float(ppo.get("clip_coef", 0.2)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        target_kl=(
            None if ppo.get("target_kl") is None else float(ppo.get("target_kl"))
        ),
        seed=int(raw["seed"]),
        noise_enabled=bool(noise["enabled"]),
        obs_sigma=float(noise["obs_sigma"]),
        obs_clip=float(noise["obs_clip"]),
        action_sigma=float(noise["action_sigma"]),
        action_clip=float(noise["action_clip"]),
        state_layout=str(noise.get("state_layout", "peg_insertion_state_v1")),
        qpos_sigma=float(noise.get("qpos_sigma", 0.003)),
        qpos_clip=float(noise.get("qpos_clip", 0.01)),
        qvel_sigma=float(noise.get("qvel_sigma", 0.01)),
        qvel_clip=float(noise.get("qvel_clip", 0.03)),
        position_sigma=float(noise.get("position_sigma", 0.002)),
        position_clip=float(noise.get("position_clip", 0.006)),
        orientation_sigma_deg=float(noise.get("orientation_sigma_deg", 0.5)),
        orientation_clip_deg=float(noise.get("orientation_clip_deg", 1.5)),
    )
