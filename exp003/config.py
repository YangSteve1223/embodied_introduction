"""Dependency-light JSON configuration for EXP-003."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentConfig:
    task: str
    robot_uids: str
    obs_mode: str
    control_mode: str
    sim_backend: str
    render_backend: str
    reward_mode: str


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int
    num_steps: int
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


@dataclass(frozen=True)
class NoiseBaseConfig:
    qpos_sigma: float
    qpos_clip: float
    qvel_sigma: float
    qvel_clip: float
    position_sigma: float
    position_clip: float
    orientation_sigma_deg: float
    orientation_clip_deg: float
    action_sigma: float
    action_clip: float


@dataclass(frozen=True)
class EvaluationConfig:
    num_envs: int
    episodes: int
    seeds: tuple[int, ...]
    multipliers: tuple[float, ...]
    components: tuple[str, ...]


@dataclass(frozen=True)
class ContinuationConfig:
    additional_timesteps: int
    checkpoint_interval_updates: int
    warmup_fraction: float


@dataclass(frozen=True)
class Exp003Config:
    experiment_id: str
    environment: EnvironmentConfig
    ppo: PPOConfig
    noise: NoiseBaseConfig
    evaluation: EvaluationConfig
    continuation: ContinuationConfig

    @property
    def batch_size(self) -> int:
        return self.ppo.num_envs * self.ppo.num_steps

    @property
    def continuation_updates(self) -> int:
        return self.continuation.additional_timesteps // self.batch_size

    def validate(self) -> None:
        if self.experiment_id != "EXP-003":
            raise ValueError("experiment_id must be EXP-003")
        if self.environment.task != "PegInsertionSide-v1":
            raise ValueError("EXP-003 is registered for PegInsertionSide-v1")
        if self.environment.obs_mode != "state":
            raise ValueError("EXP-003 requires state observations")
        if self.environment.control_mode != "pd_ee_delta_pose":
            raise ValueError("EXP-003 requires pd_ee_delta_pose")
        if self.ppo.num_envs < 1 or self.ppo.num_steps < 1:
            raise ValueError("num_envs and num_steps must be positive")
        if self.continuation.additional_timesteps < self.batch_size:
            raise ValueError("additional_timesteps must contain at least one batch")
        if self.continuation.additional_timesteps % self.batch_size:
            raise ValueError("additional_timesteps must be divisible by the batch size")
        if self.ppo.num_minibatches < 1:
            raise ValueError("num_minibatches must be positive")
        if not 0.0 <= self.continuation.warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be in [0, 1]")
        if self.continuation.checkpoint_interval_updates < 1:
            raise ValueError("checkpoint_interval_updates must be positive")
        if self.evaluation.num_envs < 1 or self.evaluation.episodes < 1:
            raise ValueError("evaluation num_envs and episodes must be positive")
        if not self.evaluation.seeds:
            raise ValueError("at least one evaluation seed is required")
        if any(value < 0 for value in self.evaluation.multipliers):
            raise ValueError("noise multipliers must be non-negative")
        allowed_components = {"none", "observation", "action", "combined"}
        unknown = set(self.evaluation.components) - allowed_components
        if unknown:
            raise ValueError(f"unknown noise components: {sorted(unknown)}")
        for name, value in vars(self.noise).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def load_config(path: str | Path) -> Exp003Config:
    raw = _load_json(path)
    evaluation = raw["evaluation"]
    config = Exp003Config(
        experiment_id=str(raw["experiment_id"]),
        environment=EnvironmentConfig(**raw["environment"]),
        ppo=PPOConfig(**raw["ppo"]),
        noise=NoiseBaseConfig(**raw["noise"]),
        evaluation=EvaluationConfig(
            num_envs=int(evaluation["num_envs"]),
            episodes=int(evaluation["episodes"]),
            seeds=tuple(int(value) for value in evaluation["seeds"]),
            multipliers=tuple(float(value) for value in evaluation["multipliers"]),
            components=tuple(str(value) for value in evaluation["components"]),
        ),
        continuation=ContinuationConfig(**raw["continuation"]),
    )
    config.validate()
    return config


def load_selection(path: str | Path) -> float:
    value = _load_json(path)
    if value.get("status") != "selected":
        raise ValueError(f"calibration selection is not usable: {path}")
    multiplier = float(value["target_multiplier"])
    if multiplier <= 0:
        raise ValueError("selected target_multiplier must be positive")
    if value.get("component") != "combined":
        raise ValueError("formal treatment requires combined noise")
    return multiplier
