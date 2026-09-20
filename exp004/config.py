"""Dependency-light configuration and registered matrices for EXP-004."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


TRAINING_SEEDS = (1001, 1002, 1003)
ARMS = ("C0", "C1", "C2", "C3")
TREATMENT_MULTIPLIERS = {"C0": 0.0, "C1": 1.0, "C2": 2.0, "C3": 3.0}
EVALUATION_CONDITIONS = ("clean", "combined4x")


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
class NoiseConfig:
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
    primary_multiplier: float
    secondary_multipliers: tuple[float, ...]


@dataclass(frozen=True)
class ContinuationConfig:
    source_global_step: int
    source_iteration: int
    total_updates: int
    additional_timesteps: int
    warmup_updates: int
    checkpoint_interval_updates: int
    primary_update: int
    secondary_update: int


@dataclass(frozen=True)
class Exp004Config:
    experiment_id: str
    environment: EnvironmentConfig
    ppo: PPOConfig
    noise: NoiseConfig
    evaluation: EvaluationConfig
    continuation: ContinuationConfig
    training_seeds: tuple[int, ...]
    arms: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return self.ppo.num_envs * self.ppo.num_steps

    @property
    def source_global_step(self) -> int:
        return self.continuation.source_global_step

    def arm_multiplier(self, arm: str) -> float:
        if arm not in TREATMENT_MULTIPLIERS:
            raise ValueError(f"unknown training arm: {arm}")
        return TREATMENT_MULTIPLIERS[arm]

    def snapshot(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "environment": asdict(self.environment),
            "ppo": asdict(self.ppo),
            "noise": asdict(self.noise),
            "evaluation": asdict(self.evaluation),
            "continuation": asdict(self.continuation),
            "training_seeds": list(self.training_seeds),
            "arms": list(self.arms),
        }

    def validate(self) -> None:
        if self.experiment_id != "EXP-004":
            raise ValueError("experiment_id must be EXP-004")
        if self.environment.task != "PegInsertionSide-v1":
            raise ValueError("EXP-004 is registered for PegInsertionSide-v1")
        if self.environment.obs_mode != "state":
            raise ValueError("EXP-004 requires state observations")
        if self.environment.control_mode != "pd_ee_delta_pose":
            raise ValueError("EXP-004 requires pd_ee_delta_pose")
        if tuple(self.training_seeds) != TRAINING_SEEDS:
            raise ValueError("training seeds must be exactly 1001, 1002, 1003")
        if tuple(self.arms) != ARMS:
            raise ValueError("training arms must be exactly C0, C1, C2, C3")
        if self.ppo.num_envs < 1 or self.ppo.num_steps < 1:
            raise ValueError("num_envs and num_steps must be positive")
        if self.batch_size != 204_800:
            raise ValueError("registered batch size must be 204800")
        cont = self.continuation
        if cont.total_updates != 150 or cont.warmup_updates != 20:
            raise ValueError("registered continuation is 150 updates with 20 warmup updates")
        if cont.additional_timesteps != cont.total_updates * self.batch_size:
            raise ValueError("additional_timesteps must equal 150 rollout batches")
        if cont.source_global_step != 102_400_000 or cont.source_iteration != 500:
            raise ValueError("source checkpoint lineage must be the EXP-002 102.4M checkpoint")
        if cont.primary_update != 100 or cont.secondary_update != 150:
            raise ValueError("endpoints must be update 100 and update 150")
        if not 1 <= cont.warmup_updates <= cont.total_updates:
            raise ValueError("warmup_updates must be within continuation")
        if cont.checkpoint_interval_updates < 1:
            raise ValueError("checkpoint interval must be positive")
        if tuple(self.evaluation.seeds) != (20260920, 20260921):
            raise ValueError("evaluation seeds must be the registered pair")
        if self.evaluation.primary_multiplier != 4.0:
            raise ValueError("primary evaluation must use combined 4x")
        if self.evaluation.num_envs < 1 or self.evaluation.episodes < 1:
            raise ValueError("evaluation size must be positive")
        if tuple(self.evaluation.secondary_multipliers) != (1.0, 2.0, 3.0):
            raise ValueError("secondary multipliers must be 1x, 2x, 3x")
        for name, value in vars(self.noise).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def load_config(path: str | Path) -> Exp004Config:
    raw = _load_json(path)
    evaluation = raw["evaluation"]
    config = Exp004Config(
        experiment_id=str(raw["experiment_id"]),
        environment=EnvironmentConfig(**raw["environment"]),
        ppo=PPOConfig(**raw["ppo"]),
        noise=NoiseConfig(**raw["noise"]),
        evaluation=EvaluationConfig(
            num_envs=int(evaluation["num_envs"]),
            episodes=int(evaluation["episodes"]),
            seeds=tuple(int(x) for x in evaluation["seeds"]),
            primary_multiplier=float(evaluation["primary_multiplier"]),
            secondary_multipliers=tuple(float(x) for x in evaluation["secondary_multipliers"]),
        ),
        continuation=ContinuationConfig(**raw["continuation"]),
        training_seeds=tuple(int(x) for x in raw["training_seeds"]),
        arms=tuple(str(x) for x in raw["arms"]),
    )
    config.validate()
    return config


def training_matrix(config: Exp004Config) -> list[dict[str, Any]]:
    return [
        {"training_seed": seed, "arm": arm, "multiplier": config.arm_multiplier(arm)}
        for seed in config.training_seeds
        for arm in config.arms
    ]


def primary_matrix(config: Exp004Config) -> list[dict[str, Any]]:
    rows = []
    for seed in config.training_seeds:
        for model in (*config.arms, "C4"):
            for condition, multiplier in (("clean", 0.0), ("combined4x", 4.0)):
                for eval_seed in config.evaluation.seeds:
                    rows.append({
                        "training_seed": seed,
                        "model": model,
                        "checkpoint_update": 100,
                        "condition": condition,
                        "multiplier": multiplier,
                        "evaluation_seed": eval_seed,
                    })
    return rows


def secondary_matrix(config: Exp004Config) -> list[dict[str, Any]]:
    rows = []
    for seed in config.training_seeds:
        for arm in config.arms:
            for condition, multiplier in (("clean", 0.0), ("combined4x", 4.0)):
                for eval_seed in config.evaluation.seeds:
                    rows.append({
                        "training_seed": seed,
                        "model": arm,
                        "checkpoint_update": 150,
                        "condition": condition,
                        "multiplier": multiplier,
                        "evaluation_seed": eval_seed,
                    })
    return rows


def cross_matrix(config: Exp004Config) -> list[dict[str, Any]]:
    """Optional cross-severity diagnostic grid; it is not a primary gate."""

    rows = []
    endpoints = [(arm, update) for arm in config.arms for update in (100, 150)]
    endpoints.append(("C4", 100))
    for seed in config.training_seeds:
        for model, update in endpoints:
            for multiplier in config.evaluation.secondary_multipliers:
                for eval_seed in config.evaluation.seeds:
                    rows.append({
                        "training_seed": seed,
                        "model": model,
                        "checkpoint_update": update,
                        "condition": f"combined{int(multiplier)}x",
                        "multiplier": multiplier,
                        "evaluation_seed": eval_seed,
                    })
    return rows


def expected_stage_count(config: Exp004Config, stage: str) -> int:
    if stage == "primary":
        return len(primary_matrix(config))
    if stage == "secondary":
        return len(secondary_matrix(config))
    if stage == "cross":
        return len(cross_matrix(config))
    raise ValueError(f"unknown stage: {stage}")
