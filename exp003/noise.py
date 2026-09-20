"""Component selection, scaling, and instrumentation for EXP-003 noise."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch

from exp002.noise import (
    InterfaceNoiseConfig,
    PEG_INSERTION_STATE_LAYOUT,
    add_peg_insertion_observation_noise,
)
from exp003.config import NoiseBaseConfig


NOISE_COMPONENTS = ("none", "observation", "action", "combined")


@dataclass(frozen=True)
class NoiseTreatment:
    component: str
    multiplier: float
    config: InterfaceNoiseConfig

    @property
    def observation_enabled(self) -> bool:
        return self.component in ("observation", "combined") and self.multiplier > 0

    @property
    def action_enabled(self) -> bool:
        return self.component in ("action", "combined") and self.multiplier > 0


@dataclass(frozen=True)
class ActionTrace:
    raw: torch.Tensor
    noise_delta: torch.Tensor
    pre_clamp: torch.Tensor
    applied: torch.Tensor
    raw_oob_fraction: torch.Tensor
    noise_induced_clip_fraction: torch.Tensor
    final_clip_fraction: torch.Tensor


def build_treatment(
    base: NoiseBaseConfig,
    component: str,
    multiplier: float,
) -> NoiseTreatment:
    if component not in NOISE_COMPONENTS:
        raise ValueError(f"unknown noise component: {component}")
    if multiplier < 0:
        raise ValueError("noise multiplier must be non-negative")

    observation_scale = multiplier if component in ("observation", "combined") else 0.0
    action_scale = multiplier if component in ("action", "combined") else 0.0
    config = InterfaceNoiseConfig(
        enabled=observation_scale > 0 or action_scale > 0,
        obs_sigma=0.0,
        obs_clip=0.0,
        action_sigma=base.action_sigma * action_scale,
        action_clip=base.action_clip * action_scale,
        state_layout="peg_insertion_state_v1",
        qpos_sigma=base.qpos_sigma * observation_scale,
        qpos_clip=base.qpos_clip * observation_scale,
        qvel_sigma=base.qvel_sigma * observation_scale,
        qvel_clip=base.qvel_clip * observation_scale,
        position_sigma=base.position_sigma * observation_scale,
        position_clip=base.position_clip * observation_scale,
        orientation_sigma_deg=base.orientation_sigma_deg * observation_scale,
        orientation_clip_deg=base.orientation_clip_deg * observation_scale,
    )
    config.validate()
    return NoiseTreatment(component=component, multiplier=multiplier, config=config)


def apply_observation_treatment(
    clean_obs: torch.Tensor,
    treatment: NoiseTreatment,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not treatment.observation_enabled:
        return clean_obs
    return add_peg_insertion_observation_noise(
        clean_obs,
        treatment.config,
        layout=PEG_INSERTION_STATE_LAYOUT,
        generator=generator,
    )


def apply_action_treatment(
    raw_action: torch.Tensor,
    treatment: NoiseTreatment,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> ActionTrace:
    if treatment.action_enabled:
        delta = torch.randn(
            raw_action.shape,
            device=raw_action.device,
            dtype=raw_action.dtype,
            generator=generator,
        ) * treatment.config.action_sigma
        if treatment.config.action_clip > 0:
            delta = torch.clamp(
                delta,
                -treatment.config.action_clip,
                treatment.config.action_clip,
            )
    else:
        delta = torch.zeros_like(raw_action)

    pre_clamp = raw_action + delta
    applied = torch.clamp(pre_clamp, action_low, action_high)
    raw_oob = (raw_action < action_low) | (raw_action > action_high)
    pre_clamp_oob = (pre_clamp < action_low) | (pre_clamp > action_high)
    noise_induced = (~raw_oob) & pre_clamp_oob
    return ActionTrace(
        raw=raw_action,
        noise_delta=delta,
        pre_clamp=pre_clamp,
        applied=applied,
        raw_oob_fraction=raw_oob.float().mean(),
        noise_induced_clip_fraction=noise_induced.float().mean(),
        final_clip_fraction=pre_clamp_oob.float().mean(),
    )


def observation_noise_statistics(
    clean_obs: torch.Tensor,
    policy_obs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    layout = PEG_INSERTION_STATE_LAYOUT
    delta = policy_obs - clean_obs
    position_parts = []
    orientation_degrees = []
    for pose in (layout.tcp_pose, layout.peg_pose, layout.box_hole_pose):
        position_parts.append(delta[..., pose.start : pose.start + 3].abs())
        clean_q = clean_obs[..., pose.start + 3 : pose.stop]
        noisy_q = policy_obs[..., pose.start + 3 : pose.stop]
        clean_q = clean_q / torch.linalg.vector_norm(
            clean_q, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(clean_q.dtype).eps)
        noisy_q = noisy_q / torch.linalg.vector_norm(
            noisy_q, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(noisy_q.dtype).eps)
        dot = (clean_q * noisy_q).sum(dim=-1).abs().clamp(0.0, 1.0)
        orientation_degrees.append(2.0 * torch.acos(dot) * (180.0 / math.pi))

    return {
        "obs_delta_abs_mean": delta.abs().mean(),
        "qpos_delta_abs_mean": delta[..., layout.qpos].abs().mean(),
        "qvel_delta_abs_mean": delta[..., layout.qvel].abs().mean(),
        "position_delta_abs_mean": torch.cat(position_parts, dim=-1).mean(),
        "orientation_delta_deg_mean": torch.stack(
            orientation_degrees, dim=-1
        ).mean(),
    }


def treatment_metadata(treatment: NoiseTreatment) -> dict[str, Any]:
    config = treatment.config
    return {
        "component": treatment.component,
        "multiplier": treatment.multiplier,
        "qpos_sigma": config.qpos_sigma,
        "qpos_clip": config.qpos_clip,
        "qvel_sigma": config.qvel_sigma,
        "qvel_clip": config.qvel_clip,
        "position_sigma": config.position_sigma,
        "position_clip": config.position_clip,
        "orientation_sigma_deg": config.orientation_sigma_deg,
        "orientation_clip_deg": config.orientation_clip_deg,
        "action_sigma": config.action_sigma,
        "action_clip": config.action_clip,
    }
