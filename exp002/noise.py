"""Reproducible interface noise for EXP-002.

The environment remains clean: reward, success, and physics are never changed
by this module. Noise is applied only to the policy observation and to the
action sent to the environment.

Observation noise is expressed in standardized feature units. The caller may
pass a per-feature ``scale`` tensor obtained during preflight. This keeps the
same noise configuration meaningful when the state vector mixes positions,
velocities, orientations, and controller features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class PegInsertionStateLayout:
    """Verified flattened state layout for PegInsertionSide-v1.

    The layout follows ManiSkill's insertion-ordered state dictionary:
    agent proprioception first, then task-specific extra observations.
    """

    qpos: slice = slice(0, 9)
    qvel: slice = slice(9, 18)
    tcp_pose: slice = slice(18, 25)
    peg_pose: slice = slice(25, 32)
    peg_half_size: slice = slice(32, 35)
    box_hole_pose: slice = slice(35, 42)
    box_hole_radius: slice = slice(42, 43)

    @property
    def total_dim(self) -> int:
        return self.box_hole_radius.stop


PEG_INSERTION_STATE_LAYOUT = PegInsertionStateLayout()


@dataclass(frozen=True)
class InterfaceNoiseConfig:
    """Configuration for the single EXP-002 interface-noise treatment."""

    enabled: bool = False
    obs_sigma: float = 0.05
    obs_clip: float = 0.15
    action_sigma: float = 0.02
    action_clip: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0
    state_layout: str = "peg_insertion_state_v1"
    qpos_sigma: float = 0.003
    qpos_clip: float = 0.01
    qvel_sigma: float = 0.01
    qvel_clip: float = 0.03
    position_sigma: float = 0.002
    position_clip: float = 0.006
    orientation_sigma_deg: float = 0.5
    orientation_clip_deg: float = 1.5

    def validate(self) -> None:
        if self.obs_sigma < 0 or self.obs_clip < 0:
            raise ValueError("observation noise sigma/clip must be non-negative")
        if self.action_sigma < 0 or self.action_clip < 0:
            raise ValueError("action noise sigma/clip must be non-negative")
        for name in (
            "qpos_sigma",
            "qpos_clip",
            "qvel_sigma",
            "qvel_clip",
            "position_sigma",
            "position_clip",
            "orientation_sigma_deg",
            "orientation_clip_deg",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be smaller than action_high")


def _normal_like(
    value: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create standard normal noise on the same device and dtype as value."""

    return torch.randn(
        value.shape,
        device=value.device,
        dtype=value.dtype,
        generator=generator,
    )


def _bounded_gaussian(
    value: torch.Tensor,
    sigma: float,
    clip: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create additive Gaussian noise with a per-component hard bound."""

    delta = _normal_like(value, generator=generator) * sigma
    if clip > 0:
        delta = torch.clamp(delta, -clip, clip)
    return delta


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    return quaternion / norm.clamp_min(torch.finfo(quaternion.dtype).eps)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in the [w, x, y, z] convention used by SAPIEN."""

    lw, lx, ly, lz = torch.unbind(left, dim=-1)
    rw, rx, ry, rz = torch.unbind(right, dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _rotation_vector_to_quaternion(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half_angle) / angle,
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half_angle), rotation_vector * scale), dim=-1)


def add_small_angle_quaternion_noise(
    quaternion: torch.Tensor,
    sigma_deg: float,
    clip_deg: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Perturb [w, x, y, z] quaternions with a bounded rotation vector."""

    if sigma_deg < 0 or clip_deg < 0:
        raise ValueError("quaternion noise sigma/clip must be non-negative")
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have a final dimension of 4")
    if sigma_deg == 0:
        return quaternion

    sigma = math.radians(sigma_deg)
    clip = math.radians(clip_deg)
    rotation_vector = _bounded_gaussian(
        quaternion[..., :3], sigma, clip, generator=generator
    )
    delta_quaternion = _rotation_vector_to_quaternion(rotation_vector)
    return _normalize_quaternion(
        _quaternion_multiply(_normalize_quaternion(quaternion), delta_quaternion)
    )


def add_observation_noise(
    obs: torch.Tensor,
    config: InterfaceNoiseConfig,
    *,
    scale: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return a noisy observation without modifying ``obs`` in-place.

    ``scale`` can be a scalar or broadcastable per-feature tensor. A scale of
    one means that ``obs_sigma`` is interpreted directly in observation units.
    The output perturbation is clipped before it is added.
    """

    config.validate()
    if not config.enabled or config.obs_sigma == 0:
        return obs

    if scale is None:
        scale = torch.ones((), device=obs.device, dtype=obs.dtype)
    else:
        scale = torch.as_tensor(scale, device=obs.device, dtype=obs.dtype)
        if torch.any(scale < 0):
            raise ValueError("observation scale must be non-negative")

    delta = _normal_like(obs, generator=generator) * (config.obs_sigma * scale)
    if config.obs_clip > 0:
        delta = torch.clamp(delta, -config.obs_clip * scale, config.obs_clip * scale)
    return obs + delta


def add_peg_insertion_observation_noise(
    obs: torch.Tensor,
    config: InterfaceNoiseConfig,
    *,
    layout: PegInsertionStateLayout = PEG_INSERTION_STATE_LAYOUT,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply field-aware interface noise to the verified 43-D state vector.

    Geometry fields are intentionally copied without noise. Pose quaternions
    are perturbed by composing a small rotation, preserving unit norm.
    """

    config.validate()
    if obs.shape[-1] != layout.total_dim:
        raise ValueError(
            f"expected {layout.total_dim} state features, got {obs.shape[-1]}"
        )
    if not config.enabled:
        return obs

    noisy = obs.clone()
    for field, sigma, clip in (
        (layout.qpos, config.qpos_sigma, config.qpos_clip),
        (layout.qvel, config.qvel_sigma, config.qvel_clip),
    ):
        noisy[..., field] = noisy[..., field] + _bounded_gaussian(
            noisy[..., field], sigma, clip, generator=generator
        )

    for pose_field in (layout.tcp_pose, layout.peg_pose, layout.box_hole_pose):
        noisy[..., pose_field.start : pose_field.start + 3] = (
            noisy[..., pose_field.start : pose_field.start + 3]
            + _bounded_gaussian(
                noisy[..., pose_field.start : pose_field.start + 3],
                config.position_sigma,
                config.position_clip,
                generator=generator,
            )
        )
        noisy[..., pose_field.start + 3 : pose_field.stop] = add_small_angle_quaternion_noise(
            noisy[..., pose_field.start + 3 : pose_field.stop],
            config.orientation_sigma_deg,
            config.orientation_clip_deg,
            generator=generator,
        )
    return noisy


def add_action_noise(
    action: torch.Tensor,
    config: InterfaceNoiseConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return the applied action after bounded noise and controller clipping."""

    config.validate()
    if not config.enabled or config.action_sigma == 0:
        return torch.clamp(action, config.action_low, config.action_high)

    delta = _normal_like(action, generator=generator) * config.action_sigma
    if config.action_clip > 0:
        delta = torch.clamp(delta, -config.action_clip, config.action_clip)
    return torch.clamp(action + delta, config.action_low, config.action_high)


def apply_interface_noise(
    obs: torch.Tensor,
    action: torch.Tensor,
    config: InterfaceNoiseConfig,
    *,
    obs_scale: Optional[torch.Tensor] = None,
    state_layout: Optional[PegInsertionStateLayout] = None,
    obs_generator: Optional[torch.Generator] = None,
    action_generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the complete EXP-002 treatment to one policy step."""

    if state_layout is None:
        noisy_obs = add_observation_noise(
            obs,
            config,
            scale=obs_scale,
            generator=obs_generator,
        )
    else:
        noisy_obs = add_peg_insertion_observation_noise(
            obs,
            config,
            layout=state_layout,
            generator=obs_generator,
        )
    applied_action = add_action_noise(
        action,
        config,
        generator=action_generator,
    )
    return noisy_obs, applied_action
