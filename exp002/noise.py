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
from typing import Optional

import torch


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

    def validate(self) -> None:
        if self.obs_sigma < 0 or self.obs_clip < 0:
            raise ValueError("observation noise sigma/clip must be non-negative")
        if self.action_sigma < 0 or self.action_clip < 0:
            raise ValueError("action noise sigma/clip must be non-negative")
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
    obs_generator: Optional[torch.Generator] = None,
    action_generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the complete EXP-002 treatment to one policy step."""

    noisy_obs = add_observation_noise(
        obs,
        config,
        scale=obs_scale,
        generator=obs_generator,
    )
    applied_action = add_action_noise(
        action,
        config,
        generator=action_generator,
    )
    return noisy_obs, applied_action
