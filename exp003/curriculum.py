"""Deterministic rollout-level noise schedules for EXP-003."""

from __future__ import annotations

import math


def warmup_updates(total_updates: int, warmup_fraction: float) -> int:
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must be in [0, 1]")
    if warmup_fraction == 0:
        return 0
    return max(1, math.ceil(total_updates * warmup_fraction))


def linear_warmup_multiplier(
    update_index: int,
    total_updates: int,
    target_multiplier: float,
    warmup_fraction: float,
) -> float:
    """Return the multiplier for a one-based rollout/update index.

    The first rollout uses zero noise. The final warmup rollout reaches the
    target. For a one-rollout warmup, that rollout uses the target directly.
    """

    if not 1 <= update_index <= total_updates:
        raise ValueError("update_index must be within the continuation")
    if target_multiplier < 0:
        raise ValueError("target_multiplier must be non-negative")
    count = warmup_updates(total_updates, warmup_fraction)
    if count == 0 or update_index > count:
        return target_multiplier
    if count == 1:
        return target_multiplier
    progress = (update_index - 1) / (count - 1)
    return target_multiplier * progress
