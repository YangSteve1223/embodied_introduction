"""Fixed 20-update rollout-level dose schedule for EXP-004."""

from __future__ import annotations


def linear_warmup_multiplier(
    update_index: int, target_multiplier: float, warmup_updates: int = 20
) -> float:
    """Return the dose for a one-based update index.

    Update 1 is clean, update 20 reaches target, and update 21 onward is
    fixed at target.  This is deliberately independent of the 150-update
    total so the registered warmup does not silently change with budget.
    """

    if update_index < 1:
        raise ValueError("update_index must be positive")
    if target_multiplier < 0:
        raise ValueError("target_multiplier must be non-negative")
    if warmup_updates < 1:
        raise ValueError("warmup_updates must be positive")
    if update_index >= warmup_updates:
        return float(target_multiplier)
    if warmup_updates == 1:
        return float(target_multiplier)
    return float(target_multiplier) * (update_index - 1) / (warmup_updates - 1)
