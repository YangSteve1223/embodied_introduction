"""Checkpoint lineage checks for EXP-004 training and evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from exp003.checkpoint import load_source_checkpoint


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exp004_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    config: Any,
    expected_seed: int,
    expected_arm: str | None = None,
    expected_update: int | None = None,
) -> dict[str, Any]:
    """Load a new EXP-004 endpoint or the restricted EXP-003 C4 reference."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint bundle must be a dict: {path}")
    experiment_id = checkpoint.get("experiment_id")
    if experiment_id == "EXP-004":
        required = {
            "agent", "optimizer", "seed", "arm", "global_step",
            "continuation_update", "source_checkpoint_sha256", "config",
        }
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(f"EXP-004 checkpoint missing keys: {sorted(missing)}")
        if int(checkpoint["seed"]) != expected_seed:
            raise ValueError("checkpoint training seed does not match matrix")
        if expected_arm is not None and checkpoint["arm"] != expected_arm:
            raise ValueError("checkpoint arm does not match matrix")
        update = int(checkpoint["continuation_update"])
        if expected_update is not None and update != expected_update:
            raise ValueError("checkpoint endpoint does not match matrix")
        expected_step = config.source_global_step + update * config.batch_size
        if int(checkpoint["global_step"]) != expected_step:
            raise ValueError("checkpoint global_step does not match continuation update")
        snapshot = checkpoint["config"]
        if snapshot.get("environment") != config.snapshot()["environment"]:
            raise ValueError("checkpoint environment does not match config")
        if snapshot.get("ppo") != config.snapshot()["ppo"]:
            raise ValueError("checkpoint PPO settings do not match config")
        return checkpoint

    if experiment_id == "EXP-003":
        if expected_arm != "C4" or expected_update != 100:
            raise ValueError("EXP-003 checkpoints are allowed only as C4 update-100 reference")
        if checkpoint.get("condition") != "treatment":
            raise ValueError("C4 reference must be the EXP-003 treatment checkpoint")
        if int(checkpoint.get("seed", -1)) != expected_seed:
            raise ValueError("C4 reference seed does not match matrix")
        if int(checkpoint.get("continuation_update", -1)) != 100:
            raise ValueError("C4 reference must be EXP-003 update 100")
        if float(checkpoint.get("target_multiplier", -1)) != 4.0:
            raise ValueError("C4 reference must use combined 4x")
        if int(checkpoint.get("global_step", -1)) != 122_880_000:
            raise ValueError("C4 reference global_step must be 122880000")
        return checkpoint

    raise ValueError(f"unsupported checkpoint experiment_id: {experiment_id!r}")


def load_source(
    path: str | Path,
    *,
    device: torch.device,
    config: Any,
    expected_seed: int,
) -> dict[str, Any]:
    """Reuse the audited EXP-003 source validator for EXP-002 clean bundles."""

    source = load_source_checkpoint(
        path, device=device, config=config, expected_seed=expected_seed
    )
    return source


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
