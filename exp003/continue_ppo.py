#!/usr/bin/env python3
"""Paired clean/noise-curriculum PPO continuation for EXP-003."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp002.noise import PEG_INSERTION_STATE_LAYOUT
from exp002.ppo import Agent
from exp003.checkpoint import (
    capture_rng_state,
    load_source_checkpoint,
    move_optimizer_state_to_device,
    sha256_file,
)
from exp003.config import load_config, load_selection
from exp003.curriculum import linear_warmup_multiplier, warmup_updates
from exp003.noise import (
    apply_action_treatment,
    apply_observation_treatment,
    build_treatment,
    observation_noise_statistics,
    treatment_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", choices=("control", "treatment"), required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selection-json")
    selection.add_argument("--target-multiplier", type=float)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--additional-timesteps", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def target_multiplier_from_args(args: argparse.Namespace) -> float:
    if args.condition == "control":
        if args.target_multiplier not in (None, 0.0) or args.selection_json:
            raise ValueError("control continuation must not specify a noise target")
        return 0.0
    if args.selection_json:
        return load_selection(args.selection_json)
    if args.target_multiplier is None or args.target_multiplier <= 0:
        raise ValueError(
            "treatment continuation requires --selection-json or a positive "
            "--target-multiplier"
        )
    return float(args.target_multiplier)


def save_checkpoint(
    path: Path,
    *,
    agent: Agent,
    optimizer: optim.Optimizer,
    config_snapshot: dict[str, Any],
    condition: str,
    seed: int,
    source_path: Path,
    source_sha256: str,
    source_global_step: int,
    source_iteration: int,
    global_step: int,
    continuation_step: int,
    continuation_update: int,
    target_multiplier: float,
    current_multiplier: float,
    obs_generator: torch.Generator,
    action_generator: torch.Generator,
) -> None:
    payload = {
        "experiment_id": "EXP-003",
        "condition": condition,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_global_step": source_global_step,
        "source_iteration": source_iteration,
        "global_step": global_step,
        "continuation_step": continuation_step,
        "continuation_update": continuation_update,
        "iteration": source_iteration + continuation_update,
        "target_multiplier": target_multiplier,
        "current_multiplier": current_multiplier,
        "config": config_snapshot,
        "rng_state": capture_rng_state(obs_generator, action_generator),
    }
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    target_multiplier = target_multiplier_from_args(args)
    num_envs = args.num_envs or config.ppo.num_envs
    num_steps = args.num_steps or config.ppo.num_steps
    additional_timesteps = (
        args.additional_timesteps
        or config.continuation.additional_timesteps
    )
    batch_size = num_envs * num_steps
    if batch_size <= 0 or additional_timesteps < batch_size:
        raise ValueError("additional_timesteps must contain at least one batch")
    if additional_timesteps % batch_size:
        raise ValueError("additional_timesteps must be divisible by num_envs*num_steps")
    planned_updates = additional_timesteps // batch_size
    num_updates = planned_updates
    if args.max_updates is not None:
        if args.max_updates < 1:
            raise ValueError("--max-updates must be positive")
        num_updates = min(num_updates, args.max_updates)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-003 continuation")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    source_path = Path(args.source_checkpoint)
    source = load_source_checkpoint(
        source_path,
        device=device,
        config=config,
        expected_seed=args.seed,
    )
    seed = int(source["seed"] if args.seed is None else args.seed)
    source_sha256 = sha256_file(source_path)
    source_global_step = int(source["global_step"])
    source_iteration = int(source["iteration"])

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "experiment_id": config.experiment_id,
        "environment": asdict(config.environment),
        "ppo": asdict(config.ppo),
        "noise": asdict(config.noise),
        "evaluation": asdict(config.evaluation),
        "continuation": asdict(config.continuation),
        "runtime": {
            "condition": args.condition,
            "seed": seed,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "additional_timesteps": additional_timesteps,
            "planned_updates": planned_updates,
            "executed_updates": num_updates,
            "target_multiplier": target_multiplier,
            "device": str(device),
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_sha256,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import exp002.headless_compat  # noqa: F401
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = None
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    try:
        env = gym.make(
            config.environment.task,
            num_envs=num_envs,
            obs_mode=config.environment.obs_mode,
            control_mode=config.environment.control_mode,
            robot_uids=config.environment.robot_uids,
            sim_backend=config.environment.sim_backend,
            render_backend=config.environment.render_backend,
            render_mode=None,
            reward_mode=config.environment.reward_mode,
            reconfiguration_freq=0,
        )
        env = ManiSkillVectorEnv(
            env,
            num_envs,
            ignore_terminations=False,
            record_metrics=True,
        )
        if env.single_observation_space.shape != (
            PEG_INSERTION_STATE_LAYOUT.total_dim,
        ):
            raise ValueError(
                f"unexpected observation shape: {env.single_observation_space.shape}"
            )
        if env.single_action_space.shape != (7,):
            raise ValueError(f"unexpected action shape: {env.single_action_space.shape}")

        observation_dim = 43
        action_dim = 7
        agent = Agent(observation_dim, action_dim).to(device)
        optimizer = optim.Adam(
            agent.parameters(),
            lr=config.ppo.learning_rate,
            eps=1e-5,
        )
        agent.load_state_dict(source["agent"])
        optimizer.load_state_dict(source["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        loaded_lr = float(optimizer.param_groups[0]["lr"])
        if loaded_lr != config.ppo.learning_rate:
            raise ValueError(
                f"optimizer learning rate {loaded_lr} does not match "
                f"{config.ppo.learning_rate}"
            )

        # EXP-002 did not save RNG/environment state. Both paired branches are
        # therefore re-seeded identically at a fresh environment reset.
        seed_everything(seed)
        obs_generator = torch.Generator(device=device).manual_seed(seed + 100_003)
        action_generator = torch.Generator(device=device).manual_seed(seed + 200_003)
        clean_obs, _ = env.reset(seed=seed)
        if not isinstance(clean_obs, torch.Tensor) or clean_obs.device != device:
            raise RuntimeError(
                f"unexpected observation device: {getattr(clean_obs, 'device', None)}"
            )

        obs_buffer = torch.zeros(
            (num_steps, num_envs, observation_dim),
            device=device,
            dtype=torch.float32,
        )
        actions = torch.zeros(
            (num_steps, num_envs, action_dim),
            device=device,
            dtype=torch.float32,
        )
        logprobs = torch.zeros((num_steps, num_envs), device=device)
        rewards = torch.zeros((num_steps, num_envs), device=device)
        dones = torch.zeros((num_steps, num_envs), device=device)
        values = torch.zeros((num_steps, num_envs), device=device)
        action_low = torch.as_tensor(env.single_action_space.low, device=device)
        action_high = torch.as_tensor(env.single_action_space.high, device=device)
        next_done = torch.zeros(num_envs, device=device)
        global_step = source_global_step
        continuation_step = 0
        start_time = time.time()
        last_multiplier = 0.0

        for update in range(1, num_updates + 1):
            if args.condition == "control":
                current_multiplier = 0.0
                component = "none"
            else:
                current_multiplier = linear_warmup_multiplier(
                    update,
                    planned_updates,
                    target_multiplier,
                    config.continuation.warmup_fraction,
                )
                component = "combined"
            treatment = build_treatment(
                config.noise,
                component,
                current_multiplier,
            )
            last_multiplier = current_multiplier
            final_values = torch.zeros((num_steps, num_envs), device=device)
            rollout_stats: dict[str, torch.Tensor] = {}
            agent.eval()
            rollout_start = time.time()

            for step in range(num_steps):
                global_step += num_envs
                continuation_step += num_envs
                policy_obs = apply_observation_treatment(
                    clean_obs,
                    treatment,
                    generator=obs_generator,
                )
                obs_buffer[step] = policy_obs
                dones[step] = next_done
                with torch.no_grad():
                    raw_action, logprob, _, value = agent.get_action_and_value(
                        policy_obs
                    )
                    values[step] = value.flatten()
                actions[step] = raw_action
                logprobs[step] = logprob
                action_trace = apply_action_treatment(
                    raw_action,
                    treatment,
                    action_low,
                    action_high,
                    generator=action_generator,
                )

                step_stats = {
                    **observation_noise_statistics(clean_obs, policy_obs),
                    "action_noise_abs_mean": action_trace.noise_delta.abs().mean(),
                    "action_applied_delta_abs_mean": (
                        action_trace.applied - raw_action
                    ).abs().mean(),
                    "raw_policy_oob_fraction": action_trace.raw_oob_fraction,
                    "noise_induced_clip_fraction": (
                        action_trace.noise_induced_clip_fraction
                    ),
                    "final_action_clip_fraction": action_trace.final_clip_fraction,
                }
                for key, value in step_stats.items():
                    rollout_stats[key] = rollout_stats.get(
                        key, torch.zeros((), device=device)
                    ) + value.detach()

                clean_obs, reward, terminations, truncations, infos = env.step(
                    action_trace.applied
                )
                next_done = torch.logical_or(terminations, truncations).float()
                rewards[step] = reward.reshape(-1)
                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    if "episode" in final_info:
                        for key, value in final_info["episode"].items():
                            writer.add_scalar(
                                f"train/{key}",
                                tensor_scalar(value[done_mask]),
                                global_step,
                            )
                    if "final_observation" in infos:
                        final_obs = apply_observation_treatment(
                            infos["final_observation"][done_mask],
                            treatment,
                            generator=obs_generator,
                        )
                        with torch.no_grad():
                            final_values[step, done_mask] = agent.get_value(
                                final_obs
                            ).view(-1)

            rollout_time = time.time() - rollout_start
            with torch.no_grad():
                next_policy_obs = apply_observation_treatment(
                    clean_obs,
                    treatment,
                    generator=obs_generator,
                )
                next_value = agent.get_value(next_policy_obs).view(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = torch.zeros(num_envs, device=device)
                for step in reversed(range(num_steps)):
                    if step == num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[step + 1]
                        nextvalues = values[step + 1]
                    real_next_values = (
                        next_not_done * nextvalues + final_values[step]
                    )
                    delta = (
                        rewards[step]
                        + config.ppo.gamma * real_next_values
                        - values[step]
                    )
                    lastgaelam = (
                        delta
                        + config.ppo.gamma
                        * config.ppo.gae_lambda
                        * next_not_done
                        * lastgaelam
                    )
                    advantages[step] = lastgaelam
                returns = advantages + values

            b_obs = obs_buffer.reshape((-1, observation_dim))
            b_actions = actions.reshape((-1, action_dim))
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            batch_size_actual = b_obs.shape[0]
            minibatch_size = batch_size_actual // config.ppo.num_minibatches
            if minibatch_size < 1:
                raise ValueError("num_minibatches is larger than the rollout batch")

            agent.train()
            indices = np.arange(batch_size_actual)
            stop_update = False
            for _epoch in range(config.ppo.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size_actual, minibatch_size):
                    mb = indices[start : start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb], b_actions[mb]
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    mb_advantages = b_advantages[mb]
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)
                    policy_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages
                        * torch.clamp(
                            ratio,
                            1.0 - config.ppo.clip_coef,
                            1.0 + config.ppo.clip_coef,
                        ),
                    ).mean()
                    value_loss = 0.5 * (
                        newvalue.view(-1) - b_returns[mb]
                    ).square().mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + config.ppo.vf_coef * value_loss
                        - config.ppo.entropy_coefficient * entropy_mean
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), config.ppo.max_grad_norm
                    )
                    optimizer.step()
                    if (
                        config.ppo.target_kl is not None
                        and approx_kl > config.ppo.target_kl
                    ):
                        stop_update = True
                        break
                if stop_update:
                    break

            elapsed = max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/SPS", continuation_step / elapsed, global_step)
            writer.add_scalar(
                "interface/noise_multiplier", current_multiplier, global_step
            )
            for key, value in rollout_stats.items():
                writer.add_scalar(
                    f"interface/{key}",
                    tensor_scalar(value / num_steps),
                    global_step,
                )
            print(
                f"condition={args.condition} update={update}/{num_updates} "
                f"global_step={global_step} continuation_step={continuation_step} "
                f"multiplier={current_multiplier:.6g} "
                f"rollout_sps={batch_size / max(rollout_time, 1e-6):.1f} "
                f"sps={continuation_step / elapsed:.1f}",
                flush=True,
            )

            if (
                update % config.continuation.checkpoint_interval_updates == 0
                or update == num_updates
            ):
                save_checkpoint(
                    checkpoints_dir / f"checkpoint_update_{update:04d}.pt",
                    agent=agent,
                    optimizer=optimizer,
                    config_snapshot=config_snapshot,
                    condition=args.condition,
                    seed=seed,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    source_global_step=source_global_step,
                    source_iteration=source_iteration,
                    global_step=global_step,
                    continuation_step=continuation_step,
                    continuation_update=update,
                    target_multiplier=target_multiplier,
                    current_multiplier=current_multiplier,
                    obs_generator=obs_generator,
                    action_generator=action_generator,
                )

        final_path = output_dir / "final_ckpt.pt"
        save_checkpoint(
            final_path,
            agent=agent,
            optimizer=optimizer,
            config_snapshot=config_snapshot,
            condition=args.condition,
            seed=seed,
            source_path=source_path,
            source_sha256=source_sha256,
            source_global_step=source_global_step,
            source_iteration=source_iteration,
            global_step=global_step,
            continuation_step=continuation_step,
            continuation_update=num_updates,
            target_multiplier=target_multiplier,
            current_multiplier=last_multiplier,
            obs_generator=obs_generator,
            action_generator=action_generator,
        )
        print(f"EXP003_CONTINUATION_OK checkpoint={final_path}", flush=True)
        return 0
    finally:
        writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
