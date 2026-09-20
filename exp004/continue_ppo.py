#!/usr/bin/env python3
"""Run one EXP-004 C0/C1/C2/C3 PPO continuation."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp002.noise import PEG_INSERTION_STATE_LAYOUT
from exp002.ppo import Agent
from exp003.noise import (
    apply_action_treatment,
    apply_observation_treatment,
    build_treatment,
    observation_noise_statistics,
)
from exp004.checkpoint import load_source, move_optimizer_state_to_device, sha256_file
from exp004.config import load_config
from exp004.curriculum import linear_warmup_multiplier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=("C0", "C1", "C2", "C3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--additional-timesteps", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def save_checkpoint(path: Path, *, agent: Agent, optimizer: optim.Optimizer,
                    snapshot: dict[str, Any], arm: str, seed: int,
                    source_path: Path, source_sha: str, global_step: int,
                    continuation_step: int, update: int, current_multiplier: float,
                    obs_generator: torch.Generator, action_generator: torch.Generator) -> None:
    payload = {
        "experiment_id": "EXP-004",
        "arm": arm,
        "condition": "clean" if arm == "C0" else "combined",
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_global_step": snapshot["continuation"]["source_global_step"],
        "source_iteration": snapshot["continuation"]["source_iteration"],
        "global_step": global_step,
        "continuation_step": continuation_step,
        "continuation_update": update,
        "iteration": snapshot["continuation"]["source_iteration"] + update,
        "target_multiplier": snapshot["arm_multiplier"],
        "current_multiplier": current_multiplier,
        "config": snapshot,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "observation_noise": obs_generator.get_state(),
            "action_noise": action_generator.get_state(),
        },
    }
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.seed not in config.training_seeds:
        raise ValueError("seed is not in the registered training seed set")
    num_envs = args.num_envs or config.ppo.num_envs
    num_steps = args.num_steps or config.ppo.num_steps
    additional = args.additional_timesteps or config.continuation.additional_timesteps
    batch_size = num_envs * num_steps
    if batch_size < 1 or additional < batch_size or additional % batch_size:
        raise ValueError("additional timesteps must be divisible by num_envs*num_steps")
    planned_updates = additional // batch_size
    updates = min(planned_updates, args.max_updates) if args.max_updates else planned_updates
    if updates < 1:
        raise ValueError("at least one update is required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-004 continuation")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    source_path = Path(args.source_checkpoint)
    source = load_source(source_path, device=device, config=config, expected_seed=args.seed)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    source_sha = sha256_file(source_path)
    target_multiplier = config.arm_multiplier(args.arm)
    snapshot = config.snapshot()
    snapshot.update({
        "runtime": {
            "arm": args.arm, "seed": args.seed, "num_envs": num_envs,
            "num_steps": num_steps, "additional_timesteps": additional,
            "planned_updates": planned_updates, "executed_updates": updates,
            "device": str(device), "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_sha,
        },
        "arm_multiplier": target_multiplier,
    })
    (out / "config.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    import exp002.headless_compat  # noqa: F401
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = None
    writer = SummaryWriter(str(out / "tensorboard"))
    try:
        env = gym.make(
            config.environment.task, num_envs=num_envs,
            obs_mode=config.environment.obs_mode,
            control_mode=config.environment.control_mode,
            robot_uids=config.environment.robot_uids,
            sim_backend=config.environment.sim_backend,
            render_backend=config.environment.render_backend, render_mode=None,
            reward_mode=config.environment.reward_mode, reconfiguration_freq=0,
        )
        env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)
        if env.single_observation_space.shape != (43,) or env.single_action_space.shape != (7,):
            raise ValueError("unexpected PegInsertionSide state/action shape")
        agent = Agent(43, 7).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config.ppo.learning_rate, eps=1e-5)
        agent.load_state_dict(source["agent"])
        optimizer.load_state_dict(source["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        if float(optimizer.param_groups[0]["lr"]) != config.ppo.learning_rate:
            raise ValueError("source optimizer learning rate differs from registered config")

        seed_everything(args.seed)
        obs_generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
        action_generator = torch.Generator(device=device).manual_seed(args.seed + 200_003)
        clean_obs, _ = env.reset(seed=args.seed)
        if not isinstance(clean_obs, torch.Tensor) or clean_obs.device != device:
            raise RuntimeError("environment observations are not on the requested device")
        obs_buffer = torch.zeros((num_steps, num_envs, 43), device=device)
        actions = torch.zeros((num_steps, num_envs, 7), device=device)
        logprobs = torch.zeros((num_steps, num_envs), device=device)
        rewards = torch.zeros((num_steps, num_envs), device=device)
        dones = torch.zeros((num_steps, num_envs), device=device)
        values = torch.zeros((num_steps, num_envs), device=device)
        action_low = torch.as_tensor(env.single_action_space.low, device=device)
        action_high = torch.as_tensor(env.single_action_space.high, device=device)
        next_done = torch.zeros(num_envs, device=device)
        global_step = int(source["global_step"])
        continuation_step = 0
        start_time = time.time()
        final_multiplier = 0.0

        for update in range(1, updates + 1):
            target = target_multiplier
            current = 0.0 if args.arm == "C0" else linear_warmup_multiplier(
                update, target, config.continuation.warmup_updates
            )
            treatment = build_treatment(config.noise, "none" if args.arm == "C0" else "combined", current)
            final_values = torch.zeros((num_steps, num_envs), device=device)
            rollout_stats: dict[str, torch.Tensor] = {}
            rollout_start = time.time()
            agent.eval()
            for step in range(num_steps):
                global_step += num_envs
                continuation_step += num_envs
                policy_obs = apply_observation_treatment(clean_obs, treatment, generator=obs_generator)
                obs_buffer[step] = policy_obs
                dones[step] = next_done
                with torch.no_grad():
                    raw_action, logprob, _, value = agent.get_action_and_value(policy_obs)
                    values[step] = value.flatten()
                actions[step] = raw_action
                logprobs[step] = logprob
                trace = apply_action_treatment(raw_action, treatment, action_low, action_high, generator=action_generator)
                stats = {
                    **observation_noise_statistics(clean_obs, policy_obs),
                    "action_noise_abs_mean": trace.noise_delta.abs().mean(),
                    "action_applied_delta_abs_mean": (trace.applied - raw_action).abs().mean(),
                    "raw_policy_oob_fraction": trace.raw_oob_fraction,
                    "noise_induced_clip_fraction": trace.noise_induced_clip_fraction,
                    "final_action_clip_fraction": trace.final_clip_fraction,
                }
                for key, value in stats.items():
                    rollout_stats[key] = rollout_stats.get(key, torch.zeros((), device=device)) + value.detach()
                clean_obs, reward, terminations, truncations, infos = env.step(trace.applied)
                next_done = torch.logical_or(terminations, truncations).float()
                rewards[step] = reward.reshape(-1)
                if "final_info" in infos and "final_observation" in infos:
                    done_mask = infos["_final_info"]
                    final_obs = apply_observation_treatment(infos["final_observation"][done_mask], treatment, generator=obs_generator)
                    with torch.no_grad():
                        final_values[step, done_mask] = agent.get_value(final_obs).view(-1)
                if "final_info" in infos and "episode" in infos["final_info"]:
                    done_mask = infos["_final_info"]
                    for key, value in infos["final_info"]["episode"].items():
                        writer.add_scalar(f"train/{key}", scalar(value[done_mask]), global_step)

            with torch.no_grad():
                next_obs = apply_observation_treatment(clean_obs, treatment, generator=obs_generator)
                next_value = agent.get_value(next_obs).view(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = torch.zeros(num_envs, device=device)
                for step in reversed(range(num_steps)):
                    next_not_done = 1.0 - (next_done if step == num_steps - 1 else dones[step + 1])
                    nextvalues = next_value if step == num_steps - 1 else values[step + 1]
                    real_next = next_not_done * nextvalues + final_values[step]
                    delta = rewards[step] + config.ppo.gamma * real_next - values[step]
                    lastgaelam = delta + config.ppo.gamma * config.ppo.gae_lambda * next_not_done * lastgaelam
                    advantages[step] = lastgaelam
                returns = advantages + values
            b_obs, b_actions = obs_buffer.reshape(-1, 43), actions.reshape(-1, 7)
            b_logprobs, b_advantages, b_returns = logprobs.reshape(-1), advantages.reshape(-1), returns.reshape(-1)
            batch_actual = b_obs.shape[0]
            minibatch_size = batch_actual // config.ppo.num_minibatches
            if minibatch_size < 1:
                raise ValueError("num_minibatches exceeds rollout batch")
            agent.train()
            indices = np.arange(batch_actual)
            for _ in range(config.ppo.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_actual, minibatch_size):
                    mb = indices[start:start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    adv = (b_advantages[mb] - b_advantages[mb].mean()) / (b_advantages[mb].std() + 1e-8)
                    policy_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - config.ppo.clip_coef, 1 + config.ppo.clip_coef)).mean()
                    value_loss = 0.5 * (newvalue.view(-1) - b_returns[mb]).square().mean()
                    loss = policy_loss + config.ppo.vf_coef * value_loss - config.ppo.entropy_coefficient * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.ppo.max_grad_norm)
                    optimizer.step()
            elapsed = max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/SPS", continuation_step / elapsed, global_step)
            writer.add_scalar("interface/noise_multiplier", current, global_step)
            for key, value in rollout_stats.items():
                writer.add_scalar(f"interface/{key}", scalar(value / num_steps), global_step)
            print(f"arm={args.arm} seed={args.seed} update={update}/{updates} global_step={global_step} multiplier={current:.6g} rollout_sps={batch_size / max(time.time() - rollout_start, 1e-6):.1f} sps={continuation_step / elapsed:.1f}", flush=True)
            final_multiplier = current
            if update % config.continuation.checkpoint_interval_updates == 0 or update == updates:
                save_checkpoint(out / "checkpoints" / f"checkpoint_update_{update:04d}.pt", agent=agent, optimizer=optimizer, snapshot=snapshot, arm=args.arm, seed=args.seed, source_path=source_path, source_sha=source_sha, global_step=global_step, continuation_step=continuation_step, update=update, current_multiplier=current, obs_generator=obs_generator, action_generator=action_generator)

        save_checkpoint(out / "final_ckpt.pt", agent=agent, optimizer=optimizer, snapshot=snapshot, arm=args.arm, seed=args.seed, source_path=source_path, source_sha=source_sha, global_step=global_step, continuation_step=continuation_step, update=updates, current_multiplier=final_multiplier, obs_generator=obs_generator, action_generator=action_generator)
        print(f"EXP004_CONTINUATION_OK arm={args.arm} seed={args.seed} checkpoint={out / 'final_ckpt.pt'}", flush=True)
        return 0
    finally:
        writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
