#!/usr/bin/env python3
"""Train the EXP-002 state-only PPO baseline.

This entry point is intentionally separate from the legacy server script. It
uses the verified headless patch before importing ManiSkill environments,
passes every EXP-002 environment parameter explicitly, and records a
checkpoint bundle containing the policy, optimizer, configuration, and step.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from exp002.config import ExperimentConfig, load_experiment_config
from exp002.noise import (
    InterfaceNoiseConfig,
    PEG_INSERTION_STATE_LAYOUT,
    add_peg_insertion_observation_noise,
    add_action_noise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(dim=1),
            distribution.entropy().sum(dim=1),
            self.critic(obs),
        )


def make_noise_config(config: ExperimentConfig) -> InterfaceNoiseConfig:
    if config.state_layout != "peg_insertion_state_v1":
        raise ValueError(f"unsupported EXP-002 state layout: {config.state_layout}")
    return InterfaceNoiseConfig(
        enabled=config.noise_enabled,
        obs_sigma=config.obs_sigma,
        obs_clip=config.obs_clip,
        action_sigma=config.action_sigma,
        action_clip=config.action_clip,
        state_layout=config.state_layout,
        qpos_sigma=config.qpos_sigma,
        qpos_clip=config.qpos_clip,
        qvel_sigma=config.qvel_sigma,
        qvel_clip=config.qvel_clip,
        position_sigma=config.position_sigma,
        position_clip=config.position_clip,
        orientation_sigma_deg=config.orientation_sigma_deg,
        orientation_clip_deg=config.orientation_clip_deg,
    )


def policy_observation(
    clean_obs: torch.Tensor, noise: InterfaceNoiseConfig
) -> torch.Tensor:
    if not noise.enabled:
        return clean_obs
    return add_peg_insertion_observation_noise(
        clean_obs,
        noise,
        layout=PEG_INSERTION_STATE_LAYOUT,
    )


def tensor_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    num_envs = args.num_envs or config.num_envs
    num_steps = args.num_steps or config.num_steps
    total_timesteps = args.total_timesteps or config.total_timesteps
    seed = args.seed if args.seed is not None else config.seed
    batch_size = num_envs * num_steps
    if batch_size <= 0 or total_timesteps < batch_size:
        raise ValueError("total_timesteps must be at least one rollout batch")
    if total_timesteps % batch_size != 0:
        raise ValueError("total_timesteps must be divisible by num_envs*num_steps")
    num_iterations = total_timesteps // batch_size
    if args.max_updates is not None:
        if args.max_updates < 1:
            raise ValueError("--max-updates must be positive")
        num_iterations = min(num_iterations, args.max_updates)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured EXP-002 run")
    device = torch.device(args.device)
    seed_everything(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": asdict(config),
                "overrides": {
                    "num_envs": num_envs,
                    "num_steps": num_steps,
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "device": str(device),
                    "max_updates": args.max_updates,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # This import must happen before importing mani_skill.envs.
    import exp002.headless_compat  # noqa: F401
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers ManiSkill tasks
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = None
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    try:
        env = gym.make(
            config.task,
            num_envs=num_envs,
            obs_mode=config.obs_mode,
            control_mode=config.control_mode,
            robot_uids=config.robot_uids,
            sim_backend=config.sim_backend,
            render_backend=config.render_backend,
            render_mode=None,
            reward_mode=config.reward_mode,
            reconfiguration_freq=0,
        )
        env = ManiSkillVectorEnv(
            env,
            num_envs,
            ignore_terminations=False,
            record_metrics=True,
        )
        if not isinstance(env.single_action_space, gym.spaces.Box):
            raise TypeError("EXP-002 requires a continuous Box action space")
        if env.single_observation_space.shape != (PEG_INSERTION_STATE_LAYOUT.total_dim,):
            raise ValueError(
                "unexpected observation shape: "
                f"{env.single_observation_space.shape}"
            )
        if env.single_action_space.shape != (7,):
            raise ValueError(f"unexpected action shape: {env.single_action_space.shape}")

        clean_obs, _ = env.reset(seed=seed)
        if not isinstance(clean_obs, torch.Tensor):
            raise TypeError("ManiSkill must return tensor observations for EXP-002")
        if clean_obs.device != device:
            raise RuntimeError(
                f"observation device {clean_obs.device} does not match {device}"
            )

        observation_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        agent = Agent(observation_dim, action_dim).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        noise = make_noise_config(config)
        noise.validate()

        obs = torch.zeros(
            (num_steps, num_envs, observation_dim), device=device, dtype=torch.float32
        )
        actions = torch.zeros(
            (num_steps, num_envs, action_dim), device=device, dtype=torch.float32
        )
        logprobs = torch.zeros((num_steps, num_envs), device=device)
        rewards = torch.zeros((num_steps, num_envs), device=device)
        dones = torch.zeros((num_steps, num_envs), device=device)
        values = torch.zeros((num_steps, num_envs), device=device)

        action_low = torch.as_tensor(env.single_action_space.low, device=device)
        action_high = torch.as_tensor(env.single_action_space.high, device=device)
        next_done = torch.zeros(num_envs, device=device)
        global_step = 0
        start_time = time.time()

        for iteration in range(1, num_iterations + 1):
            final_values = torch.zeros((num_steps, num_envs), device=device)
            agent.eval()
            rollout_start = time.time()
            action_delta_sum = 0.0
            action_clip_fraction_sum = 0.0

            for step in range(num_steps):
                global_step += num_envs
                current_obs = policy_observation(clean_obs, noise)
                obs[step] = current_obs
                dones[step] = next_done

                with torch.no_grad():
                    raw_action, logprob, _, value = agent.get_action_and_value(current_obs)
                    values[step] = value.flatten()
                actions[step] = raw_action
                logprobs[step] = logprob

                noisy_action = add_action_noise(raw_action, noise)
                applied_action = torch.clamp(noisy_action, action_low, action_high)
                action_delta_sum += tensor_scalar((applied_action - raw_action).abs())
                action_clip_fraction_sum += tensor_scalar(
                    (
                        (applied_action <= action_low + 1e-6)
                        | (applied_action >= action_high - 1e-6)
                    ).float()
                )

                clean_obs, reward, terminations, truncations, infos = env.step(applied_action)
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
                        final_obs = policy_observation(
                            infos["final_observation"][done_mask], noise
                        )
                        with torch.no_grad():
                            final_values[step, done_mask] = agent.get_value(
                                final_obs
                            ).view(-1)

            rollout_time = time.time() - rollout_start
            with torch.no_grad():
                next_value = agent.get_value(policy_observation(clean_obs, noise)).view(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = torch.zeros(num_envs, device=device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = next_not_done * nextvalues + final_values[t]
                    delta = rewards[t] + config.gamma * real_next_values - values[t]
                    lastgaelam = (
                        delta
                        + config.gamma
                        * config.gae_lambda
                        * next_not_done
                        * lastgaelam
                    )
                    advantages[t] = lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, observation_dim))
            b_actions = actions.reshape((-1, action_dim))
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            agent.train()
            batch_size_actual = b_obs.shape[0]
            minibatch_size = batch_size_actual // config.num_minibatches
            if minibatch_size < 1:
                raise ValueError("num_minibatches is larger than the rollout batch")
            indices = np.arange(batch_size_actual)
            stop_update = False
            for epoch in range(config.update_epochs):
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
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                    policy_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages
                        * torch.clamp(
                            ratio,
                            1.0 - config.clip_coef,
                            1.0 + config.clip_coef,
                        ),
                    ).mean()
                    value_loss = 0.5 * (
                        newvalue.view(-1) - b_returns[mb]
                    ).square().mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        + config.vf_coef * value_loss
                        - config.entropy_coefficient * entropy_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if config.target_kl is not None and approx_kl > config.target_kl:
                        stop_update = True
                        break
                if stop_update:
                    break

            elapsed = max(time.time() - start_time, 1e-6)
            sps = global_step / elapsed
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar(
                "interface/action_delta_abs_mean",
                action_delta_sum / num_steps,
                global_step,
            )
            writer.add_scalar(
                "interface/action_clip_fraction",
                action_clip_fraction_sum / num_steps,
                global_step,
            )
            print(
                f"iteration={iteration} global_step={global_step} "
                f"rollout_sps={num_envs * num_steps / max(rollout_time, 1e-6):.1f} "
                f"sps={sps:.1f}",
                flush=True,
            )

        checkpoint = {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "iteration": num_iterations,
            "seed": seed,
            "config": asdict(config),
            "overrides": {
                "num_envs": num_envs,
                "num_steps": num_steps,
                "total_timesteps": total_timesteps,
                "device": str(device),
            },
        }
        checkpoint_path = output_dir / "final_ckpt.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"checkpoint={checkpoint_path}", flush=True)
        return 0
    finally:
        writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
