"""Shared PPO network components for EXP-002 training and evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The actor-critic architecture used by both training and evaluation."""

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

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        return Normal(action_mean, action_std).sample()

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
