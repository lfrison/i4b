"""SAC actor-critic networks for off-policy safe RL.

Provides ``SACActorCritic`` with:
- Squashed Gaussian actor (tanh)
- Twin Q-critics for reward (min-ensemble for pessimism)
- Cost Q-critic(s) (max-ensemble for conservatism on costs)
- Target networks for both critic types

All networks are simple MLPs with ReLU activations, following the
standard SAC architecture.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2
LOG_PROB_EPSILON = 1e-6


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """Build a simple MLP with activation after each hidden layer."""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """SAC actor with tanh-squashed Gaussian output."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.trunk = _build_mlp(num_obs, hidden_dims, hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], num_actions)
        self.log_std_head = nn.Linear(hidden_dims[-1], num_actions)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) with tanh squashing."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        x = dist.rsample()  # reparameterized sample
        action = torch.tanh(x)

        # Log-prob with tanh correction
        log_prob = dist.log_prob(x).sum(-1)
        log_prob -= (2 * (
            torch.log(torch.tensor(2.0, device=obs.device))
            - x
            - F.softplus(-2 * x)
        )).sum(-1)

        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean through tanh)."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        return torch.tanh(mean)


class TwinQCritic(nn.Module):
    """Twin Q-networks for SAC (min-ensemble for reward, max for cost)."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.q1 = _build_mlp(num_obs + num_actions, hidden_dims, 1)
        self.q2 = _build_mlp(num_obs + num_actions, hidden_dims, 1)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q1, q2), each shaped (batch, 1)."""
        sa = torch.cat([obs, actions], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACActorCritic(nn.Module):
    """Full SAC actor-critic with cost critics for safe RL.

    Components:
    - actor: squashed Gaussian policy
    - critic: twin Q for reward
    - critic_target: EMA target for reward Q
    - cost_critic: twin Q for cost
    - cost_critic_target: EMA target for cost Q

    Args:
        num_obs: Observation dimension (actor input).
        num_actions: Action dimension.
        actor_hidden_dims: Actor MLP hidden sizes.
        critic_hidden_dims: Critic MLP hidden sizes.
        num_critic_obs: If provided, critics see a different (larger) obs.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        num_critic_obs: Optional[int] = None,
    ):
        super().__init__()
        self.num_obs = num_obs
        critic_input = num_critic_obs or num_obs

        self.actor = SquashedGaussianActor(num_obs, num_actions, actor_hidden_dims)
        self.critic = TwinQCritic(critic_input, num_actions, critic_hidden_dims)
        self.critic_target = deepcopy(self.critic)
        self.cost_critic = TwinQCritic(critic_input, num_actions, critic_hidden_dims)
        self.cost_critic_target = deepcopy(self.cost_critic)

        # Freeze targets
        for p in self.critic_target.parameters():
            p.requires_grad = False
        for p in self.cost_critic_target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        raise NotImplementedError("Use act() or specific networks directly.")

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample action for data collection."""
        action, _ = self.actor(obs)
        return action

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        return self.actor.deterministic(obs)

    def polyak_update(self, tau: float = 0.005):
        """Soft-update target networks."""
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.lerp_(p.data, tau)
        for p, pt in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            pt.data.lerp_(p.data, tau)
