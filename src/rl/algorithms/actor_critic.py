"""Actor-critic network architectures for on-policy RL.

Provides ``ActorCritic`` for standard RL and ``SafeActorCritic`` with an
additional cost value head for constrained/safe RL (PPO-Lagrangian).

Both support **asymmetric observations**: the actor and critic can receive
different input dimensions, enabling teacher-student training where the
critic sees privileged information (building params, extended forecasts)
not available to the actor at deployment.

Both use simple MLP backbones with ELU activations, following the
RSL-RL / IsaacLab convention.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type = nn.ELU,
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


class ActorCritic(nn.Module):
    """MLP actor-critic for continuous control.

    The actor outputs a Gaussian policy (mean + learned log-std).
    The critic outputs a scalar state value.

    Supports asymmetric observations: if ``num_critic_obs`` is provided,
    the critic receives a different (typically larger) observation vector
    than the actor. This enables teacher-student setups where the critic
    sees privileged information during training.

    Args:
        num_obs: Observation dimension for the actor (policy observations).
        num_actions: Action dimension.
        actor_hidden_dims: Hidden layer sizes for the actor MLP.
        critic_hidden_dims: Hidden layer sizes for the critic MLP.
        init_noise_std: Initial standard deviation for the policy.
        num_critic_obs: Observation dimension for the critic. If None,
            the critic uses the same dimension as the actor (symmetric).
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        init_noise_std: float = 1.0,
        num_critic_obs: Optional[int] = None,
    ):
        super().__init__()
        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs or num_obs
        self.is_asymmetric = num_critic_obs is not None and num_critic_obs != num_obs

        self.actor = _build_mlp(num_obs, actor_hidden_dims, num_actions)
        self.critic = _build_mlp(self.num_critic_obs, critic_hidden_dims, 1)
        self.LOG_STD_MIN = -2.0   # std >= 0.14
        self.LOG_STD_MAX = 0.5    # std <= 1.65
        self.log_std = nn.Parameter(
            torch.full((num_actions,), fill_value=float(torch.tensor(init_noise_std).log()))
        )

    def forward(self, obs: torch.Tensor):
        """Not used directly — call act() or evaluate() instead."""
        raise NotImplementedError("Use act() or evaluate().")

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the policy (used during rollout collection).

        Args:
            obs: Actor observations ``(batch, num_obs)``.
            critic_obs: Critic observations ``(batch, num_critic_obs)``.
                If None, the critic uses ``obs``.

        Returns:
            actions: Sampled actions ``(batch, num_actions)``.
            values: Value estimates ``(batch,)``.
            log_probs: Log-probabilities of the sampled actions ``(batch,)``.
        """
        mean = self.actor(obs)
        std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp().expand_as(mean)
        dist = Normal(mean, std)

        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(critic_obs if critic_obs is not None else obs).squeeze(-1)
        return actions, values, log_probs

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under the current policy (used during PPO update).

        Args:
            obs: Actor observations ``(batch, num_obs)``.
            actions: Actions to evaluate ``(batch, num_actions)``.
            critic_obs: Critic observations ``(batch, num_critic_obs)``.
                If None, the critic uses ``obs``.

        Returns:
            values: Value estimates ``(batch,)``.
            log_probs: Log-probabilities ``(batch,)``.
            entropy: Policy entropy ``(batch,)``.
        """
        mean = self.actor(obs)
        std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp().expand_as(mean)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(critic_obs if critic_obs is not None else obs).squeeze(-1)
        return values, log_probs, entropy


class SafeActorCritic(ActorCritic):
    """Actor-critic with twin cost value heads for safe RL.

    Inherits the standard actor and reward critic from ActorCritic, and
    adds **twin** cost critic MLPs. The max over the two cost value
    estimates is used as the conservative (pessimistic-on-safety) cost
    prediction — this is the "double cost critic" trick from CSAC-LB /
    WCSAC (saferl-lib):

    - Reward critic: single V(s) (or could be twin-min for reward)
    - Cost critics: twin V_c(s), take **max** → overestimate cost → safer

    By overestimating the expected cost, the policy is pushed harder
    toward constraint satisfaction.

    Supports asymmetric observations: both the reward critic and cost
    critics receive ``critic_obs`` (privileged), while the actor receives
    standard ``obs``.

    Args:
        num_obs: Observation dimension for the actor.
        num_actions: Action dimension.
        actor_hidden_dims: Actor MLP hidden sizes.
        critic_hidden_dims: Reward critic MLP hidden sizes.
        cost_critic_hidden_dims: Cost critic MLP hidden sizes.
        init_noise_std: Initial policy standard deviation.
        num_critic_obs: Observation dimension for both critics. If None,
            critics use the same dimension as the actor.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        cost_critic_hidden_dims: Sequence[int] = (256, 256, 256),
        init_noise_std: float = 1.0,
        num_critic_obs: Optional[int] = None,
    ):
        super().__init__(
            num_obs, num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            init_noise_std=init_noise_std,
            num_critic_obs=num_critic_obs,
        )
        critic_input = num_critic_obs or num_obs
        # Twin cost critics: take max for conservative cost estimation.
        self.cost_critic_1 = _build_mlp(critic_input, cost_critic_hidden_dims, 1)
        self.cost_critic_2 = _build_mlp(critic_input, cost_critic_hidden_dims, 1)

    @property
    def cost_critic(self):
        """Backward-compatible access (returns cost_critic_1).

        For code that only needs a single reference, e.g. parameter
        counting. Prefer using cost_values via act()/evaluate() which
        already apply the max-ensemble.
        """
        return self.cost_critic_1

    def _cost_value(self, c_obs: torch.Tensor) -> torch.Tensor:
        """Compute conservative cost value: max over twin cost critics.

        Args:
            c_obs: Critic observations (batch, num_critic_obs).

        Returns:
            cost_values: Max of twin estimates (batch,).
        """
        cv1 = self.cost_critic_1(c_obs).squeeze(-1)
        cv2 = self.cost_critic_2(c_obs).squeeze(-1)
        return torch.max(cv1, cv2)

    def _cost_values_both(self, c_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both cost value estimates (for computing individual losses).

        Returns:
            (cv1, cv2): Each shaped (batch,).
        """
        cv1 = self.cost_critic_1(c_obs).squeeze(-1)
        cv2 = self.cost_critic_2(c_obs).squeeze(-1)
        return cv1, cv2

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and return cost value estimates alongside reward values.

        Cost values are the **max** over twin cost critics (conservative).

        Args:
            obs: Actor observations ``(batch, num_obs)``.
            critic_obs: Critic observations ``(batch, num_critic_obs)``.

        Returns:
            actions, values, log_probs, cost_values
        """
        mean = self.actor(obs)
        std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp().expand_as(mean)
        dist = Normal(mean, std)

        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        c_obs = critic_obs if critic_obs is not None else obs
        values = self.critic(c_obs).squeeze(-1)
        cost_values = self._cost_value(c_obs)
        return actions, values, log_probs, cost_values

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and return cost value estimates.

        Cost values are the **max** over twin cost critics (conservative).

        Args:
            obs: Actor observations ``(batch, num_obs)``.
            actions: Actions ``(batch, num_actions)``.
            critic_obs: Critic observations ``(batch, num_critic_obs)``.

        Returns:
            values, log_probs, entropy, cost_values
        """
        values, log_probs, entropy = ActorCritic.evaluate(self, obs, actions, critic_obs=critic_obs)
        c_obs = critic_obs if critic_obs is not None else obs
        cost_values = self._cost_value(c_obs)
        return values, log_probs, entropy, cost_values
