"""PPO-Lagrangian algorithm for constrained / safe RL.

Extends PPO with a learnable Lagrange multiplier that penalizes
constraint violations. The cost constraint is enforced by augmenting
the surrogate objective:

    L_safe = L_ppo - lambda * (mean_cost - cost_limit)

where lambda is updated via dual gradient ascent after each PPO update.

Reference: Ray et al., "Benchmarking Safe Exploration in Deep RL" (2019).

Usage::

    actor_critic = SafeActorCritic(num_obs, num_actions).to("cuda:0")
    ppo_lag = PPOLag(actor_critic, cost_limit=25.0, device="cuda:0")
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.algorithms.actor_critic import SafeActorCritic
from src.rl.algorithms.ppo import PPOConfig


@dataclass
class PPOLagConfig(PPOConfig):
    """PPO-Lagrangian hyperparameters (extends PPOConfig)."""

    cost_limit: float = 25.0
    lagrange_lr: float = 5e-3
    lagrange_init: float = 1.0
    cost_value_loss_coef: float = 0.5
    cost_gamma: float = 0.99
    cost_lam: float = 0.95


class PPOLag:
    """PPO-Lagrangian for safe on-policy training.

    Maintains a learnable Lagrange multiplier (``log_lagrange``) that
    is updated via dual gradient ascent to satisfy cost constraints.

    Args:
        actor_critic: A SafeActorCritic with cost value head.
        cfg: PPO-Lagrangian hyperparameters.
        device: PyTorch device.
    """

    def __init__(
        self,
        actor_critic: SafeActorCritic,
        cfg: PPOLagConfig | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = cfg or PPOLagConfig()
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)

        # Learnable Lagrange multiplier (parameterized in log-space for positivity).
        init_log = float(torch.tensor(max(self.cfg.lagrange_init, 1e-8)).log())
        self.log_lagrange = nn.Parameter(
            torch.tensor(init_log, dtype=torch.float32, device=self.device)
        )

        # Separate optimizers for policy and Lagrange multiplier.
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.cfg.learning_rate
        )
        self.lagrange_optimizer = optim.Adam(
            [self.log_lagrange], lr=self.cfg.lagrange_lr
        )
        self.transition = _SafeRolloutTransition()

    @property
    def lagrange_multiplier(self) -> float:
        """Current Lagrange multiplier value."""
        return self.log_lagrange.exp().item()

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample actions and cache transition data including cost values.

        Args:
            obs: Actor observations ``(num_envs, num_obs)``.
            critic_obs: Critic observations ``(num_envs, num_critic_obs)``.

        Returns:
            actions: Sampled actions ``(num_envs, num_actions)``.
        """
        actions, values, log_probs, cost_values = self.actor_critic.act(obs, critic_obs=critic_obs)
        self.transition.actions = actions
        self.transition.values = values
        self.transition.log_probs = log_probs
        self.transition.cost_values = cost_values
        return actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        costs: torch.Tensor,
    ) -> None:
        """Cache environment outputs including cost signal.

        Args:
            rewards: Rewards ``(num_envs,)``.
            dones: Done flags ``(num_envs,)``.
            costs: Per-step costs ``(num_envs,)``.
        """
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.transition.costs = costs

    def update(self, storage) -> dict[str, float]:
        """Run PPO-Lagrangian update over the collected rollout.

        Args:
            storage: A RolloutStorage with ``has_cost=True`` and
                ``compute_returns()`` already called.

        Returns:
            Dict with loss statistics including ``lagrange_multiplier``,
            ``mean_cost``, ``cost_value_loss``.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_entropy = 0.0
        total_batches = 0

        lagrange = self.log_lagrange.exp().detach()

        for epoch in range(self.cfg.num_learning_epochs):
            for batch in storage.mini_batch_generator(self.cfg.num_mini_batches):
                values, log_probs, entropy, cost_values = self.actor_critic.evaluate(
                    batch["obs"], batch["actions"],
                    critic_obs=batch.get("critic_obs"),
                )

                # Reward surrogate loss (standard PPO).
                ratio = torch.exp(log_probs - batch["log_probs"])
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
                ) * advantages
                surrogate_loss = -torch.min(surr1, surr2).mean()

                # Cost surrogate loss (penalize high cost advantage).
                cost_advantages = batch["cost_advantages"]
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                    cost_advantages.std() + 1e-8
                )
                cost_surr1 = ratio * cost_advantages
                cost_surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
                ) * cost_advantages
                cost_surrogate = torch.max(cost_surr1, cost_surr2).mean()

                # Value losses.
                value_loss = (values - batch["returns"]).pow(2).mean()
                # Train both twin cost critics against the same target.
                c_obs = batch.get("critic_obs", batch["obs"])
                cv1, cv2 = self.actor_critic._cost_values_both(c_obs)
                cost_value_loss = (
                    (cv1 - batch["cost_returns"]).pow(2).mean()
                    + (cv2 - batch["cost_returns"]).pow(2).mean()
                ) * 0.5

                # Combined loss: policy + value - entropy + lagrange * cost_policy.
                loss = (
                    surrogate_loss
                    + lagrange * cost_surrogate
                    + self.cfg.value_loss_coef * value_loss
                    + self.cfg.cost_value_loss_coef * cost_value_loss
                    - self.cfg.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_entropy += entropy.mean().item()
                total_batches += 1

        # Update Lagrange multiplier via dual gradient ascent.
        mean_cost = storage.costs.mean().item()
        lagrange_loss = -self.log_lagrange.exp() * (mean_cost - self.cfg.cost_limit)
        self.lagrange_optimizer.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optimizer.step()

        return {
            "value_loss": mean_value_loss / max(total_batches, 1),
            "surrogate_loss": mean_surrogate_loss / max(total_batches, 1),
            "cost_value_loss": mean_cost_value_loss / max(total_batches, 1),
            "entropy": mean_entropy / max(total_batches, 1),
            "lagrange_multiplier": self.lagrange_multiplier,
            "mean_cost": mean_cost,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }


class _SafeRolloutTransition:
    """Temporary container for a single safe-RL transition."""

    __slots__ = (
        "actions", "values", "log_probs", "rewards", "dones",
        "costs", "cost_values",
    )

    def __init__(self):
        self.actions: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.log_probs: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None
        self.costs: torch.Tensor | None = None
        self.cost_values: torch.Tensor | None = None
