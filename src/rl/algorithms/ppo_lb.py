"""PPO with Log-Barrier constraint for safe RL.

Adapts the CSAC-LB log-barrier approach (Ha et al., saferl-lib / nrgrp) to
on-policy PPO.

In CSAC-LB (off-policy SAC), the actor loss is:

    J_actor = E[ alpha * log_pi(a|s) - Q_reward(s,a) + barrier(Q_cost(s,a) - d) ]

Q_cost(s,a) is differentiable w.r.t. action through the critic network, giving
the actor direct gradient on how each action affects cost.

In PPO (on-policy), we approximate Q_cost(s,a) using GAE:

    Q_cost(s,a) ~ V_cost(s) + A_cost(s,a) = cost_returns

The actor loss becomes:

    L = -reward_surrogate + barrier_multiplier * cost_barrier_surrogate
        + value_losses - entropy

where cost_barrier_surrogate uses **cost advantages** for direction (which
actions increase/decrease cost) and the **barrier function** for magnitude
(how strongly to penalize based on cost level).

Extended log-barrier (from CSAC-LB):

    barrier(x, t) =
        -1/t * log(-x + eps)              if x <= -1/t^2
        t*x - 1/t*log(1/t^2) + 1/t        otherwise

where x = cost_return - cost_limit.

Reference:
    - Ha et al., CSAC-LB from saferl-lib (nrgrp/saferl-lib)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.algorithms.actor_critic import SafeActorCritic
from src.rl.algorithms.ppo import PPOConfig


@dataclass
class PPOLBConfig(PPOConfig):
    """PPO-LogBarrier hyperparameters (extends PPOConfig)."""

    cost_limit: float = 0.05
    cost_value_loss_coef: float = 0.5
    cost_gamma: float = 0.99
    cost_lam: float = 0.95

    # Log-barrier parameters
    barrier_factor: float = 3.0       # t: sharpness of barrier (fixed)
    barrier_multiplier: float = 1.0   # scaling weight on barrier term


class PPOLB:
    """PPO with Log-Barrier constraint for safe on-policy training.

    Follows the CSAC-LB design: the barrier is applied directly to Q_cost
    estimates (approximated via cost_returns from GAE), not as an adaptive
    weight.  Cost advantages provide the directional signal for the actor.

    Args:
        actor_critic: A SafeActorCritic with twin cost value heads.
        cfg: PPO-LogBarrier hyperparameters.
        device: PyTorch device.
    """

    def __init__(
        self,
        actor_critic: SafeActorCritic,
        cfg: PPOLBConfig | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = cfg or PPOLBConfig()
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)

        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.cfg.learning_rate
        )
        self.transition = _SafeRolloutTransition()

        self._barrier_factor = float(self.cfg.barrier_factor)

    # -----------------------------------------------------------------
    # Log-barrier
    # -----------------------------------------------------------------
    @staticmethod
    def log_barrier_extension(
        x: torch.Tensor,
        t: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Extended log-barrier function from CSAC-LB.

        Args:
            x: Input tensor.  Negative = safe, positive = violated.
            t: Barrier sharpness factor (larger = harder constraint).
            eps: Small constant for numerical stability.

        Returns:
            Barrier penalty tensor, same shape as x.
        """
        threshold = -1.0 / (t * t)
        safe = -1.0 / t * torch.log(-x + eps)
        infeasible = (
            t * x
            - 1.0 / t * torch.log(torch.tensor(1.0 / (t * t), device=x.device))
            + 1.0 / t
        )
        return torch.where(x <= threshold, safe, infeasible)

    # -----------------------------------------------------------------
    # Act / process
    # -----------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions, values, log_probs, cost_values = self.actor_critic.act(
            obs, critic_obs=critic_obs
        )
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
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.transition.costs = costs

    # -----------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------
    def update(self, storage, **kwargs) -> dict[str, float]:
        """Run PPO-LB update following CSAC-LB design.

        In CSAC-LB:
            actor_loss = alpha*log_pi - Q_reward + barrier(Q_cost - d)

        In PPO-LB, Q_cost(s,a) ~ cost_returns = V_cost + A_cost (from GAE).
        We use cost advantages for directional gradient and the barrier
        function applied to cost_returns for per-sample penalty magnitude.

        The cost barrier surrogate combines both:
            cost_barrier_surr = ratio * cost_advantages * barrier(cost_returns - d)

        - cost_advantages: direction (positive = action increases cost)
        - barrier(cost_returns - d): magnitude (large when near/above limit)
        - ratio: importance weighting for off-policy correction
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_barrier_loss = 0.0
        mean_entropy = 0.0
        total_batches = 0

        t = self._barrier_factor

        for epoch in range(self.cfg.num_learning_epochs):
            for batch in storage.mini_batch_generator(self.cfg.num_mini_batches):
                values, log_probs, entropy, cost_values = self.actor_critic.evaluate(
                    batch["obs"], batch["actions"],
                    critic_obs=batch.get("critic_obs"),
                )

                # ---- Importance ratio ----
                ratio = torch.exp(log_probs - batch["log_probs"])
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
                )

                # ---- Reward surrogate (standard PPO clip) ----
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                surrogate_loss = -torch.min(surr1, surr2).mean()

                # ---- Cost barrier surrogate (CSAC-LB adapted) ----
                # barrier(cost_returns - limit): per-sample penalty magnitude.
                # cost_returns ~ Q_cost from old policy (same role as Q_cost in CSAC-LB).
                with torch.no_grad():
                    barrier_values = self.log_barrier_extension(
                        batch["cost_returns"] - self.cfg.cost_limit, t
                    )

                # cost_advantages: direction signal (which actions increase cost).
                # Normalize like reward advantages for stable gradients.
                cost_advantages = batch["cost_advantages"]
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                    cost_advantages.std() + 1e-8
                )

                # Combined: direction * magnitude * importance weight.
                # Use max (pessimistic) clipping for cost (opposite of reward).
                cost_surr1 = ratio * cost_advantages * barrier_values
                cost_surr2 = clipped_ratio * cost_advantages * barrier_values
                cost_barrier_surrogate = torch.max(cost_surr1, cost_surr2).mean()

                # ---- Value losses ----
                value_loss = (values - batch["returns"]).pow(2).mean()
                # Twin cost critics — independent losses
                c_obs = batch.get("critic_obs", batch["obs"])
                cv1, cv2 = self.actor_critic._cost_values_both(c_obs)
                cost_value_loss = (
                    (cv1 - batch["cost_returns"]).pow(2).mean()
                    + (cv2 - batch["cost_returns"]).pow(2).mean()
                ) * 0.5

                # ---- Combined loss ----
                loss = (
                    surrogate_loss
                    + self.cfg.barrier_multiplier * cost_barrier_surrogate
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
                mean_barrier_loss += cost_barrier_surrogate.item()
                mean_entropy += entropy.mean().item()
                total_batches += 1

        n = max(total_batches, 1)
        mean_cost = storage.costs.mean().item()

        # Adaptive learning rate based on approximate KL divergence.
        if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
            with torch.no_grad():
                kl = (batch["log_probs"] - log_probs).mean().item()
            if kl > 2.0 * self.cfg.desired_kl:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1e-5)
            elif kl < 0.5 * self.cfg.desired_kl:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = min(pg["lr"] * 2.0, 1e-2)

        return {
            "value_loss": mean_value_loss / n,
            "surrogate_loss": mean_surrogate_loss / n,
            "cost_value_loss": mean_cost_value_loss / n,
            "barrier_loss": mean_barrier_loss / n,
            "barrier_factor": self._barrier_factor,
            "entropy": mean_entropy / n,
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
        self.actions = None
        self.values = None
        self.log_probs = None
        self.rewards = None
        self.dones = None
        self.costs = None
        self.cost_values = None
