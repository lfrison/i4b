"""Proximal Policy Optimization (PPO) algorithm.

Implements the clipped surrogate objective (Schulman et al., 2017) with
value function clipping and entropy bonus. Operates entirely on GPU tensors.

Usage::

    actor_critic = ActorCritic(num_obs, num_actions).to("cuda:0")
    ppo = PPO(actor_critic, device="cuda:0")

    # After collecting rollout and computing GAE:
    for batch in storage.mini_batch_generator(num_mini_batches):
        ppo.update(batch)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.algorithms.actor_critic import ActorCritic


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    learning_rate: float = 3e-4
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    use_clipped_value_loss: bool = True
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float | None = 0.01
    schedule: str = "adaptive"  # "adaptive" or "fixed"


class PPO:
    """PPO algorithm with optional adaptive learning rate.

    Args:
        actor_critic: The policy/value network.
        cfg: PPO hyperparameters.
        device: PyTorch device.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        cfg: PPOConfig | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.cfg.learning_rate
        )
        self.transition = RolloutTransition()

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample actions from the policy and cache transition data.

        Args:
            obs: Actor observations ``(num_envs, num_obs)``.
            critic_obs: Critic observations ``(num_envs, num_critic_obs)``.
                If None, the critic uses ``obs``.

        Returns:
            actions: Sampled actions ``(num_envs, num_actions)``.
        """
        actions, values, log_probs = self.actor_critic.act(obs, critic_obs=critic_obs)
        self.transition.actions = actions
        self.transition.values = values
        self.transition.log_probs = log_probs
        return actions

    def process_env_step(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> None:
        """Cache environment outputs for the current transition.

        Args:
            rewards: Rewards ``(num_envs,)``.
            dones: Done flags ``(num_envs,)``.
        """
        self.transition.rewards = rewards
        self.transition.dones = dones

    def update(self, storage) -> dict[str, float]:
        """Run PPO update over the collected rollout.

        Args:
            storage: A RolloutStorage that has had ``compute_returns()`` called.

        Returns:
            Dict with loss statistics: ``value_loss``, ``surrogate_loss``,
            ``entropy``, ``kl``, ``learning_rate``.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        total_batches = 0

        for epoch in range(self.cfg.num_learning_epochs):
            for batch in storage.mini_batch_generator(self.cfg.num_mini_batches):
                values, log_probs, entropy = self.actor_critic.evaluate(
                    batch["obs"], batch["actions"],
                    critic_obs=batch.get("critic_obs"),
                )

                # Surrogate loss.
                ratio = torch.exp(log_probs - batch["log_probs"])
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
                surrogate_loss = -torch.min(surr1, surr2).mean()

                # Value loss.
                if self.cfg.use_clipped_value_loss:
                    value_clipped = batch["values"] + torch.clamp(
                        values - batch["values"],
                        -self.cfg.clip_param,
                        self.cfg.clip_param,
                    )
                    value_loss = torch.max(
                        (values - batch["returns"]).pow(2),
                        (value_clipped - batch["returns"]).pow(2),
                    ).mean()
                else:
                    value_loss = (values - batch["returns"]).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.cfg.value_loss_coef * value_loss
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
                mean_entropy += entropy.mean().item()
                total_batches += 1

        # Adaptive learning rate based on approximate KL divergence.
        if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
            with torch.no_grad():
                # Compute KL on last mini-batch as proxy.
                kl = (batch["log_probs"] - log_probs).mean().item()
            if kl > 2.0 * self.cfg.desired_kl:
                self._scale_lr(0.5)
            elif kl < 0.5 * self.cfg.desired_kl:
                self._scale_lr(2.0)
        else:
            kl = 0.0

        return {
            "value_loss": mean_value_loss / max(total_batches, 1),
            "surrogate_loss": mean_surrogate_loss / max(total_batches, 1),
            "entropy": mean_entropy / max(total_batches, 1),
            "kl": kl,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _scale_lr(self, factor: float) -> None:
        """Scale the learning rate by a multiplicative factor."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = max(pg["lr"] * factor, 1e-6)


class RolloutTransition:
    """Temporary container for a single transition's data."""

    __slots__ = ("actions", "values", "log_probs", "rewards", "dones")

    def __init__(self):
        self.actions: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.log_probs: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None
