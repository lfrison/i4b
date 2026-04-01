"""SAC-Lagrangian algorithm for constrained/safe RL.

Extends Soft Actor-Critic with a learnable Lagrange multiplier to enforce
cost constraints. The algorithm jointly optimizes:

1. **Actor**: maximize Q_reward - alpha*log_prob - beta*Q_cost
2. **Reward critic**: minimize TD error on reward Q-values (min-ensemble)
3. **Cost critic**: minimize TD error on cost Q-values (max-ensemble)
4. **Entropy coef (alpha)**: auto-tune to match target entropy
5. **Lagrange multiplier (beta)**: dual ascent to enforce cost_limit

Based on the algorithm from saferl-lib (nrgrp/saferl-lib), adapted for
GPU-resident training with our JAX env pipeline.

Reference:
    - Ha et al., "Learning to be Safe: Deep RL with a Safety Critic", 2020
    - Haarnoja et al., "Soft Actor-Critic", 2018
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.rl.algorithms.sac import SACActorCritic
from src.rl.storage.replay_buffer import GPUReplayBuffer, ReplayBufferSamples


@dataclass
class SACLagConfig:
    """Hyperparameters for SAC-Lagrangian."""

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    lagrange_lr: float = 3e-4

    # SAC parameters
    gamma: float = 0.99
    cost_gamma: float = 0.99
    tau: float = 0.005
    target_update_interval: int = 1

    # Entropy
    init_alpha: float = 1.0
    target_entropy: Optional[float] = None  # auto: -num_actions

    # Constraint
    cost_limit: float = 0.05
    lagrange_init: float = 0.0  # initial beta (before softplus)

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 1000  # random exploration steps

    # Training
    gradient_steps: int = 1  # gradient updates per env step


class SACLag:
    """SAC-Lagrangian algorithm.

    Follows the same interface pattern as our PPOLag but for off-policy
    training: collect transitions -> store in replay buffer -> sample and
    update.

    Args:
        actor_critic: SACActorCritic module.
        cfg: SACLagConfig hyperparameters.
        device: PyTorch device string.
    """

    def __init__(
        self,
        actor_critic: SACActorCritic,
        cfg: SACLagConfig = SACLagConfig(),
        device: str = "cuda:0",
    ):
        self.ac = actor_critic.to(device)
        self.cfg = cfg
        self.device = torch.device(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.ac.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.ac.critic.parameters(), lr=cfg.critic_lr
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            self.ac.cost_critic.parameters(), lr=cfg.critic_lr
        )

        # Auto-entropy: log_alpha is the learnable parameter
        self.log_alpha = torch.tensor(
            float(torch.tensor(cfg.init_alpha).log()),
            device=self.device, requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=cfg.alpha_lr
        )

        # Lagrange multiplier: parameterized via softplus(soft_beta)
        self.soft_beta = torch.tensor(
            cfg.lagrange_init, device=self.device, requires_grad=True,
        )
        self.beta_optimizer = torch.optim.Adam(
            [self.soft_beta], lr=cfg.lagrange_lr
        )

        # Target entropy
        self._target_entropy = cfg.target_entropy

        self._update_count = 0

    @property
    def target_entropy(self) -> float:
        if self._target_entropy is not None:
            return self._target_entropy
        # Auto: -dim(A) (standard SAC heuristic)
        return -float(self.ac.actor.mean_head.out_features)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.soft_beta).detach()

    def update(self, replay_buffer: GPUReplayBuffer) -> Dict[str, float]:
        """Run one gradient update on all components.

        Args:
            replay_buffer: GPU replay buffer to sample from.

        Returns:
            Dict of scalar metrics for logging.
        """
        cfg = self.cfg
        batch = replay_buffer.sample(cfg.batch_size)

        # Use privileged obs for critics if available.
        critic_obs = batch.critic_observations if batch.critic_observations is not None else batch.observations
        next_critic_obs = batch.next_critic_observations if batch.next_critic_observations is not None else batch.next_observations

        # ------------------------------------------------------------------
        # 1. Compute targets (no gradient)
        # ------------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_prob = self.ac.actor(batch.next_observations)
            alpha = self.log_alpha.exp()

            # Reward target: min Q - alpha * log_prob
            next_q1, next_q2 = self.ac.critic_target(next_critic_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).squeeze(-1)
            target_q = batch.rewards + (1 - batch.dones) * cfg.gamma * (next_q - alpha * next_log_prob)

            # Cost target: max Qc (conservative on costs)
            next_qc1, next_qc2 = self.ac.cost_critic_target(next_critic_obs, next_actions)
            next_qc = torch.max(next_qc1, next_qc2).squeeze(-1)
            target_qc = batch.costs + (1 - batch.dones) * cfg.cost_gamma * next_qc

        # ------------------------------------------------------------------
        # 2. Update reward critic
        # ------------------------------------------------------------------
        q1, q2 = self.ac.critic(critic_obs, batch.actions)
        critic_loss = 0.5 * (
            F.mse_loss(q1.squeeze(-1), target_q)
            + F.mse_loss(q2.squeeze(-1), target_q)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------
        # 3. Update cost critic
        # ------------------------------------------------------------------
        qc1, qc2 = self.ac.cost_critic(critic_obs, batch.actions)
        cost_critic_loss = 0.5 * (
            F.mse_loss(qc1.squeeze(-1), target_qc)
            + F.mse_loss(qc2.squeeze(-1), target_qc)
        )
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        # ------------------------------------------------------------------
        # 4. Update actor
        # ------------------------------------------------------------------
        actions_pi, log_prob_pi = self.ac.actor(batch.observations)
        alpha = self.log_alpha.exp().detach()
        beta = F.softplus(self.soft_beta)

        # Reward Q for current policy actions
        q1_pi, q2_pi = self.ac.critic(critic_obs, actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi).squeeze(-1)

        # Cost Q for current policy actions
        qc1_pi, qc2_pi = self.ac.cost_critic(critic_obs, actions_pi)
        max_qc_pi = torch.max(qc1_pi, qc2_pi).squeeze(-1)

        # Actor loss: maximize reward Q, minimize entropy penalty + cost penalty
        actor_loss = (alpha * log_prob_pi - min_q_pi + beta * max_qc_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------------
        # 5. Update entropy coefficient (alpha)
        # ------------------------------------------------------------------
        alpha_loss = -(
            self.log_alpha * (log_prob_pi.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------------------------------------------------
        # 6. Update Lagrange multiplier (beta)
        # ------------------------------------------------------------------
        beta_loss = F.softplus(self.soft_beta) * (
            cfg.cost_limit - max_qc_pi.detach().mean()
        )
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()

        # ------------------------------------------------------------------
        # 7. Soft-update target networks
        # ------------------------------------------------------------------
        self._update_count += 1
        if self._update_count % cfg.target_update_interval == 0:
            self.ac.polyak_update(cfg.tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "cost_critic_loss": cost_critic_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "beta": F.softplus(self.soft_beta).item(),
            "alpha_loss": alpha_loss.item(),
            "beta_loss": beta_loss.item(),
            "mean_q": min_q_pi.mean().item(),
            "mean_qc": max_qc_pi.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "target_qc_mean": target_qc.mean().item(),
        }

    def save(self, path: str, extra: dict = None):
        """Save checkpoint."""
        state = {
            "actor_critic": self.ac.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "cost_critic_optimizer": self.cost_critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "soft_beta": self.soft_beta.data,
            "beta_optimizer": self.beta_optimizer.state_dict(),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load(self, path: str):
        """Load checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(state["actor_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.cost_critic_optimizer.load_state_dict(state["cost_critic_optimizer"])
        self.log_alpha.data.copy_(state["log_alpha"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.soft_beta.data.copy_(state["soft_beta"])
        self.beta_optimizer.load_state_dict(state["beta_optimizer"])
