"""GPU-resident rollout buffer for on-policy algorithms.

Pre-allocates all tensors on the target device at init. Stores transitions
during rollout collection, then computes GAE advantages in-place before
the policy update. Supports an optional cost signal for safe RL.

Usage::

    storage = RolloutStorage(
        num_envs=4096, num_transitions_per_env=32,
        obs_shape=(obs_dim,), action_shape=(act_dim,),
        device="cuda:0",
    )

    for step in range(32):
        obs, rewards, dones, info = env.step(actions)
        storage.add_transition(obs, actions, rewards, dones, values, log_probs)

    storage.compute_returns(last_values, dones)
"""

from __future__ import annotations

import torch


class RolloutStorage:
    """Fixed-size GPU buffer for on-policy rollout data.

    All tensors are pre-allocated on ``device`` at construction.
    Transitions are written sequentially via ``add_transition()``
    and advantages are computed in-place via ``compute_returns()``.

    Attributes:
        num_envs: Number of parallel environments.
        num_transitions: Rollout length (steps per env per update).
        device: PyTorch device where tensors reside.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        action_shape: tuple,
        device: str = "cuda:0",
        has_cost: bool = False,
        privileged_obs_shape: tuple | None = None,
    ):
        """Allocate rollout buffers.

        Args:
            num_envs: Number of parallel environments.
            num_transitions_per_env: Steps collected per env before each update.
            obs_shape: Shape of a single observation (e.g. ``(obs_dim,)``).
            action_shape: Shape of a single action (e.g. ``(1,)``).
            device: PyTorch device for all tensors.
            has_cost: If True, allocate cost-related buffers for safe RL.
            privileged_obs_shape: Shape of privileged observations for the
                critic. If None, no separate buffer is allocated and the
                critic uses the same observations as the actor.
        """
        self.num_envs = num_envs
        self.num_transitions = num_transitions_per_env
        self.device = torch.device(device)
        self.has_cost = has_cost
        self.has_privileged_obs = privileged_obs_shape is not None
        self._step = 0

        # Rollout buffers: (num_transitions, num_envs, *shape)
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape,
            dtype=torch.float32, device=self.device,
        )
        # Privileged observations buffer for asymmetric actor-critic.
        if self.has_privileged_obs:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape,
                dtype=torch.float32, device=self.device,
            )
        else:
            self.privileged_observations = None
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *action_shape,
            dtype=torch.float32, device=self.device,
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.float32, device=self.device,
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.float32, device=self.device,
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.float32, device=self.device,
        )
        self.log_probs = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.float32, device=self.device,
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.float32, device=self.device,
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.float32, device=self.device,
        )

        # Cost buffers for safe RL (PPO-Lagrangian).
        if has_cost:
            self.costs = torch.zeros(
                num_transitions_per_env, num_envs,
                dtype=torch.float32, device=self.device,
            )
            self.cost_values = torch.zeros(
                num_transitions_per_env, num_envs,
                dtype=torch.float32, device=self.device,
            )
            self.cost_advantages = torch.zeros(
                num_transitions_per_env, num_envs,
                dtype=torch.float32, device=self.device,
            )
            self.cost_returns = torch.zeros(
                num_transitions_per_env, num_envs,
                dtype=torch.float32, device=self.device,
            )

    def add_transition(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        costs: torch.Tensor | None = None,
        cost_values: torch.Tensor | None = None,
        privileged_observations: torch.Tensor | None = None,
    ) -> None:
        """Record one transition for all environments.

        Args:
            observations: ``(num_envs, obs_dim)``
            actions: ``(num_envs, act_dim)``
            rewards: ``(num_envs,)``
            dones: ``(num_envs,)``
            values: ``(num_envs,)`` — critic value estimate.
            log_probs: ``(num_envs,)`` — action log probability.
            costs: ``(num_envs,)`` — optional cost signal for safe RL.
            cost_values: ``(num_envs,)`` — optional cost value estimate.
            privileged_observations: ``(num_envs, privileged_obs_dim)`` —
                optional privileged observations for the critic.
        """
        if self._step >= self.num_transitions:
            raise RuntimeError(
                f"Storage full ({self._step}/{self.num_transitions}). "
                "Call clear() before collecting more transitions."
            )
        self.observations[self._step] = observations
        if self.has_privileged_obs and privileged_observations is not None:
            self.privileged_observations[self._step] = privileged_observations
        self.actions[self._step] = actions
        self.rewards[self._step] = rewards
        self.dones[self._step] = dones.float()
        self.values[self._step] = values
        self.log_probs[self._step] = log_probs

        if self.has_cost and costs is not None:
            self.costs[self._step] = costs
        if self.has_cost and cost_values is not None:
            self.cost_values[self._step] = cost_values

        self._step += 1

    def clear(self) -> None:
        """Reset the write pointer for the next rollout."""
        self._step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
        last_cost_values: torch.Tensor | None = None,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns in-place.

        Uses Generalized Advantage Estimation (Schulman et al., 2016).

        Args:
            last_values: Critic values at T+1 ``(num_envs,)``.
            last_dones: Done flags at T+1 ``(num_envs,)``.
            gamma: Discount factor for rewards.
            lam: GAE lambda for bias-variance tradeoff.
            last_cost_values: Cost critic values at T+1 (for safe RL).
            cost_gamma: Discount factor for costs.
            cost_lam: GAE lambda for cost advantages.
        """
        self._gae(
            self.rewards, self.values, self.dones,
            last_values, last_dones,
            gamma, lam,
            self.advantages, self.returns,
        )

        if self.has_cost and last_cost_values is not None:
            self._gae(
                self.costs, self.cost_values, self.dones,
                last_cost_values, last_dones,
                cost_gamma, cost_lam,
                self.cost_advantages, self.cost_returns,
            )

    @staticmethod
    def _gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        lam: float,
        out_advantages: torch.Tensor,
        out_returns: torch.Tensor,
    ) -> None:
        """Compute GAE in reverse order, writing results into pre-allocated tensors."""
        num_transitions = rewards.shape[0]
        last_gae = torch.zeros_like(last_values)

        for t in reversed(range(num_transitions)):
            if t == num_transitions - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            out_advantages[t] = last_gae

        out_returns.copy_(out_advantages + values)

    def mini_batch_generator(
        self, num_mini_batches: int
    ):
        """Yield randomized mini-batches for the policy update.

        Flattens ``(num_transitions, num_envs)`` → ``(total,)`` and splits
        into ``num_mini_batches`` random partitions.

        Yields:
            Dict with keys: ``obs``, ``actions``, ``values``, ``log_probs``,
            ``advantages``, ``returns``. Optionally includes ``critic_obs``
            (if privileged obs are stored) and cost-related keys.
        """
        total = self.num_transitions * self.num_envs
        batch_size = total // num_mini_batches
        indices = torch.randperm(total, device=self.device)

        # Flatten leading two dims.
        obs_flat = self.observations.reshape(total, -1)
        actions_flat = self.actions.reshape(total, -1)
        values_flat = self.values.reshape(total)
        log_probs_flat = self.log_probs.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        returns_flat = self.returns.reshape(total)

        if self.has_privileged_obs:
            priv_obs_flat = self.privileged_observations.reshape(total, -1)

        if self.has_cost:
            costs_flat = self.costs.reshape(total)
            cost_values_flat = self.cost_values.reshape(total)
            cost_advantages_flat = self.cost_advantages.reshape(total)
            cost_returns_flat = self.cost_returns.reshape(total)

        for i in range(num_mini_batches):
            idx = indices[i * batch_size : (i + 1) * batch_size]
            batch = {
                "obs": obs_flat[idx],
                "actions": actions_flat[idx],
                "values": values_flat[idx],
                "log_probs": log_probs_flat[idx],
                "advantages": advantages_flat[idx],
                "returns": returns_flat[idx],
            }
            if self.has_privileged_obs:
                batch["critic_obs"] = priv_obs_flat[idx]
            if self.has_cost:
                batch["costs"] = costs_flat[idx]
                batch["cost_values"] = cost_values_flat[idx]
                batch["cost_advantages"] = cost_advantages_flat[idx]
                batch["cost_returns"] = cost_returns_flat[idx]
            yield batch
