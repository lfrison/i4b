"""GPU-resident replay buffer for off-policy safe RL.

Stores transitions entirely on GPU to avoid CPU-GPU data transfers during
training. Supports cost signals for constrained RL (SAC-Lag, WCSAC, etc.).

All tensors are pre-allocated at initialization. Sampling is done with
random integer indexing on GPU.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class ReplayBufferSamples:
    """Container for a batch of replay buffer samples."""

    __slots__ = (
        "observations", "actions", "next_observations",
        "rewards", "dones", "costs",
        "critic_observations", "next_critic_observations",
    )

    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        costs: torch.Tensor,
        critic_observations: Optional[torch.Tensor] = None,
        next_critic_observations: Optional[torch.Tensor] = None,
    ):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.dones = dones
        self.costs = costs
        self.critic_observations = critic_observations
        self.next_critic_observations = next_critic_observations


class GPUReplayBuffer:
    """Fixed-size GPU-resident replay buffer with cost support.

    All data stays on the target device throughout the buffer's lifetime.
    No CPU staging is needed.

    Args:
        capacity: Maximum number of transitions to store.
        num_obs: Observation dimension (actor).
        num_actions: Action dimension.
        device: PyTorch device string.
        num_critic_obs: If provided, also stores privileged observations
            for asymmetric actor-critic training.
    """

    def __init__(
        self,
        capacity: int,
        num_obs: int,
        num_actions: int,
        device: str = "cuda:0",
        num_critic_obs: Optional[int] = None,
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        self._size = 0
        self._pos = 0

        # Pre-allocate all buffers on GPU.
        self.observations = torch.zeros(capacity, num_obs, device=self.device)
        self.actions = torch.zeros(capacity, num_actions, device=self.device)
        self.next_observations = torch.zeros(capacity, num_obs, device=self.device)
        self.rewards = torch.zeros(capacity, device=self.device)
        self.dones = torch.zeros(capacity, device=self.device)
        self.costs = torch.zeros(capacity, device=self.device)

        # Optional privileged obs for asymmetric training.
        if num_critic_obs is not None:
            self.critic_observations = torch.zeros(
                capacity, num_critic_obs, device=self.device
            )
            self.next_critic_observations = torch.zeros(
                capacity, num_critic_obs, device=self.device
            )
        else:
            self.critic_observations = None
            self.next_critic_observations = None

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        costs: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
        next_critic_obs: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch of transitions to the buffer.

        Args:
            obs: (batch, num_obs)
            actions: (batch, num_actions)
            next_obs: (batch, num_obs)
            rewards: (batch,)
            dones: (batch,)
            costs: (batch,)
            critic_obs: (batch, num_critic_obs) optional
            next_critic_obs: (batch, num_critic_obs) optional
        """
        batch_size = obs.shape[0]

        if batch_size <= self.capacity:
            # Check if we need to wrap around.
            end = self._pos + batch_size
            if end <= self.capacity:
                self.observations[self._pos:end] = obs
                self.actions[self._pos:end] = actions
                self.next_observations[self._pos:end] = next_obs
                self.rewards[self._pos:end] = rewards
                self.dones[self._pos:end] = dones
                self.costs[self._pos:end] = costs
                if self.critic_observations is not None and critic_obs is not None:
                    self.critic_observations[self._pos:end] = critic_obs
                    self.next_critic_observations[self._pos:end] = next_critic_obs
            else:
                # Split across wrap boundary.
                first = self.capacity - self._pos
                self.observations[self._pos:] = obs[:first]
                self.observations[:batch_size - first] = obs[first:]
                self.actions[self._pos:] = actions[:first]
                self.actions[:batch_size - first] = actions[first:]
                self.next_observations[self._pos:] = next_obs[:first]
                self.next_observations[:batch_size - first] = next_obs[first:]
                self.rewards[self._pos:] = rewards[:first]
                self.rewards[:batch_size - first] = rewards[first:]
                self.dones[self._pos:] = dones[:first]
                self.dones[:batch_size - first] = dones[first:]
                self.costs[self._pos:] = costs[:first]
                self.costs[:batch_size - first] = costs[first:]
                if self.critic_observations is not None and critic_obs is not None:
                    self.critic_observations[self._pos:] = critic_obs[:first]
                    self.critic_observations[:batch_size - first] = critic_obs[first:]
                    self.next_critic_observations[self._pos:] = next_critic_obs[:first]
                    self.next_critic_observations[:batch_size - first] = next_critic_obs[first:]

            self._pos = (self._pos + batch_size) % self.capacity
            self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            ReplayBufferSamples with all tensors on self.device.
        """
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        critic_obs = None
        next_critic_obs = None
        if self.critic_observations is not None:
            critic_obs = self.critic_observations[indices]
            next_critic_obs = self.next_critic_observations[indices]

        return ReplayBufferSamples(
            observations=self.observations[indices],
            actions=self.actions[indices],
            next_observations=self.next_observations[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
            costs=self.costs[indices],
            critic_observations=critic_obs,
            next_critic_observations=next_critic_obs,
        )
