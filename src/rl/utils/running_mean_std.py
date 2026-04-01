"""GPU-resident running mean and variance tracker (Welford's algorithm).

Used for observation and reward normalization during training. All state
tensors stay on the specified device — no CPU round-trips.

Usage::

    rms = RunningMeanStd(shape=(obs_dim,), device="cuda:0")
    rms.update(obs_batch)                    # (batch, obs_dim)
    normalized = (obs_batch - rms.mean) / (rms.std + 1e-8)
"""

from __future__ import annotations

import torch


class RunningMeanStd:
    """Tracks running mean and variance using Welford's online algorithm.

    Attributes:
        mean: Current mean estimate, shape ``(shape,)``.
        var: Current variance estimate, shape ``(shape,)``.
        count: Number of samples seen so far.
    """

    def __init__(self, shape: tuple = (), device: str = "cpu", epsilon: float = 1e-4):
        """Initialize with zero mean and unit variance.

        Args:
            shape: Shape of the statistic being tracked (e.g. observation dim).
            device: Torch device for internal state tensors.
            epsilon: Initial count to prevent division by zero.
        """
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count: float = epsilon

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation (clamped to avoid numerical issues)."""
        return torch.sqrt(self.var).clamp(min=1e-6)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Update statistics with a new batch of data.

        Args:
            batch: Tensor of shape ``(batch_size, *shape)`` or ``(*shape,)``.
        """
        if batch.dim() == self.mean.dim():
            batch = batch.unsqueeze(0)

        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, correction=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ) -> None:
        """Combine batch statistics with running statistics (parallel Welford)."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using current running statistics."""
        return (x - self.mean) / (self.std + 1e-8)

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "mean": self.mean.cpu(),
            "var": self.var.cpu(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.mean = state["mean"].to(self.device)
        self.var = state["var"].to(self.device)
        self.count = state["count"]
