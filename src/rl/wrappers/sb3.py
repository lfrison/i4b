"""Stable-Baselines3 compatible vectorized environment wrapper.

Adapts I4bVecEnvWrapper to the SB3 VecEnv protocol, enabling use of
SB3 algorithms (PPO, SAC, etc.) with i4b's GPU-accelerated environments
while keeping observations on GPU via DLPack.

SB3's VecEnv protocol expects NumPy arrays, so this wrapper converts
PyTorch tensors to NumPy on output. For fully on-GPU training, prefer
the RSL-RL wrapper or the native OnPolicyRunner instead.

Usage::

    from src.gym_interface.vec_env import RoomHeatVecEnv
    from src.rl.wrappers.sb3 import Sb3VecEnvWrapper

    jax_env = RoomHeatVecEnv(num_envs=8, ..., device="cpu")
    env = Sb3VecEnvWrapper(jax_env)
    # Use with SB3 algorithms directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

from src.rl.wrappers.base import I4bVecEnvWrapper


class Sb3VecEnvWrapper(gym.vector.VectorEnv):
    """SB3-compatible VecEnv backed by I4bVecEnvWrapper.

    Converts GPU tensors to NumPy for SB3 compatibility. The underlying
    simulation still runs on the JAX device for maximum throughput.
    """

    def __init__(
        self,
        env,
        torch_device: str = "cpu",
    ):
        """Initialize SB3 wrapper.

        Args:
            env: A RoomHeatVecEnv instance.
            torch_device: Device for intermediate torch tensors (use "cpu"
                for SB3 since it expects NumPy arrays).
        """
        self._wrapper = I4bVecEnvWrapper(env, torch_device=torch_device)
        self._env = env

        super().__init__(
            num_envs=env.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments.

        Returns:
            obs: NumPy array ``(num_envs, obs_dim)``.
            info: Empty dict.
        """
        obs = self._wrapper.reset()
        return obs.cpu().numpy(), {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step all environments.

        Args:
            actions: NumPy array ``(num_envs, action_dim)``.

        Returns:
            SB3-style 5-tuple: (obs, rewards, terminated, truncated, info).
        """
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self._wrapper.device)
        obs, rewards, dones, info = self._wrapper.step(actions_t)

        obs_np = obs.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        # SB3 expects separate terminated/truncated arrays.
        terminated_np = np.zeros(self.num_envs, dtype=bool)
        truncated_np = dones.cpu().numpy().astype(bool)

        info_np = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info_np[k] = v.cpu().numpy()
            else:
                info_np[k] = v

        return obs_np, rewards_np, terminated_np, truncated_np, info_np
