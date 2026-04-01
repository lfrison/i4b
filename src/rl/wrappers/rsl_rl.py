"""RSL-RL compatible vectorized environment wrapper.

Adapts I4bVecEnvWrapper to match the interface expected by RSL-RL's
OnPolicyRunner. This follows the same pattern used by IsaacLab/Orbit
for bridging simulation envs to RSL-RL.

The RSL-RL protocol expects:
    - ``env.get_observations()`` → ``(obs, privileged_obs)``
    - ``env.step(actions)`` → ``(obs, privileged_obs, rewards, dones, info)``
    - ``env.reset()`` → ``(obs, privileged_obs)``
    - ``env.num_envs``, ``env.num_obs``, ``env.num_actions``
    - ``env.num_privileged_obs`` (None if symmetric)
    - ``env.max_episode_length``
    - ``env.device``

Privileged observations support asymmetric actor-critic training
(teacher-student), where the critic sees extra information not available
to the actor at deployment time.

Usage::

    from src.gym_interface.vec_env import RoomHeatVecEnv
    from src.rl.wrappers.rsl_rl import RslRlVecEnvWrapper

    jax_env = RoomHeatVecEnv(num_envs=4096, ..., device="gpu")

    # Symmetric (actor = critic obs):
    env = RslRlVecEnvWrapper(jax_env)

    # Asymmetric (critic gets building params + 24-step forecast):
    env = RslRlVecEnvWrapper(
        jax_env,
        privileged_obs_keys=["building_params", "extended_forecast"],
        extended_forecast_steps=24,
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from src.rl.wrappers.base import I4bVecEnvWrapper


class RslRlVecEnvWrapper(I4bVecEnvWrapper):
    """Wrapper matching the RSL-RL VecEnv protocol.

    Extends I4bVecEnvWrapper with the privileged-observations interface
    and episode timeout tracking that RSL-RL's OnPolicyRunner expects.

    Attributes:
        num_privileged_obs: Dimension of privileged observations, or None
            if the critic uses the same observations as the actor.
        episode_length_buf: Per-env step counter for timeout detection.
    """

    def __init__(
        self,
        env,
        torch_device: str = "cuda:0",
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        reward_scale: float = 1.0,
        comfort_penalty_weight: float = 0.0,
        privileged_obs_keys: Sequence[str] | None = None,
        extended_forecast_steps: int = 0,
        obs_history_length: int = 1,
    ):
        super().__init__(
            env,
            torch_device=torch_device,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            reward_scale=reward_scale,
            comfort_penalty_weight=comfort_penalty_weight,
            privileged_obs_keys=privileged_obs_keys,
            extended_forecast_steps=extended_forecast_steps,
            obs_history_length=obs_history_length,
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reset all environments.

        Returns:
            obs: Observation tensor ``(num_envs, num_obs)``.
            privileged_obs: ``(num_envs, num_privileged_obs)`` or None.
        """
        obs, privileged_obs = super().reset()
        self.episode_length_buf.zero_()
        return obs, privileged_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step all environments with RSL-RL's expected return signature.

        Args:
            actions: Tensor ``(num_envs, num_actions)``.

        Returns:
            obs: Observations ``(num_envs, num_obs)``.
            privileged_obs: ``(num_envs, num_privileged_obs)`` or None.
            rewards: Rewards ``(num_envs,)``.
            dones: Done flags ``(num_envs,)`` (terminated | truncated).
            info: Dict with ``time_outs`` key for RSL-RL's GAE truncation handling.
        """
        obs, privileged_obs, rewards, dones, info = super().step(actions)
        self.episode_length_buf += 1

        # RSL-RL distinguishes terminal dones from timeouts for GAE bootstrapping.
        time_outs = self.episode_length_buf >= self.max_episode_length
        info["time_outs"] = time_outs

        # Reset episode counters for done envs.
        self.episode_length_buf[dones.bool()] = 0

        return obs, privileged_obs, rewards, dones, info

    def get_observations(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return current observations in RSL-RL format.

        Returns:
            obs: Observation tensor ``(num_envs, num_obs)``.
            privileged_obs: ``(num_envs, num_privileged_obs)`` or None.
        """
        return super().get_observations()
