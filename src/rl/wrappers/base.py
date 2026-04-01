"""Base wrapper adapting RoomHeatVecEnv to PyTorch-based RL algorithms.

The wrapper bridges JAX (env backend) and PyTorch (RL algorithm) via DLPack
zero-copy transfers. Observations and rewards stay on GPU throughout —
no CPU staging occurs during training.

Supports configurable **privileged observations** for asymmetric actor-critic
training (teacher-student). The actor sees standard env observations while
the critic can receive additional information (building params, extended
forecasts, full thermal state) specified via ``privileged_obs_keys``.

Supports **observation history** for the actor: instead of a single 5D obs,
the actor receives a window of past observations ``[obs_t, obs_t-1, ...,
obs_t-k+1]`` flattened to ``raw_obs_dim * history_length``.  This allows an
MLP actor to infer building thermal properties from how temperatures respond
to its actions over time.

Typical usage::

    from src.gym_interface.vec_env import RoomHeatVecEnv
    from src.rl.wrappers.base import I4bVecEnvWrapper

    jax_env = RoomHeatVecEnv(num_envs=4096, ..., device="gpu")

    # Standard (symmetric) — actor and critic see same obs:
    env = I4bVecEnvWrapper(jax_env)

    # Asymmetric with history — actor gets 12-step history, critic gets
    # building params + forecast:
    env = I4bVecEnvWrapper(
        jax_env,
        obs_history_length=12,
        privileged_obs_keys=["building_params", "extended_forecast"],
        extended_forecast_steps=24,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.gym_interface.framework_export import jax_to_torch

# Building parameter keys available as privileged observations.
# These are scalar values from env_params that describe the building's
# thermal properties — not available from real sensors.
_BUILDING_PARAM_KEYS = (
    "H_ve", "H_tr", "H_tr_light", "c_bldg", "area_floor",
)


class I4bVecEnvWrapper:
    """Core JAX-to-PyTorch adapter for RoomHeatVecEnv.

    Converts JAX arrays to PyTorch tensors via DLPack and provides a
    simplified step/reset interface suitable for on-policy RL training.

    Attributes:
        env: Underlying RoomHeatVecEnv instance.
        num_envs: Number of parallel environments.
        num_obs: Observation dimension per environment (actor input).
            If ``obs_history_length > 1``, this is
            ``raw_obs_dim * obs_history_length``.
        num_privileged_obs: Privileged observation dimension (critic input),
            or None if actor and critic share the same observations.
        num_actions: Action dimension per environment.
        device: PyTorch device for output tensors.
        max_episode_length: Maximum steps per episode.
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
        """Initialize the wrapper.

        Args:
            env: A RoomHeatVecEnv instance (JAX backend).
            torch_device: PyTorch device string for output tensors.
            clip_obs: Observation clipping range [-clip_obs, clip_obs].
            clip_reward: Reward clipping range [-clip_reward, clip_reward].
            reward_scale: Multiply rewards by this factor (applied before
                clipping). Use 0.001 to convert Wh → kWh.
            comfort_penalty_weight: Weight for comfort deviation penalty
                added to the reward. The final reward becomes
                ``reward * reward_scale - comfort_penalty_weight * dev_sum``.
                Set to 0.0 (default) for energy-only reward.
            privileged_obs_keys: List of extra information sources for the
                critic. Supported keys:

                - ``"building_params"``: Building thermal properties
                  (H_ve, H_tr, H_tr_light, c_bldg, area_floor).
                - ``"extended_forecast"``: Future T_amb values beyond the
                  actor's forecast horizon. Number of steps set by
                  ``extended_forecast_steps``.
                - ``"full_state"``: All RC thermal states (redundant if
                  already in obs, but useful when actor obs is reduced).

                If None or empty, privileged_obs is disabled and the critic
                receives the same observations as the actor.
            extended_forecast_steps: Number of additional future T_amb
                steps to include in privileged obs (only used when
                ``"extended_forecast"`` is in ``privileged_obs_keys``).
            obs_history_length: Number of past observations to stack for
                the actor.  1 = current obs only (no history).  12 = 3
                hours of history at 15-min steps.  96 = full day.
                The actor's ``num_obs`` becomes
                ``raw_obs_dim * obs_history_length``.
        """
        self.env = env
        self.device = torch.device(torch_device)
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.reward_scale = reward_scale
        self.comfort_penalty_weight = comfort_penalty_weight
        self.extended_forecast_steps = extended_forecast_steps

        self.num_envs: int = env.num_envs
        self._raw_obs_dim: int = env.single_observation_space.shape[0]
        self.obs_history_length: int = max(1, obs_history_length)
        self.num_obs: int = self._raw_obs_dim * self.obs_history_length
        self.num_actions: int = env.single_action_space.shape[0]
        self.max_episode_length: int = env.p_arr.shape[0] - 1

        # Observation history buffer: (num_envs, history_length, raw_obs_dim)
        # Most recent obs is at index 0, oldest at index history_length-1.
        if self.obs_history_length > 1:
            self._obs_history = torch.zeros(
                self.num_envs, self.obs_history_length, self._raw_obs_dim,
                dtype=torch.float32, device=self.device,
            )
        else:
            self._obs_history = None

        # Configure privileged observations.
        self._privileged_keys: list[str] = list(privileged_obs_keys or [])
        self._validate_privileged_keys()
        if self._privileged_keys:
            # Privileged obs = actor obs (with history) + extra components.
            extra_dim = self._compute_privileged_extra_dim()
            self.num_privileged_obs: int | None = self.num_obs + extra_dim
            # Pre-compute static building params tensor (doesn't change per step).
            self._static_building_params = self._build_static_building_params()
        else:
            self.num_privileged_obs = None
            self._static_building_params = None

    # ------------------------------------------------------------------
    # Observation history
    # ------------------------------------------------------------------

    def _push_obs(self, obs: torch.Tensor, reset_mask: torch.Tensor | None = None):
        """Push new observation into history buffer.

        Args:
            obs: Raw observation ``(num_envs, raw_obs_dim)``.
            reset_mask: Optional bool tensor ``(num_envs,)`` — True for
                envs that just reset.  Their history is zeroed out
                (unknown past → zero).
        """
        if self._obs_history is None:
            return

        if reset_mask is not None and reset_mask.any():
            # Zero out history for reset envs — past is unknown.
            mask = reset_mask[:, None, None].expand_as(self._obs_history)
            self._obs_history = torch.where(mask, torch.zeros_like(self._obs_history), self._obs_history)

        # Shift history: drop oldest, prepend newest.
        self._obs_history = torch.cat(
            [obs.unsqueeze(1), self._obs_history[:, :-1, :]], dim=1
        )

    def _get_obs_with_history(self, obs: torch.Tensor) -> torch.Tensor:
        """Return flattened observation with history.

        Args:
            obs: Current raw observation ``(num_envs, raw_obs_dim)``.

        Returns:
            Flattened ``(num_envs, raw_obs_dim * history_length)``.
            Current obs is first, oldest is last.
        """
        if self._obs_history is None:
            return obs
        # History buffer already has current obs at index 0 (from _push_obs).
        return self._obs_history.reshape(self.num_envs, -1)

    # ------------------------------------------------------------------
    # Privileged observation configuration
    # ------------------------------------------------------------------

    def _validate_privileged_keys(self) -> None:
        """Validate that all requested privileged obs keys are supported."""
        valid = {"building_params", "extended_forecast", "full_state"}
        for key in self._privileged_keys:
            if key not in valid:
                raise ValueError(
                    f"Unknown privileged_obs_key: {key!r}. "
                    f"Supported: {sorted(valid)}"
                )
        if "extended_forecast" in self._privileged_keys and self.extended_forecast_steps <= 0:
            raise ValueError(
                "extended_forecast_steps must be > 0 when "
                "'extended_forecast' is in privileged_obs_keys"
            )

    def _compute_privileged_extra_dim(self) -> int:
        """Compute the dimension of extra privileged features."""
        dim = 0
        if "building_params" in self._privileged_keys:
            dim += len(_BUILDING_PARAM_KEYS)
        if "extended_forecast" in self._privileged_keys:
            dim += self.extended_forecast_steps
        if "full_state" in self._privileged_keys:
            dim += len(self.env.obs_keys)
        return dim

    def _build_static_building_params(self) -> torch.Tensor | None:
        """Extract building params as a static tensor (num_envs, num_params).

        These values only change on domain randomization events, so they
        are rebuilt lazily when a reset occurs.
        """
        if "building_params" not in self._privileged_keys:
            return None
        import numpy as np
        params = np.zeros(
            (self.num_envs, len(_BUILDING_PARAM_KEYS)), dtype=np.float32
        )
        for i, ep in enumerate(self.env.env_params):
            for j, key in enumerate(_BUILDING_PARAM_KEYS):
                params[i, j] = float(ep.get(key, 0.0))
        return torch.tensor(params, dtype=torch.float32, device=self.device)

    def _build_privileged_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Construct privileged observations by appending extra features.

        Args:
            actor_obs: Actor observations ``(num_envs, num_obs)`` (may
                include history).

        Returns:
            Privileged observations ``(num_envs, num_privileged_obs)``.
        """
        parts = [actor_obs]

        if "building_params" in self._privileged_keys:
            parts.append(self._static_building_params)

        if "extended_forecast" in self._privileged_keys:
            jnp = self.env.jnp
            t_arr = jnp.asarray(self.env.t, dtype=jnp.int32)
            forecasts = []
            # Start from where the actor's forecast ends.
            actor_fc_end = len(self.env.weather_forecast_steps)
            for step in range(actor_fc_end + 1, actor_fc_end + 1 + self.extended_forecast_steps):
                idx = jnp.clip(t_arr + step, 0, self.env.p_arr.shape[0] - 1)
                fc = self.env.p_arr[idx, self.env.p_env_idx, 0]  # T_amb column
                forecasts.append(self._jax_to_torch(fc))
            parts.append(torch.stack(forecasts, dim=-1))

        if "full_state" in self._privileged_keys:
            state_torch = self._jax_to_torch(self.env.state)
            parts.append(state_torch)

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # JAX <-> PyTorch conversion
    # ------------------------------------------------------------------

    def _jax_to_torch(self, arr) -> torch.Tensor:
        """Convert JAX array to PyTorch tensor via DLPack (zero-copy on GPU)."""
        return jax_to_torch(arr, device=self.device)

    def _torch_to_jax(self, tensor: torch.Tensor):
        """Convert PyTorch tensor to JAX array via DLPack (zero-copy on GPU)."""
        import jax.numpy as jnp
        return jnp.from_dlpack(tensor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Reset all environments and return observations.

        Returns:
            obs: Observation tensor ``(num_envs, num_obs)``.
            privileged_obs: Privileged observations ``(num_envs, num_privileged_obs)``
                or None if privileged obs are not configured.
        """
        obs_jax, _ = self.env.reset()
        raw_obs = torch.clamp(self._jax_to_torch(obs_jax), -self.clip_obs, self.clip_obs)

        # Zero history, then push current obs to slot 0.
        if self._obs_history is not None:
            all_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._push_obs(raw_obs, reset_mask=all_reset)

        obs = self._get_obs_with_history(raw_obs)

        # Rebuild static building params after reset (DR may have changed them).
        if self._privileged_keys and "building_params" in self._privileged_keys:
            self._static_building_params = self._build_static_building_params()

        privileged_obs = self._build_privileged_obs(obs) if self._privileged_keys else None
        return obs, privileged_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Take one step in all environments.

        Args:
            actions: Tensor of shape ``(num_envs, num_actions)`` on any device.

        Returns:
            obs: Clipped observations ``(num_envs, num_obs)``.
            privileged_obs: Privileged observations or None.
            rewards: Clipped rewards ``(num_envs,)``.
            dones: Boolean done flags ``(num_envs,)`` (terminated | truncated).
            info: Dict with keys from the underlying env (e.g. ``Q_el_kWh``).
        """
        # Clip actions to valid range [-1, 1] before passing to env.
        actions_clipped = torch.clamp(actions, -1.0, 1.0)
        actions_jax = self._torch_to_jax(actions_clipped)
        obs_jax, reward_jax, term_jax, trunc_jax, info_jax = self.env.step(actions_jax)

        raw_obs = torch.clamp(self._jax_to_torch(obs_jax), -self.clip_obs, self.clip_obs)
        rewards = self._jax_to_torch(reward_jax) * self.reward_scale
        if self.comfort_penalty_weight != 0.0:
            dev_sum = self._jax_to_torch(info_jax.get("dev_sum", reward_jax * 0.0))
            rewards = rewards - self.comfort_penalty_weight * dev_sum
        rewards = torch.clamp(rewards, -self.clip_reward, self.clip_reward)
        terminated = self._jax_to_torch(term_jax)
        truncated = self._jax_to_torch(trunc_jax)
        dones = terminated | truncated

        # Auto-reset done environments on device.
        if dones.any():
            self.env.auto_reset(self._torch_to_jax(dones))
            reset_obs_jax = self.env._build_observation(
                self.env.state,
                self.env.p_arr[0, self.env.p_env_idx, : len(self.env.p_keys)],
                p_arr_full=self.env.p_arr,
                t=self.env.t,
            )
            reset_obs = torch.clamp(
                self._jax_to_torch(reset_obs_jax), -self.clip_obs, self.clip_obs
            )
            raw_obs = torch.where(dones.unsqueeze(-1), reset_obs, raw_obs)

            # Rebuild building params for reset envs (DR changed them).
            if self._privileged_keys and "building_params" in self._privileged_keys:
                self._static_building_params = self._build_static_building_params()

        # Update history and build actor obs.
        if self._obs_history is not None:
            self._push_obs(raw_obs, reset_mask=dones if dones.any() else None)
        obs = self._get_obs_with_history(raw_obs)

        privileged_obs = self._build_privileged_obs(obs) if self._privileged_keys else None

        info = {}
        for k, v in info_jax.items():
            if isinstance(v, dict):
                info[k] = v
            else:
                try:
                    info[k] = self._jax_to_torch(v)
                except Exception:
                    info[k] = v

        return obs, privileged_obs, rewards, dones, info

    def get_observations(self) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Return current observations without stepping.

        Returns:
            obs: Observation tensor ``(num_envs, num_obs)``.
            privileged_obs: Privileged observations or None.
        """
        obs_jax = self.env._build_observation(
            self.env.state,
            self.env.p_arr[self.env.t[0], self.env.p_env_idx, : len(self.env.p_keys)],
            p_arr_full=self.env.p_arr,
            t=self.env.t,
        )
        raw_obs = torch.clamp(self._jax_to_torch(obs_jax), -self.clip_obs, self.clip_obs)
        obs = self._get_obs_with_history(raw_obs)
        privileged_obs = self._build_privileged_obs(obs) if self._privileged_keys else None
        return obs, privileged_obs
