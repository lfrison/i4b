"""On-policy training loop runner.

Manages the alternating rollout-collection → GAE-computation → PPO-update
cycle. Follows the RSL-RL OnPolicyRunner pattern, adapted for i4b's
JAX-to-PyTorch pipeline.

All data stays on GPU throughout: JAX env outputs are zero-copy transferred
to PyTorch via DLPack, stored in pre-allocated GPU rollout buffers, and
consumed by the PyTorch PPO optimizer.

Supports asymmetric actor-critic training: when the env provides privileged
observations, the critic receives them while the actor only sees standard
observations.

Usage::

    from src.rl.runners.on_policy_runner import OnPolicyRunner, RunnerConfig

    runner = OnPolicyRunner(env, algorithm, cfg=RunnerConfig(...))
    runner.learn(num_learning_iterations=1000)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from src.rl.storage.rollout_storage import RolloutStorage


@dataclass
class RunnerConfig:
    """Configuration for the on-policy training runner.

    Attributes:
        num_steps_per_env: Rollout length per environment per update.
        save_interval: Save checkpoint every N iterations.
        log_interval: Print metrics every N iterations.
        save_dir: Directory for checkpoints and logs.
        device: PyTorch device string.
        normalize_obs: Whether to normalize observations via running stats.
        normalize_reward: Whether to normalize rewards via running stats.
        clip_reward: Reward clipping range.
        gamma: Discount factor (for GAE).
        lam: GAE lambda.
    """

    num_steps_per_env: int = 32
    save_interval: int = 50
    log_interval: int = 1
    save_dir: str = "runs/i4b_rl"
    device: str = "cuda:0"
    normalize_obs: bool = False
    normalize_reward: bool = False
    clip_reward: float = 10.0
    gamma: float = 0.99
    lam: float = 0.95


class OnPolicyRunner:
    """Training loop for on-policy RL algorithms (PPO / PPO-Lagrangian).

    Orchestrates:
        1. Rollout collection (env.step → storage.add_transition)
        2. GAE advantage computation
        3. PPO / PPO-Lag policy update
        4. Logging and checkpointing

    Supports asymmetric actor-critic: when ``env.num_privileged_obs`` is
    set, privileged observations are stored separately and passed to the
    critic during both rollout collection and policy updates.

    Args:
        env: Wrapped environment (must have ``step()``, ``reset()``,
            ``get_observations()``, ``num_envs``, ``num_obs``, ``num_actions``).
        algorithm: PPO or PPOLag instance.
        cfg: Runner configuration.
    """

    def __init__(
        self,
        env,
        algorithm,
        cfg: RunnerConfig | None = None,
    ):
        self.cfg = cfg or RunnerConfig()
        self.env = env
        self.alg = algorithm
        self.device = torch.device(self.cfg.device)

        has_cost = hasattr(self.alg, "transition") and hasattr(
            self.alg.transition, "cost_values"
        )

        # Detect asymmetric (privileged) observations.
        num_priv = getattr(env, "num_privileged_obs", None)
        priv_obs_shape = (num_priv,) if num_priv is not None else None

        # Teacher mode: when the actor's input dim matches the privileged
        # obs dim, both actor and critic receive privileged observations.
        actor_num_obs = getattr(self.alg.actor_critic, "num_obs", env.num_obs)
        self._teacher_mode = (
            num_priv is not None and actor_num_obs == num_priv
        )

        # In teacher mode, actor receives privileged obs → store them as obs.
        # In student/asymmetric mode, actor receives standard obs.
        actor_obs_dim = num_priv if self._teacher_mode else env.num_obs
        # Privileged obs buffer is only needed for asymmetric (non-teacher) mode.
        storage_priv_shape = None if self._teacher_mode else priv_obs_shape

        self.storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=self.cfg.num_steps_per_env,
            obs_shape=(actor_obs_dim,),
            action_shape=(env.num_actions,),
            device=self.cfg.device,
            has_cost=has_cost,
            privileged_obs_shape=storage_priv_shape,
        )
        self.has_cost = has_cost
        self.has_privileged_obs = priv_obs_shape is not None

        # Optional observation/reward normalization.
        self.obs_rms = None
        self.reward_rms = None
        if self.cfg.normalize_obs:
            from src.rl.utils.running_mean_std import RunningMeanStd
            self.obs_rms = RunningMeanStd(
                shape=(env.num_obs,), device=self.cfg.device
            )
        if self.cfg.normalize_reward:
            from src.rl.utils.running_mean_std import RunningMeanStd
            self.reward_rms = RunningMeanStd(shape=(), device=self.cfg.device)

        self.current_iteration = 0
        self._logger = _MetricsLogger()

    def learn(self, num_learning_iterations: int) -> None:
        """Run the full training loop.

        Args:
            num_learning_iterations: Total number of update iterations.
        """
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        # Get initial observations.
        obs, privileged_obs = self._get_obs()
        self.alg.actor_critic.train()

        total_env_steps = 0
        start_time = time.time()

        for it in range(self.current_iteration, num_learning_iterations):
            t_iter = time.time()

            # --- Rollout collection ---
            self.storage.clear()
            for step in range(self.cfg.num_steps_per_env):
                # In teacher mode, actor receives privileged obs.
                actor_obs = privileged_obs if self._teacher_mode else obs
                critic_obs = privileged_obs if not self._teacher_mode else None
                actions = self.alg.act(actor_obs, critic_obs=critic_obs)

                step_result = self.env.step(actions)
                # Handle both 4-tuple (base) and 5-tuple (RSL-RL) returns.
                if len(step_result) == 5:
                    obs, privileged_obs, rewards, dones, info = step_result
                else:
                    obs, rewards, dones, info = step_result
                    privileged_obs = None

                if self.obs_rms is not None:
                    self.obs_rms.update(obs)
                    obs = self.obs_rms.normalize(obs)
                if self.reward_rms is not None:
                    self.reward_rms.update(rewards)
                    rewards = self.reward_rms.normalize(rewards)
                    rewards = torch.clamp(rewards, -self.cfg.clip_reward, self.cfg.clip_reward)

                # Determine what to store as actor obs.
                stored_obs = privileged_obs if self._teacher_mode else obs
                stored_priv = None if self._teacher_mode else privileged_obs

                # Cache environment step data.
                if self.has_cost:
                    costs = info.get("dev_sum", torch.zeros_like(rewards))
                    self.alg.process_env_step(rewards, dones, costs)
                    self.storage.add_transition(
                        stored_obs, self.alg.transition.actions,
                        rewards, dones,
                        self.alg.transition.values,
                        self.alg.transition.log_probs,
                        costs=costs,
                        cost_values=self.alg.transition.cost_values,
                        privileged_observations=stored_priv,
                    )
                else:
                    self.alg.process_env_step(rewards, dones)
                    self.storage.add_transition(
                        stored_obs, self.alg.transition.actions,
                        rewards, dones,
                        self.alg.transition.values,
                        self.alg.transition.log_probs,
                        privileged_observations=stored_priv,
                    )

            total_env_steps += self.cfg.num_steps_per_env * self.env.num_envs

            # --- Compute GAE ---
            last_actor_obs = privileged_obs if self._teacher_mode else obs
            last_critic_obs = privileged_obs if not self._teacher_mode else None
            with torch.no_grad():
                if self.has_cost:
                    _, last_values, _, last_cost_values = self.alg.actor_critic.act(
                        last_actor_obs, critic_obs=last_critic_obs
                    )
                else:
                    _, last_values, _ = self.alg.actor_critic.act(
                        last_actor_obs, critic_obs=last_critic_obs
                    )
                    last_cost_values = None

            self.storage.compute_returns(
                last_values, dones,
                gamma=self.cfg.gamma, lam=self.cfg.lam,
                last_cost_values=last_cost_values,
            )

            # --- Policy update ---
            progress = (it + 1) / num_learning_iterations
            if hasattr(self.alg, '_update_barrier_schedule'):
                update_stats = self.alg.update(self.storage, progress=progress)
            else:
                update_stats = self.alg.update(self.storage)

            # --- Logging ---
            iter_time = time.time() - t_iter
            fps = self.cfg.num_steps_per_env * self.env.num_envs / iter_time

            self._logger.log(
                iteration=it,
                total_env_steps=total_env_steps,
                fps=fps,
                mean_reward=self.storage.rewards.mean().item(),
                **update_stats,
            )

            if (it + 1) % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                self._logger.print_summary(it, elapsed, fps)

            # --- Checkpoint ---
            if (it + 1) % self.cfg.save_interval == 0:
                self.save(os.path.join(self.cfg.save_dir, f"model_{it + 1}.pt"))

            self.current_iteration = it + 1

        # Final save.
        self.save(os.path.join(self.cfg.save_dir, "model_final.pt"))
        total_time = time.time() - start_time
        print(f"\nTraining complete: {num_learning_iterations} iterations, "
              f"{total_env_steps:,} env steps in {total_time:.1f}s")

    def _get_obs(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get current observations, applying normalization if enabled.

        Returns:
            obs: Actor observations.
            privileged_obs: Critic observations (or None).
        """
        if hasattr(self.env, "get_observations"):
            result = self.env.get_observations()
            if isinstance(result, tuple):
                obs, privileged_obs = result
            else:
                obs, privileged_obs = result, None
        else:
            result = self.env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                obs, privileged_obs = result
            else:
                obs, privileged_obs = result, None

        if self.obs_rms is not None:
            self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)
        return obs, privileged_obs

    def save(self, path: str) -> None:
        """Save checkpoint (model + optimizer + runner state).

        Args:
            path: File path for the checkpoint.
        """
        state = {
            "actor_critic": self.alg.actor_critic.state_dict(),
            "optimizer": self.alg.optimizer.state_dict(),
            "iteration": self.current_iteration,
        }
        if self.obs_rms is not None:
            state["obs_rms"] = self.obs_rms.state_dict()
        if self.reward_rms is not None:
            state["reward_rms"] = self.reward_rms.state_dict()
        if hasattr(self.alg, "log_lagrange"):
            state["log_lagrange"] = self.alg.log_lagrange.data.cpu()
            state["lagrange_optimizer"] = self.alg.lagrange_optimizer.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load checkpoint and restore training state.

        Args:
            path: Path to a saved checkpoint.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(state["actor_critic"])
        self.alg.optimizer.load_state_dict(state["optimizer"])
        self.current_iteration = state.get("iteration", 0)
        if self.obs_rms is not None and "obs_rms" in state:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if self.reward_rms is not None and "reward_rms" in state:
            self.reward_rms.load_state_dict(state["reward_rms"])
        if hasattr(self.alg, "log_lagrange") and "log_lagrange" in state:
            self.alg.log_lagrange.data.copy_(state["log_lagrange"].to(self.device))
            self.alg.lagrange_optimizer.load_state_dict(state["lagrange_optimizer"])


class _MetricsLogger:
    """Simple training metrics accumulator."""

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def log(self, **kwargs) -> None:
        self.history.append(kwargs)

    def print_summary(self, iteration: int, elapsed: float, fps: float) -> None:
        if not self.history:
            return
        latest = self.history[-1]
        parts = [
            f"[Iter {iteration + 1}]",
            f"reward={latest.get('mean_reward', 0):.4f}",
            f"v_loss={latest.get('value_loss', 0):.4f}",
            f"s_loss={latest.get('surrogate_loss', 0):.4f}",
            f"ent={latest.get('entropy', 0):.4f}",
            f"lr={latest.get('learning_rate', 0):.2e}",
            f"fps={fps:.0f}",
            f"elapsed={elapsed:.1f}s",
        ]
        if "lagrange_multiplier" in latest:
            parts.append(f"lambda={latest['lagrange_multiplier']:.4f}")
            parts.append(f"cost={latest.get('mean_cost', 0):.4f}")
        if "barrier_loss" in latest:
            parts.append(f"barrier={latest['barrier_loss']:.4f}")
            parts.append(f"t={latest.get('barrier_factor', 0):.2f}")
            parts.append(f"cost={latest.get('mean_cost', 0):.4f}")
        print(" | ".join(parts))
