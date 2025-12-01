#!/usr/bin/env python3
"""Launcher to train Saferl agents on i4b RoomHeat environments without touching core saferl."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig


def _ensure_repo_on_path() -> Path:
    """Ensure both the saferl repo root and the i4b root are on sys.path."""
    repo_root = Path(__file__).resolve().parents[3]  # .../saferl
    i4b_root = repo_root / "third_party" / "i4b"

    for p in (repo_root, i4b_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    return repo_root


def _ensure_saferl_available() -> None:
    if importlib.util.find_spec("saferl") is None:
        print(
            "Saferl is not available on PYTHONPATH. "
            "Please clone the saferl repository and add it to PYTHONPATH before running this launcher."
        )
        sys.exit(1)


def _extract_room_heat_kwargs(env_cfg: Any) -> Dict[str, Any]:
    def _get(name: str, default: Any = None) -> Any:
        # DictConfig implements get(); fallback to getattr for vanilla objects.
        if hasattr(env_cfg, "get"):
            return env_cfg.get(name, default)
        return getattr(env_cfg, name, default)

    delta_t = _get("delta_t", _get("timestep", 900))
    goal_temp_range = _get("goal_temp_range", (19.0, 28.0))
    if goal_temp_range is not None:
        goal_temp_range = tuple(goal_temp_range)

    return {
        "hp_model": _get("hp_model"),
        "building": _get("building"),
        "method": _get("method"),
        "mdot_HP": float(_get("mdot_HP")),
        "internal_gain_profile": _get("internal_gain_profile"),
        "weather_forecast_steps": list(_get("weather_forecast_steps", [])) or [],
        "delta_t": int(delta_t),
        "days": _get("days"),
        "random_init": bool(_get("random_init", False)),
        "goal_based": bool(_get("goal_based", False)),
        "goal_temp_range": goal_temp_range,
        "temp_deviation_weight": float(_get("temp_deviation_weight", 0.0)),
        "noise_level": float(_get("noise_level", 0.0)),
    }


def _patch_instantiate_env() -> None:
    from gymnasium.wrappers.flatten_observation import FlattenObservation
    from gymnasium.wrappers.rescale_action import RescaleAction
    from saferl.common.monitor import CostMonitor
    from saferl.common.utils import instantiate_env as original_instantiate_env
    from saferl.common.wrappers import SafetyGymWrapper
    from saferl.third_party.i4b.src.gym_interface.room_env import RoomHeatEnv

    def instantiate_env_room_heat(
        env_cfg: Any,
        seed: int = 0,
        norm_act: bool = True,
        min_action: float = -1.0,
        max_action: float = 1.0,
        monitor: bool = True,
        save_path: str = None,
        env_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        env_name = getattr(env_cfg, "env_name", "")
        if "RoomHeat" not in env_name:
            raise ValueError(f"Environment {env_name} is not a RoomHeat environment")

        runtime_env_kwargs: Dict[str, Any] = dict(env_kwargs or {})
        cost_dim = int(runtime_env_kwargs.pop("cost_dim", 1))
        room_heat_kwargs = _extract_room_heat_kwargs(env_cfg)

        safe_env = RoomHeatEnv(**room_heat_kwargs)
        env = SafetyGymWrapper(safe_env, cost_dim, **runtime_env_kwargs)
        env = FlattenObservation(env)

        if norm_act:
            env = RescaleAction(env, min_action, max_action)
        if monitor:
            env = CostMonitor(env, filename=save_path)

        return env

    import saferl.common.utils as saferl_utils

    saferl_utils.instantiate_env = instantiate_env_room_heat  # type: ignore


SAFERL_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = SAFERL_ROOT / "examples" / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point largely mirroring saferl.examples.main.main."""
    _ensure_repo_on_path()
    _ensure_saferl_available()
    _patch_instantiate_env()

    import os
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CallbackList
    import saferl.common.utils as saferl_utils
    from saferl.common.utils import (
        create_training_model,
        create_on_step_callback,
        evaluate_after_training,
    )
    from saferl.common.wrappers import ExtendedVecNormalize
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    print("STARTED (i4b saferl launcher)")
    set_random_seed(cfg.seed)
    save_path = os.getcwd()

    # Local copy of create_env with per-env seeding to avoid identical seeds
    def create_i4b_env(
        env_cfg,
        seed: int,
        monitor: bool,
        save_path: str,
        norm_obs: bool,
        norm_act: bool,
        norm_reward: bool,
        norm_cost: bool,
        num_env: int,
        training: bool,
        env_kwargs: dict,
        use_multi_process: bool,
    ):
        if use_multi_process:
            assert num_env > 1, "use_multi_process is set to True but num_env is not greater than 1"
            env = SubprocVecEnv(
                [
                    (lambda i=i: lambda: saferl_utils.instantiate_env(
                        env_cfg,
                        seed=seed + i,
                        norm_act=norm_act,
                        monitor=monitor,
                        save_path=save_path,
                        env_kwargs=env_kwargs,
                    ))()
                    for i in range(num_env)
                ]
            )
        else:
            env = DummyVecEnv(
                [
                    (lambda i=i: lambda: saferl_utils.instantiate_env(
                        env_cfg,
                        seed=seed + i,
                        norm_act=norm_act,
                        monitor=monitor,
                        save_path=save_path,
                        env_kwargs=env_kwargs,
                    ))()
                    for i in range(num_env)
                ]
            )
        print("All envs created (local i4b launcher)")
        if norm_obs or norm_reward or norm_cost:
            env = ExtendedVecNormalize(
                env,
                norm_reward=norm_reward,
                norm_obs=norm_obs,
                norm_cost=norm_cost,
                training=training,
            )
        return env

    env = create_i4b_env(
        cfg.env.train_env,
        cfg.seed,
        monitor=True,
        save_path=save_path,
        norm_obs=cfg.norm_obs,
        norm_act=cfg.norm_act,
        norm_reward=cfg.norm_reward,
        norm_cost=cfg.norm_cost,
        num_env=cfg.env.train_env.num_env,
        training=True,
        env_kwargs=cfg.env.train_env.env_kwargs,
        use_multi_process=cfg.use_multi_process,
    )
    print(f"{env.num_envs} training environments created")

    eval_env = create_i4b_env(
        cfg.env.eval_env,
        cfg.seed,
        monitor=False,
        save_path=save_path,
        norm_obs=cfg.norm_obs,
        norm_act=cfg.norm_act,
        norm_reward=cfg.norm_reward,
        norm_cost=cfg.norm_cost,
        num_env=1,
        training=False,
        env_kwargs=cfg.env.eval_env.env_kwargs,
        use_multi_process=False,
    )
    print(f"{eval_env.num_envs} evaluation environments created")

    # Ensure evaluation uses the correct episode length from the env config
    # (default in saferl is 1000 if _max_episode_steps is missing).
    eval_env._max_episode_steps = cfg.env.episode_len

    callback_list = []
    for key in cfg:
        if "callback" in key:
            callback_list.append(cfg[key])
    on_step_callback = CallbackList(
        [
            create_on_step_callback(
                callback.on_step_callback, eval_env=eval_env, save_path=save_path
            )
            for callback in callback_list
        ]
    )

    model = create_training_model(cfg.algorithm, env, tensorboard_log=save_path)

    if "load_model" in cfg and cfg.load_model is not None:
        model.set_parameters(cfg.load_model, device=cfg.device)

    print("Model created")
    model.learn(
        total_timesteps=cfg.env.total_timesteps,
        log_interval=1,
        tb_log_name=cfg.algorithm.algorithm_name,
        callback=on_step_callback,
    )
    print("Training finished")
    model.save(os.path.join(save_path, "model"))

    # Save normalization statistics for later evaluation
    from saferl.common.wrappers import ExtendedVecNormalize
    if isinstance(env, ExtendedVecNormalize):
        env.save(os.path.join(save_path, "env.zip"))

    if "eval_after_training_num_episodes" in cfg and cfg.eval_after_training_num_episodes:
        print(f"Evaluating the model for {cfg.eval_after_training_num_episodes} episodes")
        evaluate_after_training(
            model,
            eval_env,
            num_episodes=cfg.eval_after_training_num_episodes,
            cvar_alphas=None,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()

