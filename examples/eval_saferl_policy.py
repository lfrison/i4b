#!/usr/bin/env python3
"""Evaluation script for Saferl agents on i4b RoomHeat environments.

This:
  - Reconstructs the eval environment from a Saferl Hydra config (.hydra/config.yaml)
  - Restores observation normalization (obs_rms) from env.zip
  - Loads the trained model and runs evaluation episodes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def _ensure_paths() -> Path:
    """Ensure repo root and i4b root are on sys.path."""
    import sys

    repo_root = Path(__file__).resolve().parents[3]  # .../saferl
    i4b_root = repo_root / "third_party" / "i4b"
    for p in (repo_root, i4b_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return repo_root


def _patch_room_heat_env() -> None:
    """Reuse the RoomHeatEnv monkeypatch from the training launcher."""
    from saferl.third_party.i4b.examples.train_saferl_csac_lb import _patch_instantiate_env  # type: ignore

    _patch_instantiate_env()


def _load_cfg(cfg_dir: Path) -> Any:
    """Load Hydra config from the base seed directory (contains .hydra/config.yaml)."""
    cfg_path = cfg_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Could not find Hydra config at {cfg_path}")
    return OmegaConf.load(str(cfg_path))


def _build_eval_env(cfg: Any, save_path: str):
    """Create eval env and restore normalization from env.zip exactly once (no double-normalization)."""
    from saferl.common.utils import create_env, load_env
    from saferl.common.wrappers import ExtendedVecNormalize

    # Reconstruct env_kwargs exactly as in training-style evaluation:
    # take all cfg.env.eval_env fields except control/meta keys, then merge env_kwargs.
    env_kwargs = dict(cfg.env.eval_env)
    for key in ["env_name", "num_env", "env_kwargs"]:
        env_kwargs.pop(key, None)
    if hasattr(cfg.env.eval_env, "env_kwargs") and cfg.env.eval_env.env_kwargs:
        env_kwargs.update(cfg.env.eval_env.env_kwargs)

    # 1) Build a *raw* VecEnv (no normalization), just like training but with training=False
    raw_env = create_env(
        cfg.env.eval_env,
        cfg.seed,
        monitor=False,
        save_path=save_path,
        # IMPORTANT: do not wrap here, we will attach the trained ExtendedVecNormalize from env.zip
        norm_obs=False,
        norm_act=cfg.norm_act,
        norm_reward=False,
        norm_cost=False,
        num_env=1,
        training=False,
        env_kwargs=env_kwargs,
        use_multi_process=False,
    )

    # Ensure correct episode length for evaluation
    # 2) Attach the trained ExtendedVecNormalize from env.zip on top of the raw env
    model_dir = Path(save_path)
    env_norm_path = model_dir / "env.zip"
    if env_norm_path.is_file():
        print(f"Attempting to load {env_norm_path}")
        try:
            eval_env = load_env(str(env_norm_path), venv=raw_env)
            # Freeze stats during eval
            eval_env.training = False
            eval_env.norm_reward = False
            print("Environment normalization loaded successfully from env.zip!")
        except Exception as e:
            print(f"ERROR: Could not load environment normalization from env.zip: {e}")
            # Fall back to unnormalized eval
            eval_env = raw_env
            eval_env.training = False
            print("Results may be incorrect without proper normalization.")
    else:
        # No normalization file found; still ensure eval mode and use raw env
        eval_env = raw_env
        eval_env.training = False
        print("WARNING: No normalization stats found (env.zip). Using unnormalized eval env.")

    # Ensure correct episode length for evaluation
    if hasattr(cfg.env, "episode_len"):
        setattr(eval_env, "_max_episode_steps", cfg.env.episode_len)

    # Print obs_rms mean/std (first few entries) for verification
    if isinstance(eval_env, ExtendedVecNormalize) and hasattr(eval_env, "obs_rms"):
        mean = eval_env.obs_rms.mean
        var = eval_env.obs_rms.var
        std = np.sqrt(var + 1e-8)
        print(f"obs_rms mean (first 5): {mean[:5]}")
        print(f"obs_rms std  (first 5): {std[:5]}")

    return eval_env


def _load_model(cfg: Any, model_dir: Path, eval_env, device: str):
    """Load the trained Saferl model using hydra config (cost_constraint, etc.)."""
    model_path = model_dir / "model.zip"
    if not model_path.is_file():
        model_path = model_dir / "model"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model at {model_path}")

    # Instantiate a fresh model with the same hyperparameters as training
    # (including correct cost_constraint), then load weights.
    from saferl.common.utils import create_training_model

    model = create_training_model(cfg.algorithm, eval_env, device=device)
    # SB3-style weight loading; ignores hyperparams, keeps them from hydra cfg
    model.set_parameters(str(model_path), device=device)
    return model


def evaluate_policy(cfg_dir: Path, model_dir: Path, num_episodes: int, device: str) -> None:
    cfg = _load_cfg(cfg_dir)
    eval_env = _build_eval_env(cfg, save_path=str(model_dir))
    model = _load_model(cfg, model_dir, eval_env, device=device)

    # Derive episode length from env (set in _build_eval_env)
    if hasattr(eval_env, "_max_episode_steps"):
        episode_len = eval_env._max_episode_steps
    else:
        episode_len = 1000

    # Unwrap to underlying RoomHeatEnv to get setpoint temperature (like eval_sb3_policy)
    base_env = eval_env
    # VecEnv -> underlying env
    if hasattr(base_env, "envs"):
        base_env = base_env.envs[0]
    # Peel off wrappers until we reach RoomHeatEnv
    while hasattr(base_env, "env"):
        base_env = base_env.env
    # RoomHeatEnv has bldg_model with T_room_set_lower
    setpoint_temp = getattr(getattr(base_env, "bldg_model", base_env), "T_room_set_lower", 20.0)

    print(f"Evaluating for {num_episodes} episodes, episode_len={episode_len}")

    returns = []
    energy = []
    dev_max_neg = []   # Maximum negative temp deviation per episode [K]
    dev_mean_neg = []  # Mean negative temp deviation per episode [K]

    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        ep_ret = 0.0
        ep_energy = 0.0
        ep_temps = []
        steps = 0

        while not done and steps < episode_len:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)

            # Single-env VecEnv
            reward = float(rewards[0])
            info = infos[0]

            ep_ret += reward
            ep_energy += info.get("Q_el_kWh", 0.0)
            ep_temps.append(info.get("T_room", 0.0))

            done = bool(dones[0])
            steps += 1

        ep_temps = np.array(ep_temps)
        negative_deviation = np.maximum(0, setpoint_temp - ep_temps)
        ep_dev_mean = np.mean(negative_deviation) if len(negative_deviation) > 0 else 0.0
        ep_dev_max = np.max(negative_deviation) if len(negative_deviation) > 0 else 0.0

        returns.append(ep_ret)
        energy.append(ep_energy)
        dev_max_neg.append(ep_dev_max)
        dev_mean_neg.append(ep_dev_mean)

    returns = np.array(returns)
    energy = np.array(energy)
    dev_max_neg = np.array(dev_max_neg)
    dev_mean_neg = np.array(dev_mean_neg)

    print("\n" + "=" * 50)
    print("SAFERL EVALUATION RESULTS (RoomHeat)")
    print("=" * 50)
    print(f"Average return:              {returns.mean():8.3f} ± {returns.std():.3f}")
    print(f"Average energy:              {energy.mean():8.3f} ± {energy.std():.3f} kWh")
    print(f"Mean neg. temp. deviation:   {dev_mean_neg.mean():8.4f} ± {dev_mean_neg.std():.4f} K")
    print(f"Max neg. temp. deviation:    {dev_max_neg.mean():8.4f} ± {dev_max_neg.std():.4f} K")
    print(f"Episodes evaluated:          {num_episodes}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Saferl policy on i4b RoomHeat env")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to Saferl experiment directory (containing model and .hydra/config.yaml)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=None,
        help="Optional training timestep subdirectory under exp_dir/eval/ to load model/env from (e.g., 960000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation",
    )
    args = parser.parse_args()

    _ensure_paths()
    _patch_room_heat_env()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    cfg_dir = Path(args.exp_dir).resolve()
    # If eval_step is provided, load model/env from exp_dir/eval/<step>, otherwise from exp_dir
    if args.eval_step is not None:
        model_dir = cfg_dir / "eval" / str(args.eval_step)
    else:
        model_dir = cfg_dir
    evaluate_policy(cfg_dir, model_dir, num_episodes=args.num_episodes, device=device)


if __name__ == "__main__":
    main()


