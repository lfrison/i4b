"""Training script for PPO agent on room heating control task."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# Allow running as `python examples/train_sb3_ppo.py` from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.gym_interface import make_room_heat_env
from src.gym_interface.config import load_yaml_config, make_room_heat_env_from_config

def make_env_fn(args, cfg=None):
    if cfg is not None:
        env = make_room_heat_env_from_config(
            cfg,
            env_key="env",
            randomization_key="domain_randomization",
        )
    else:
        env = make_room_heat_env(
            building=args.building,
            hp_model=args.hp_model,
            method=args.method,
            mdot_HP=args.mdot_hp,
            internal_gain_profile=args.internal_gain_profile,
            weather_forecast_steps=list(range(1, args.forecast + 1)) if args.forecast > 0 else [],
            delta_t=args.delta_t,
            days=args.days,
            random_init=args.random_init,
            goal_based=args.goal_based,
            goal_temp_range=(args.goal_temp_min, args.goal_temp_max),
            temp_deviation_weight=args.temp_deviation_weight,
            noise_level=args.obs_noise,
        )
    env = Monitor(env)
    return env

def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for room heating control"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config. If set, env/training settings are read from YAML."
    )
    # Environment parameters
    parser.add_argument(
        '--building', type=str, default='sfh_2016_now_0_soc',
        help='Building model name'
    )
    parser.add_argument(
        '--hp_model', type=str, default='Heatpump_AW',
        help='Heat pump model type'
    )
    parser.add_argument(
        '--method', type=str, default='4R3C',
        help='Building thermal model'
    )
    parser.add_argument(
        '--mdot_hp', type=float, default=0.25,
        help='Heat pump mass flow [kg/s]'
    )
    parser.add_argument(
        '--internal_gain_profile', type=str,
        default='data/profiles/InternalGains/ResidentialDetached.csv',
        help='Internal gains profile path'
    )
    parser.add_argument(
        '--delta_t', type=int, default=900,
        help='Environment timestep in seconds, if 3600, \
            each timestep in the simulation is 1 hour, \
            if 900, each timestep in the simulation is 15 minutes'
    )
    parser.add_argument(
        '--days', type=int, default=30,
        help='Number of simulation days per episode'
    )
    parser.add_argument(
        '--random_init', action='store_true',
        help='Use random initial conditions'
    )
    parser.add_argument(
        '--forecast', type=int, default=0,
        help='Number of forecast steps to include in observations (0 = no forecast, >0 = number of steps ahead)'
    )
    parser.add_argument(
        '--obs_noise', type=float, default=0.0,
        help='Observation noise standard deviation'
    )
    
    # Goal-based learning parameters
    parser.add_argument(
        '--goal_based', action='store_true',
        help='Enable goal-based learning'
    )
    parser.add_argument(
        '--goal_temp_min', type=float, default=19.0,
        help='Minimum goal temperature [C]'
    )
    parser.add_argument(
        '--goal_temp_max', type=float, default=28.0,
        help='Maximum goal temperature [C]'
    )
    parser.add_argument(
        '--temp_deviation_weight', type=float, default=5.0,
        help='Weight for temperature deviation in reward (0=disabled)'
    )
    
    # Training parameters
    parser.add_argument(
        '--total_timesteps', type=int, default=200_000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--logdir', type=str, default='runs/ppo_roomheat',
        help='Directory for logs and saved models'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training (auto=use GPU if available)'
    )
    
    args = parser.parse_args()

    cfg = load_yaml_config(args.config) if args.config else None
    training_cfg = cfg.get("training", {}) if cfg else {}

    seed = int(training_cfg.get("seed", args.seed))
    total_timesteps = int(training_cfg.get("total_timesteps", args.total_timesteps))
    logdir = str(training_cfg.get("logdir", args.logdir))
    requested_device = str(training_cfg.get("device", args.device))
    model_name = str(training_cfg.get("model_name", "ppo_roomheat"))

    # Set random seed
    np.random.seed(seed)

    # Determine device
    if requested_device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device.upper()}")
    else:
        device = requested_device
        print(f"Using {device.upper()} for training")
    
    # Set CUDA seed and show GPU info if using GPU
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create environment
    os.makedirs(logdir, exist_ok=True)
    env = DummyVecEnv([lambda: make_env_fn(args, cfg=cfg)])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        seed=seed,
        device=device,
    )

    # Train model
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    save_path = os.path.join(logdir, model_name)
    model.save(save_path)
    print(f"\n✓ Training complete! Model saved to {save_path}")


if __name__ == '__main__':
    main()
