"""Evaluation script for trained PPO agent on room heating control task."""
import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from src.gym_interface import make_room_heat_env


def evaluate(model, env, num_episodes=5):
    returns = []
    energy = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_energy = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_energy += info.get('Q_el_kWh', 0.0)
        returns.append(ep_ret)
        energy.append(ep_energy)
    return np.array(returns), np.array(energy)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent for room heating control"
    )
    parser.add_argument(
        '--building', type=str, default='sfh_2016_now_0_soc',
        help='Building model name'
    )
    parser.add_argument(
        '--hp_model', type=str, default='HPbasic',
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
        '--timestep', type=int, default=3600,
        help='Environment timestep in seconds'
    )
    parser.add_argument(
        '--days', type=int, default=30,
        help='Number of simulation days per episode'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--num_episodes', type=int, default=5,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for evaluation (auto=use GPU if available)'
    )
    
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device.upper()}")
    else:
        device = args.device
        print(f"Using {device.upper()} for evaluation")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create environment
    env = make_room_heat_env(
        building=args.building,
        hp_model=args.hp_model,
        method=args.method,
        mdot_HP=args.mdot_hp,
        internal_gain_profile=args.internal_gain_profile,
        weather_forecast_steps=[],
        timestep=args.timestep,
        days=args.days,
        random_init=False,
        noise_level=0.0,
    )

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = PPO.load(args.model_path, device=device)
    
    # Evaluate
    print(f"Evaluating for {args.num_episodes} episodes...\n")
    rets, energy = evaluate(model, env, num_episodes=args.num_episodes)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average return:      {rets.mean():8.3f} ± {rets.std():.3f}")
    print(f"Average energy:      {energy.mean():8.3f} ± {energy.std():.3f} kWh")
    print(f"Episodes evaluated:  {args.num_episodes}")
    print("="*50)


if __name__ == '__main__':
    main()



