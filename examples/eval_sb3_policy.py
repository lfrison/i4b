"""Evaluation script for trained PPO agent on room heating control task."""
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from src.gym_interface import make_room_heat_env

def evaluate(model, env, num_episodes=5):
    returns = []
    energy = []
    dev_max_neg = []  # Maximum negative temperature deviation per episode [K]
    dev_mean_neg = []  # Mean negative temperature deviation per episode [K]
    
    # Get setpoint temperature from environment (matching analyze_mpc_results.py)
    # Use unwrapped to access the underlying RoomHeatEnv instance
    setpoint_temp = env.unwrapped.bldg_model.T_room_set_lower
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_energy = 0.0
        ep_temps = []  # Track T_room values to calculate deviations
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_energy += info.get('Q_el_kWh', 0.0)
            # Track T_room for deviation calculation (matching analyze_mpc_results.py)
            ep_temps.append(info.get('T_room', 0.0))
        
        # Calculate deviations matching analyze_mpc_results.py approach
        ep_temps = np.array(ep_temps)
        # Negative deviations (below setpoint) - only count when temperature is below setpoint
        negative_deviation = np.maximum(0, setpoint_temp - ep_temps)
        
        # Mean and max negative deviation in K (matching analyze_mpc_results.py)
        ep_dev_mean = np.mean(negative_deviation) if len(negative_deviation) > 0 else 0.0
        ep_dev_max = np.max(negative_deviation) if len(negative_deviation) > 0 else 0.0
        
        returns.append(ep_ret)
        energy.append(ep_energy)
        dev_max_neg.append(ep_dev_max)
        dev_mean_neg.append(ep_dev_mean)
    
    return np.array(returns), np.array(energy), np.array(dev_max_neg), np.array(dev_mean_neg)


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
        '--delta_t', type=int, default=3600,
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
        delta_t=args.delta_t,
        days=args.days,
        random_init=False,
        noise_level=0.0,
    )

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = PPO.load(args.model_path, device=device)
    
    # Evaluate
    print(f"Evaluating for {args.num_episodes} episodes...\n")
    rets, energy, dev_max_neg, dev_mean_neg = evaluate(model, env, num_episodes=args.num_episodes)
    
    # Print results (matching analyze_mpc_results.py format)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average return:              {rets.mean():8.3f} ± {rets.std():.3f}")
    print(f"Average energy:              {energy.mean():8.3f} ± {energy.std():.3f} kWh")
    print(f"Mean neg. temp. deviation:  {dev_mean_neg.mean():8.4f} ± {dev_mean_neg.std():.4f} K")
    print(f"Max neg. temp. deviation:    {dev_max_neg.mean():8.4f} ± {dev_max_neg.std():.4f} K")
    print(f"Episodes evaluated:          {args.num_episodes}")
    print("="*50)


if __name__ == '__main__':
    main()



