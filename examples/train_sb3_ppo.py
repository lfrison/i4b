import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Local imports
from src.gym_interface import make_room_heat_env


def make_env_fn(args):
    def _thunk():
        env = make_room_heat_env(
            building=args.building,
            hp_model=args.hp_model,
            method=args.method,
            mdot_HP=args.mdot_hp,
            internal_gain_profile=args.internal_gain_profile,
            weather_forecast_steps=[1, 2, 3] if args.forecast else [],
            timestep=args.timestep,
            days=args.days,
            random_init=args.random_init,
            goal_based=args.goal_based,
            goal_temp_range=(args.goal_temp_min, args.goal_temp_max),
            temp_deviation_weight=args.temp_deviation_weight,
            noise_level=args.obs_noise,
        )
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--building', type=str, default='sfh_2016_now_0_soc')
    parser.add_argument('--hp_model', type=str, default='Heatpump_AW')
    parser.add_argument('--method', type=str, default='4R3C')
    parser.add_argument('--mdot_hp', type=float, default=0.25)
    parser.add_argument('--internal_gain_profile', type=str, default='data/profiles/InternalGains/ResidentialDetached.csv')
    parser.add_argument('--timestep', type=int, default=3600)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--forecast', action='store_true')
    parser.add_argument('--obs_noise', type=float, default=0.0)
    # Goal-based learning parameters
    parser.add_argument('--goal_based', action='store_true', help='Enable goal-based learning')
    parser.add_argument('--goal_temp_min', type=float, default=19.0, help='Minimum goal temperature')
    parser.add_argument('--goal_temp_max', type=float, default=28.0, help='Maximum goal temperature')
    parser.add_argument('--temp_deviation_weight', type=float, default=0.0, 
                        help='Weight for temperature deviation in reward (0=disabled)')
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='runs/ppo_roomheat')
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)
    env = DummyVecEnv([make_env_fn(args)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps)
    save_path = os.path.join(args.logdir, 'ppo_roomheat')
    model.save(save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    main()



