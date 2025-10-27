import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from gym_interface import make_room_heat_env


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--building', type=str, default='sfh_2016_now_0_soc')
    parser.add_argument('--hp_model', type=str, default='HPbasic')
    parser.add_argument('--method', type=str, default='4R3C')
    parser.add_argument('--mdot_hp', type=float, default=0.25)
    parser.add_argument('--internal_gain_profile', type=str, default='data/profiles/InternalGains/ResidentialDetached.csv')
    parser.add_argument('--timestep', type=int, default=3600)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

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

    model = PPO.load(args.model_path)
    rets, energy = evaluate(model, env)
    print(f"Avg return: {rets.mean():.3f} +- {rets.std():.3f}")
    print(f"Avg energy (kWh): {energy.mean():.3f} +- {energy.std():.3f}")


if __name__ == '__main__':
    main()



