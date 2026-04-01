"""Minimal PPO training with energy + comfort reward.

Demonstrates GPU-native PPO training on i4b with a composite reward:

    reward = -energy - comfort_penalty_weight * comfort_deviation

where ``energy`` is the electrical consumption per step (kWh) and
``comfort_deviation`` is the sum of temperature violations outside the
comfort band [20, 22] °C.

All computation stays on GPU: JAX environment → DLPack zero-copy → PyTorch.

Usage::

    JAX_PLATFORMS=cuda python examples/train_ppo_comfort.py

    # Customise reward tradeoff and training
    JAX_PLATFORMS=cuda python examples/train_ppo_comfort.py \
        --comfort-weight 0.1 \
        --num-envs 8192 \
        --iterations 500
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gym_interface.vec_env import RoomHeatVecEnv
from src.rl.wrappers.rsl_rl import RslRlVecEnvWrapper
from src.rl.algorithms.actor_critic import ActorCritic
from src.rl.algorithms.ppo import PPO, PPOConfig
from src.rl.runners.on_policy_runner import OnPolicyRunner, RunnerConfig


def main():
    parser = argparse.ArgumentParser(
        description="PPO with energy + comfort reward for room heating control"
    )
    # Environment
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--building", type=str, default="sfh_2016_now_0_soc")
    parser.add_argument("--method", type=str, default="4R3C")
    parser.add_argument("--delta-t", type=int, default=900, help="Timestep in seconds")
    parser.add_argument("--days", type=int, default=30, help="Episode length in days")

    # Reward
    parser.add_argument(
        "--comfort-weight", type=float, default=0.1,
        help="Weight for comfort deviation penalty (0 = energy-only)",
    )

    # Training
    parser.add_argument("--num-steps", type=int, default=32, help="Rollout length per update")
    parser.add_argument("--iterations", type=int, default=200, help="PPO update iterations")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="runs/ppo_comfort")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # --- 1. Create JAX environment (GPU) ---
    jax_env = RoomHeatVecEnv(
        num_envs=args.num_envs,
        building=args.building,
        hp_model="Heatpump_AW",
        method=args.method,
        mdot_HP=0.25,
        internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
        delta_t=args.delta_t,
        days=args.days,
        device="gpu",
        performance_mode=True,
    )

    # --- 2. Wrap for PyTorch with composite reward ---
    #   reward = -energy - comfort_weight * comfort_deviation
    env = RslRlVecEnvWrapper(
        jax_env,
        torch_device=args.device,
        comfort_penalty_weight=args.comfort_weight,
    )
    print(f"Env: {args.num_envs} envs | obs_dim={env.num_obs} | act_dim={env.num_actions}")
    print(f"Reward: -energy - {args.comfort_weight} * comfort_deviation")

    # --- 3. Build actor-critic + PPO ---
    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
    )
    ppo = PPO(actor_critic, cfg=PPOConfig(learning_rate=args.lr), device=args.device)

    # --- 4. Train ---
    runner = OnPolicyRunner(
        env, ppo,
        cfg=RunnerConfig(
            num_steps_per_env=args.num_steps,
            save_dir=args.save_dir,
            device=args.device,
            log_interval=1,
            save_interval=50,
        ),
    )

    total_steps = args.iterations * args.num_steps * args.num_envs
    print(f"Training: {args.iterations} iters, {total_steps:,} total env steps")
    runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
    main()
