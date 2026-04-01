"""GPU-native PPO training on i4b RoomHeat environments.

All computation stays on GPU: the JAX environment produces observations
that are zero-copy transferred to PyTorch via DLPack, and the PPO
algorithm runs its forward/backward passes on CUDA tensors.

Usage::

    # Default: 4096 envs, 200K total env steps
    JAX_PLATFORMS=cuda python examples/train_ppo_gpu.py

    # Custom configuration
    JAX_PLATFORMS=cuda python examples/train_ppo_gpu.py \
        --num-envs 8192 \
        --num-steps 64 \
        --iterations 500 \
        --save-dir runs/ppo_8k
"""

import argparse
import sys
from pathlib import Path

# Allow running from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gym_interface.vec_env import RoomHeatVecEnv
from src.rl.wrappers.rsl_rl import RslRlVecEnvWrapper
from src.rl.algorithms.actor_critic import ActorCritic
from src.rl.algorithms.ppo import PPO, PPOConfig
from src.rl.runners.on_policy_runner import OnPolicyRunner, RunnerConfig


def main():
    parser = argparse.ArgumentParser(description="GPU-native PPO for room heating control")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=32, help="Rollout length per env")
    parser.add_argument("--iterations", type=int, default=200, help="PPO update iterations")
    parser.add_argument("--building", type=str, default="sfh_2016_now_0_soc")
    parser.add_argument("--method", type=str, default="4R3C")
    parser.add_argument("--delta-t", type=int, default=900)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="runs/ppo_gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # --- Create JAX env ---
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
    print(f"Created JAX env: {args.num_envs} envs, obs_dim={jax_env.single_observation_space.shape[0]}")

    # --- Wrap for PyTorch ---
    env = RslRlVecEnvWrapper(jax_env, torch_device=args.device)

    # --- Build actor-critic + PPO ---
    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
    )
    ppo_cfg = PPOConfig(learning_rate=args.lr)
    ppo = PPO(actor_critic, cfg=ppo_cfg, device=args.device)

    # --- Train ---
    runner_cfg = RunnerConfig(
        num_steps_per_env=args.num_steps,
        save_dir=args.save_dir,
        device=args.device,
        log_interval=1,
        save_interval=50,
    )
    runner = OnPolicyRunner(env, ppo, cfg=runner_cfg)

    total_steps = args.iterations * args.num_steps * args.num_envs
    print(f"Starting PPO training: {args.iterations} iterations, "
          f"{total_steps:,} total env steps")
    print(f"Device: {args.device}")

    runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
    main()
