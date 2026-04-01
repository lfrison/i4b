"""GPU-native RL training framework for i4b environments.

Provides wrappers, algorithms, and runners for fully on-device training
using JAX environments with PyTorch RL algorithms connected via DLPack
zero-copy transfers.

Modules:
    wrappers   — Env adapters for RSL-RL, SB3, and custom training loops.
    algorithms — PPO and PPO-Lagrangian with optional cost constraints.
    storage    — GPU rollout buffers with GAE computation.
    runners    — On-policy training loop managers.
    utils      — Running statistics, normalization helpers.
"""

from src.rl.wrappers.base import I4bVecEnvWrapper
from src.rl.wrappers.rsl_rl import RslRlVecEnvWrapper
from src.rl.storage.rollout_storage import RolloutStorage
from src.rl.algorithms.actor_critic import ActorCritic, SafeActorCritic
from src.rl.algorithms.ppo import PPO
from src.rl.algorithms.ppo_lag import PPOLag
from src.rl.runners.on_policy_runner import OnPolicyRunner

__all__ = [
    "I4bVecEnvWrapper",
    "RslRlVecEnvWrapper",
    "RolloutStorage",
    "ActorCritic",
    "SafeActorCritic",
    "PPO",
    "PPOLag",
    "OnPolicyRunner",
]
