import sys
from pathlib import Path

import numpy as np

# Allow running as `python examples/compare_accuracy.py` from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.gym_interface.room_env import RoomHeatEnv


def run_env(env, steps):
    obs, _ = env.reset()
    states = []
    actions = np.zeros((1,), dtype=np.float32)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        states.append(obs[: len(env.obs_keys)].copy())
        if terminated or truncated:
            break
    return np.array(states)


def compare(setup, steps=96):
    building, method = setup
    env_cpu = RoomHeatEnv(
        building=building,
        hp_model="Heatpump_AW",
        method=method,
        mdot_HP=0.25,
        internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
        delta_t=900,
        days=2,
        backend="legacy",
    )
    env_gpu = RoomHeatEnv(
        building=building,
        hp_model="Heatpump_AW",
        method=method,
        mdot_HP=0.25,
        internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
        delta_t=900,
        days=2,
        backend="jax",
        device="cpu",
    )
    s_cpu = run_env(env_cpu, steps)
    s_gpu = run_env(env_gpu, steps)
    min_len = min(len(s_cpu), len(s_gpu))
    s_cpu = s_cpu[:min_len]
    s_gpu = s_gpu[:min_len]
    diff = np.abs(s_cpu - s_gpu)
    return diff.max(axis=0), diff.mean(axis=0)


if __name__ == "__main__":
    setups = [
        ("sfh_1958_1968_0_soc", "2R2C"),
        ("sfh_1984_1994_0_soc", "4R3C"),
        ("sfh_2016_now_0_soc", "5R4C"),
    ]
    for building, method in setups:
        max_err, mean_err = compare((building, method))
        print(f"{building} {method}: max_err={max_err} mean_err={mean_err}")
