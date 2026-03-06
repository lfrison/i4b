"""Benchmark: Python step-by-step vs jax.lax.scan rollout.

Shows the dispatch overhead cost and how scan eliminates it.

Metrics: env-steps/sec = n_envs * n_rollout_steps / wall_time

Run:
    # GPU (isolated subprocess sets JAX_PLATFORMS=cuda)
    python examples/benchmark_rollout.py --mode gpu

    # CPU
    python examples/benchmark_rollout.py --mode cpu

    # Both (default, runs in subprocesses)
    python examples/benchmark_rollout.py
"""
from __future__ import annotations

import os
import sys
import subprocess
import time

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_env(n_envs: int, device: str):
    from src.gym_interface.vec_env import RoomHeatVecEnv
    return RoomHeatVecEnv(
        num_envs=n_envs,
        building="sfh_2016_now_0_soc",
        hp_model="Heatpump_AW",
        method="4R3C",
        mdot_HP=0.25,
        internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
        delta_t=900,
        days=30,           # long episode so we don't truncate during rollout
        device=device,
        performance_mode=True,
        return_numpy=False,
    )


def _bench_stepwise(env, n_rollout_steps: int, repeats: int = 5):
    """Baseline: n_rollout_steps separate env.step() calls from Python."""
    import numpy as np
    import jax.numpy as jnp

    n_envs = env.num_envs
    actions = np.zeros((n_envs, 1), dtype="float32")

    # compile
    env.reset()
    obs, *_ = env.step(actions)
    obs.block_until_ready()

    times = []
    for _ in range(repeats):
        env.reset()
        t0 = time.perf_counter()
        for _ in range(n_rollout_steps):
            obs, *_ = env.step(actions)
        obs.block_until_ready()
        times.append(time.perf_counter() - t0)

    best = min(times)
    env_steps_per_sec = n_envs * n_rollout_steps / best
    return env_steps_per_sec, best


def _bench_scan(env, n_rollout_steps: int, repeats: int = 5):
    """Scan rollout: one jax.lax.scan call for the full rollout."""
    import numpy as np
    import jax.numpy as jnp

    n_envs = env.num_envs
    actions_seq = np.zeros((n_rollout_steps, n_envs), dtype="float32")

    # compile: first rollout call triggers XLA compilation
    env.reset()
    obs_seq, *_ = env.rollout(actions_seq)
    obs_seq.block_until_ready()

    times = []
    for _ in range(repeats):
        env.reset()
        t0 = time.perf_counter()
        obs_seq, *_ = env.rollout(actions_seq)
        obs_seq.block_until_ready()
        times.append(time.perf_counter() - t0)

    best = min(times)
    env_steps_per_sec = n_envs * n_rollout_steps / best
    return env_steps_per_sec, best


def run_benchmark(
    env_counts: list,
    rollout_steps_list: list,
    device: str,
    repeats: int = 5,
):
    print(f"\n{'='*72}")
    print(f"Device: {device.upper()}")
    print(f"{'='*72}")
    print(f"{'n_envs':>8}  {'n_steps':>8}  {'step/s (step)':>16}  {'step/s (scan)':>16}  {'speedup':>8}")
    print(f"{'-'*72}")

    for n_envs in env_counts:
        env = _make_env(n_envs, device)
        for n_steps in rollout_steps_list:
            sps_step, t_step = _bench_stepwise(env, n_steps, repeats)
            sps_scan, t_scan = _bench_scan(env, n_steps, repeats)
            speedup = sps_scan / sps_step
            print(
                f"{n_envs:>8}  {n_steps:>8}  {sps_step:>16,.0f}  {sps_scan:>16,.0f}  {speedup:>7.1f}x"
            )
        print()


def _run_mode(mode: str, env_counts: list, rollout_steps_list: list, repeats: int):
    run_benchmark(env_counts, rollout_steps_list, device=mode, repeats=repeats)


def _subprocess_mode(mode: str, env_counts: list, rollout_steps_list: list, repeats: int):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda" if mode == "gpu" else "cpu"
    cmd = [
        sys.executable, __file__,
        "--mode", mode,
        "--env-counts", ",".join(str(v) for v in env_counts),
        "--rollout-steps", ",".join(str(v) for v in rollout_steps_list),
        "--repeats", str(repeats),
    ]
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    if "--mode" in sys.argv:
        def _arg(flag, default):
            if flag not in sys.argv:
                return default
            idx = sys.argv.index(flag) + 1
            return sys.argv[idx] if idx < len(sys.argv) else default

        mode = _arg("--mode", "gpu")
        env_counts = [int(v) for v in _arg("--env-counts", "64,256,1024,4096").split(",") if v]
        rollout_steps_list = [int(v) for v in _arg("--rollout-steps", "1,4,16,64,256,1024,2048").split(",") if v]
        repeats = int(_arg("--repeats", "5"))
        _run_mode(mode, env_counts, rollout_steps_list, repeats)
        raise SystemExit(0)

    # Parent process: run both devices in isolated subprocesses
    env_counts = [64, 256, 1024, 4096]
    rollout_steps_list = [1, 4, 16, 64, 256, 1024, 2048]
    repeats = 5
    _subprocess_mode("cpu", env_counts, rollout_steps_list, repeats)
    _subprocess_mode("gpu", env_counts, rollout_steps_list, repeats)
