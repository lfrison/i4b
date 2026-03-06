"""Benchmark: env-level parallelization throughput (env.step calls/sec).

Measures how throughput scales with the number of parallel environments using
the Python-dispatch step path (``env.step()``). Use ``benchmark_rollout.py``
for the scan-based (dispatch-free) rollout comparison.

Output columns:
  envs       – number of parallel environments
  steps/sec  – wall-clock env-steps per second (after JIT warm-up)
  compile_s  – time of the first (compilation) step in seconds

Run:
    # GPU only
    JAX_PLATFORMS=cuda python examples/benchmark_sim.py --mode gpu

    # CPU only
    JAX_PLATFORMS=cpu  python examples/benchmark_sim.py --mode cpu

    # Both CPU and GPU in isolated subprocesses (default)
    python examples/benchmark_sim.py
"""
import time
import os
import sys
import subprocess
import numpy as np

# Allow running as `python examples/benchmark_sim.py` from any CWD.
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.gym_interface.vec_env import RoomHeatVecEnv
from src.gym_interface.config import load_yaml_config, make_room_heat_vec_env_from_config


def run_benchmark(
    env_counts,
    steps=200,
    warmup=5,
    device="cpu",
    steps_per_call=10,
    batch_mode="stacked",
    saveat_mode="t1",
    norm_mode="max",
    fast_cost=True,
    integration_steps=10,
    step_mode="fixed",
    integrator="auto",
    return_trajectories=False,
    compute_metrics=False,
    sync_time=True,
    performance_mode=True,
    config=None,
):
    results = []
    for n in env_counts:
        if config is not None:
            env = make_room_heat_vec_env_from_config(
                config,
                env_key="env",
                randomization_key="domain_randomization",
                overrides={
                    "num_envs": n,
                    "device": device,
                    "steps_per_call": steps_per_call,
                    "batch_mode": batch_mode,
                    "saveat_mode": saveat_mode,
                    "norm_mode": norm_mode,
                    "fast_cost": fast_cost,
                    "integration_steps": integration_steps,
                    "step_mode": step_mode,
                    "integrator": integrator,
                    "return_trajectories": return_trajectories,
                    "compute_metrics": compute_metrics,
                    "sync_time": sync_time,
                    "performance_mode": performance_mode,
                },
            )
        else:
            env = RoomHeatVecEnv(
                num_envs=n,
                building="sfh_2016_now_0_soc",
                hp_model="Heatpump_AW",
                method="4R3C",
                mdot_HP=0.25,
                internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
                delta_t=900,
                days=2,
                backend="jax",
                device=device,
                return_numpy=False,
                steps_per_call=steps_per_call,
                batch_mode=batch_mode,
                saveat_mode=saveat_mode,
                norm_mode=norm_mode,
                fast_cost=fast_cost,
                integration_steps=integration_steps,
                step_mode=step_mode,
                integrator=integrator,
                return_trajectories=return_trajectories,
                compute_metrics=compute_metrics,
                sync_time=sync_time,
                performance_mode=performance_mode,
            )
        obs, _ = env.reset()
        actions = np.zeros((n, 1), dtype=np.float32)
        # First step (captures compile time)
        t_compile_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(actions)
        if hasattr(obs, "block_until_ready"):
            obs.block_until_ready()
        compile_s = time.perf_counter() - t_compile_start

        # Additional warmup (no timing)
        for _ in range(max(0, warmup - 1)):
            obs, reward, terminated, truncated, info = env.step(actions)
        if hasattr(obs, "block_until_ready"):
            obs.block_until_ready()

        # Timed benchmark
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(actions)
        if hasattr(obs, "block_until_ready"):
            obs.block_until_ready()
        t1 = time.perf_counter()
        results.append((n, steps / (t1 - t0), compile_s))
    return results


def _run_mode(
    mode: str,
    steps_per_call: int,
    batch_mode: str,
    saveat_mode: str,
    norm_mode: str,
    fast_cost: bool,
    integration_steps: int,
    step_mode: str,
    integrator: str,
    return_trajectories: bool,
    compute_metrics: bool,
    sync_time: bool,
    performance_mode: bool,
    steps: int,
    warmup: int,
    env_counts,
    config_path: str | None = None,
):
    config = load_yaml_config(config_path) if config_path else None
    if mode == "cpu":
        print("CPU benchmark")
        cpu = run_benchmark(
            env_counts,
            steps=steps,
            warmup=warmup,
            device="cpu",
            steps_per_call=steps_per_call,
            batch_mode=batch_mode,
            saveat_mode=saveat_mode,
            norm_mode=norm_mode,
            fast_cost=fast_cost,
            integration_steps=integration_steps,
            step_mode=step_mode,
            integrator=integrator,
            return_trajectories=return_trajectories,
            compute_metrics=compute_metrics,
            sync_time=sync_time,
            performance_mode=performance_mode,
            config=config,
        )
        for n, sps, compile_s in cpu:
            print(f"envs={n} steps/sec={sps:.2f} compile_s={compile_s:.4f}")
    elif mode == "gpu":
        print("GPU benchmark")
        gpu = run_benchmark(
            env_counts,
            steps=steps,
            warmup=warmup,
            device="gpu",
            steps_per_call=steps_per_call,
            batch_mode=batch_mode,
            saveat_mode=saveat_mode,
            norm_mode=norm_mode,
            fast_cost=fast_cost,
            integration_steps=integration_steps,
            step_mode=step_mode,
            integrator=integrator,
            return_trajectories=return_trajectories,
            compute_metrics=compute_metrics,
            sync_time=sync_time,
            performance_mode=performance_mode,
            config=config,
        )
        for n, sps, compile_s in gpu:
            print(f"envs={n} steps/sec={sps:.2f} compile_s={compile_s:.4f}")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _subprocess_mode(
    mode: str,
    steps_per_call: int,
    batch_mode: str,
    saveat_mode: str,
    norm_mode: str,
    fast_cost: bool,
    integration_steps: int,
    step_mode: str,
    integrator: str,
    return_trajectories: bool,
    compute_metrics: bool,
    sync_time: bool,
    performance_mode: bool,
    steps: int,
    warmup: int,
    env_counts,
    config_path: str | None = None,
):
    env = os.environ.copy()
    if mode == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif mode == "gpu":
        env["JAX_PLATFORMS"] = "cuda"
    cmd = [
        sys.executable,
        __file__,
        "--mode",
        mode,
        "--steps-per-call",
        str(steps_per_call),
        "--batch-mode",
        batch_mode,
        "--saveat-mode",
        saveat_mode,
        "--norm-mode",
        norm_mode,
        "--fast-cost",
        "1" if fast_cost else "0",
        "--integration-steps",
        str(integration_steps),
        "--step-mode",
        step_mode,
        "--integrator",
        integrator,
        "--return-trajectories",
        "1" if return_trajectories else "0",
        "--compute-metrics",
        "1" if compute_metrics else "0",
        "--sync-time",
        "1" if sync_time else "0",
        "--performance-mode",
        "1" if performance_mode else "0",
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--env-counts",
        ",".join(str(v) for v in env_counts),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    return subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
        def _arg_value(flag: str, default: str) -> str:
            if flag not in sys.argv:
                return default
            idx = sys.argv.index(flag) + 1
            if idx >= len(sys.argv):
                return default
            return sys.argv[idx]

        steps_per_call = int(_arg_value("--steps-per-call", "10"))
        batch_mode = _arg_value("--batch-mode", "stacked")
        saveat_mode = _arg_value("--saveat-mode", "t1")
        norm_mode = _arg_value("--norm-mode", "max")
        fast_cost = bool(int(_arg_value("--fast-cost", "1")))
        integration_steps = int(_arg_value("--integration-steps", "10"))
        step_mode = _arg_value("--step-mode", "fixed")
        integrator = _arg_value("--integrator", "auto")
        return_trajectories = bool(int(_arg_value("--return-trajectories", "0")))
        compute_metrics = bool(int(_arg_value("--compute-metrics", "0")))
        sync_time = bool(int(_arg_value("--sync-time", "1")))
        performance_mode = bool(int(_arg_value("--performance-mode", "1")))
        steps = int(_arg_value("--steps", "200"))
        warmup = int(_arg_value("--warmup", "5"))
        env_counts_str = _arg_value("--env-counts", "1,8,64,256,1024,4096")
        env_counts = [int(v) for v in env_counts_str.split(",") if v]
        config_path = _arg_value("--config", "")
        config_path = config_path if config_path else None
        _run_mode(
            mode,
            steps_per_call,
            batch_mode,
            saveat_mode,
            norm_mode,
            fast_cost,
            integration_steps,
            step_mode,
            integrator,
            return_trajectories,
            compute_metrics,
            sync_time,
            performance_mode,
            steps,
            warmup,
            env_counts,
            config_path=config_path,
        )
        raise SystemExit(0)

    # Parent process: run CPU then GPU in isolated subprocesses to control JAX backend
    steps_per_call = 10
    batch_mode = "stacked"
    saveat_mode = "t1"
    norm_mode = "max"
    fast_cost = True
    integration_steps = 10
    step_mode = "fixed"
    integrator = "auto"
    return_trajectories = False
    compute_metrics = False
    sync_time = True
    performance_mode = True
    steps = 200
    warmup = 5
    env_counts = [1, 8, 64, 256, 1024, 4096]
    config_path = None
    _subprocess_mode(
        "cpu",
        steps_per_call,
        batch_mode,
        saveat_mode,
        norm_mode,
        fast_cost,
        integration_steps,
        step_mode,
        integrator,
        return_trajectories,
        compute_metrics,
        sync_time,
        performance_mode,
        steps,
        warmup,
        env_counts,
        config_path=config_path,
    )
    _subprocess_mode(
        "gpu",
        steps_per_call,
        batch_mode,
        saveat_mode,
        norm_mode,
        fast_cost,
        integration_steps,
        step_mode,
        integrator,
        return_trajectories,
        compute_metrics,
        sync_time,
        performance_mode,
        steps,
        warmup,
        env_counts,
        config_path=config_path,
    )
