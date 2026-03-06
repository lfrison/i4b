#!/usr/bin/env python
import argparse
import time
import numpy as np

# Allow running as `python scripts/profile_step.py` from any CWD.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from src.gym_interface.vec_env import RoomHeatVecEnv


def main():
    parser = argparse.ArgumentParser(description="Profile RoomHeatVecEnv step timing.")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps-per-call", type=int, default=1)
    parser.add_argument("--batch-mode", default="stacked", choices=["vmap", "stacked"])
    parser.add_argument("--saveat-mode", default="t1", choices=["t1", "ts"])
    parser.add_argument("--norm-mode", default="max", choices=["max", "rms"])
    parser.add_argument("--fast-cost", type=int, default=1)
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--step-mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--integrator", default="auto", choices=["auto", "diffrax", "jax_rk4", "linear"])
    parser.add_argument("--performance-mode", type=int, default=1)
    parser.add_argument("--sync-time", type=int, default=1)
    parser.add_argument("--return-trajectories", type=int, default=0)
    args = parser.parse_args()

    env = RoomHeatVecEnv(
        num_envs=args.num_envs,
        building="sfh_2016_now_0_soc",
        hp_model="Heatpump_AW",
        method="4R3C",
        mdot_HP=0.25,
        internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
        delta_t=900,
        days=2,
        backend="jax",
        device=args.device,
        return_numpy=False,
        steps_per_call=args.steps_per_call,
        batch_mode=args.batch_mode,
        saveat_mode=args.saveat_mode,
        norm_mode=args.norm_mode,
        fast_cost=bool(args.fast_cost),
        integration_steps=args.integration_steps,
        step_mode=args.step_mode,
        integrator=args.integrator,
        return_trajectories=bool(args.return_trajectories),
        performance_mode=bool(args.performance_mode),
        sync_time=bool(args.sync_time),
        diagnostics=True,
    )

    obs, _ = env.reset()
    actions = np.zeros((args.num_envs, 1), dtype=np.float32)

    # Warmup
    for _ in range(args.warmup):
        env.step(actions)

    timings = {"total_s": [], "prep_s": [], "device_s": [], "block_s": []}
    for _ in range(args.steps):
        _, _, _, _, info = env.step(actions)
        t = info.get("timing", {})
        for k in timings:
            if k in t:
                timings[k].append(t[k])

    def summarize(vals):
        if not vals:
            return "n/a"
        arr = np.array(vals)
        return f"avg={arr.mean():.6f}s p50={np.percentile(arr,50):.6f}s p90={np.percentile(arr,90):.6f}s"

    print(f"device={args.device} envs={args.num_envs} steps={args.steps} warmup={args.warmup}")
    for k in ["total_s", "prep_s", "device_s", "block_s"]:
        print(f"{k}: {summarize(timings[k])}")


if __name__ == "__main__":
    main()
