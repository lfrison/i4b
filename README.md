# i4b — Intelligence for Buildings

## GPU-Accelerated Building Heat Pump Control Simulator

i4b is a lightweight Python framework for simulating building heat pump systems using RC thermal models. It is designed for high-throughput parallel RL training and supports up to **524,288 parallel environments** at **~1.28 billion env-steps/second** on a single GPU.

Core capabilities:
- Vectorized RC thermal models (2R2C through 7R5C) running entirely on GPU via JAX/XLA
- `jax.lax.scan`-based rollout API that eliminates Python dispatch overhead
- Domain randomization with configurable per-episode or per-interval parameter events
- Zero-copy output to PyTorch, TensorFlow, and CuPy via DLPack
- Gymnasium-compatible single-env and vectorized-env interfaces
- MPC (CasADi) and heating curve controller integrations

![I4C_Grafik](https://github.com/lfrison/i4b/assets/104891971/65cce2cf-8801-45ba-811d-a965a0115c08)

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Building Models](#building-models)
4. [Disturbances](#disturbances)
5. [GPU-Accelerated Vectorized Environment](#gpu-accelerated-vectorized-environment)
6. [Scan-Based Rollout API](#scan-based-rollout-api)
7. [Domain Randomization](#domain-randomization)
8. [Framework Export (PyTorch / TF / CuPy)](#framework-export)
9. [Benchmarking](#benchmarking)
10. [Single-Env Gymnasium Interface](#single-env-gymnasium-interface)
11. [YAML Configuration](#yaml-configuration)
12. [GPU-Native PPO Training](#gpu-native-ppo-training)
13. [Training with Stable-Baselines3](#training-with-stable-baselines3)
14. [Model Predictive Control (MPC)](#model-predictive-control-mpc)
15. [License](#license)

---

## Installation

### Core dependencies

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `pandas`, `pvlib`, `gymnasium`.

### JAX (required for GPU/vectorized simulation)

```bash
# CUDA 12
pip install -U "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other CUDA versions or CPU-only installs.

### RL extras

```bash
pip install -r requirements-rl.txt
```

### Verify GPU setup

```python
import jax
print(jax.devices())  # should show CudaDevice
```

---

## Quick Start

### Single environment (CPU, standard Gymnasium loop)

Use `RoomHeatEnv` for prototyping, debugging, or integration with frameworks that expect a single Gymnasium env (e.g. Stable-Baselines3).

```python
from src.gym_interface import make_room_heat_env

env = make_room_heat_env(
    building="sfh_2016_now_0_soc",
    hp_model="Heatpump_AW",
    method="4R3C",
    mdot_HP=0.25,
    internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
    delta_t=900,       # 15-minute timestep
    days=30,           # 30-day episodes
)

obs, info = env.reset()
total_reward = 0.0

for step in range(2880):  # 30 days * 96 steps/day
    action = env.action_space.sample()              # your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode done at step {step}, total reward: {total_reward:.2f}")
        obs, info = env.reset()
        total_reward = 0.0
```

- **Action**: scalar in [-1, 1], mapped to supply temperature [20, 65] °C
- **Observation**: thermal states + [T_amb, Qdot_gains] (+ optional forecast / goal)
- **Reward**: negative electrical energy consumption per step (kWh)
- **info**: `Q_el_kWh`, `dev_sum`, `dev_max`, `T_room`, `u` (applied setpoint)
- **Output type**: `jax.Array` by default (backend="jax"). Set `return_numpy=True` for NumPy arrays

### Vectorized environment (GPU, high-throughput training)

Use `RoomHeatVecEnv` for GPU-parallel training. All arrays stay on device — no host transfers during rollout.

```python
import jax.numpy as jnp
from src.gym_interface.vec_env import RoomHeatVecEnv
from src.gym_interface import jax_to_torch  # optional: zero-copy to PyTorch

env = RoomHeatVecEnv(
    num_envs=4096,
    building="sfh_2016_now_0_soc",
    hp_model="Heatpump_AW",
    method="4R3C",
    mdot_HP=0.25,
    internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
    delta_t=900,
    days=30,
    device="gpu",
    performance_mode=True,
)

obs, info = env.reset()  # obs: jax.Array (4096, obs_dim)

# By default all outputs (obs, reward, terminated, truncated) are jax.Array.
# Set return_numpy=True in the constructor to get NumPy arrays instead.

# --- Option A: step-by-step loop ---
for step in range(100):
    actions = jnp.zeros((4096,))                         # your policy here
    obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated | truncated
    env.auto_reset(done)                                  # reset finished envs in-place

# --- Option B: scan rollout (much faster, no Python dispatch per step) ---
actions_seq = jnp.zeros((32, 4096))                       # (n_steps, n_envs)
obs_seq, reward_seq, info = env.rollout(actions_seq)
# obs_seq: (32, 4096, obs_dim), reward_seq: (32, 4096)

# Zero-copy GPU transfer to PyTorch for policy update
obs_torch = jax_to_torch(obs_seq)   # torch.Tensor on cuda:0
```

---

## Building Models

Building parameters are stored in `/data/buildings/` as Python dicts. The framework includes single-family homes from the German TABULA dataset across construction periods 1919–present, in three refurbishment states (`_0_soc` state-of-construction, `_1_enev` EnEV-standard, `_2_kfw` KfW-standard).

RC model variants and their state vectors:

| Method | States | Notes |
|--------|--------|-------|
| `2R2C` | T_room, T_hp_ret | Minimal model |
| `4R3C` | T_room, T_wall, T_hp_ret | Default, best speed/accuracy tradeoff |
| `5R4C` | T_room, T_int, T_wall, T_hp_ret | Interior mass |
| `6R4C` | T_room, T_surf, T_op, T_mass, T_hp_ret | ISO 13790 |
| `7R5C` | T_room, T_surf_wall, T_op, T_mass, T_surf_floor, T_hp_ret | Underfloor heating |

Custom buildings can be added as a `.py` file with keys from the [TABULA web tool](https://webtool.building-typology.eu): `H_ve`, `H_tr`, `H_tr_light`, `c_bldg`, `area_floor`, `height_room`, `windows`, `position`.

---

## Disturbances

Disturbance profiles are loaded per building from weather data (via `pvlib`) and internal gains CSVs:

- **T_amb**: Ambient temperature [°C]
- **Qdot_gains**: Combined solar + internal heat gains [W]
- **Qdot_int / Qdot_sol**: Split components (6R4C and 7R5C only)

The vectorized env stores only **unique disturbance profiles** in GPU memory regardless of env count. For a homogeneous batch (all envs same building and location), a single `(T, 1, F)` profile array is stored and broadcast via an index gather — reducing VRAM by `n_envs×` compared to pre-broadcast storage.

---

## GPU-Accelerated Vectorized Environment

`RoomHeatVecEnv` is the primary interface for GPU-parallel RL training. All state, parameter, and disturbance arrays live on-device between calls; no host-device transfers occur during rollout.

```python
from src.gym_interface.vec_env import RoomHeatVecEnv

env = RoomHeatVecEnv(
    num_envs=4096,
    building="sfh_2016_now_0_soc",
    hp_model="Heatpump_AW",
    method="4R3C",
    mdot_HP=0.25,
    internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
    delta_t=900,        # 15-minute timestep
    days=30,
    device="gpu",       # "cpu" or "gpu"
    integrator="auto",  # "auto" resolves to "linear" for RC models
    performance_mode=True,
)

obs, info = env.reset()                     # (4096, obs_dim)
actions = jnp.zeros((4096, 1))
obs, reward, terminated, truncated, info = env.step(actions)
```

### Key constructor parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_envs` | Number of parallel environments | — |
| `device` | `"gpu"` or `"cpu"` | `"cpu"` |
| `integrator` | `"auto"` / `"linear"` / `"jax_rk4"` / `"diffrax"` | `"auto"` |
| `performance_mode` | Enable all throughput-oriented defaults | `False` |
| `weather_forecast_steps` | List of future T_amb offsets to append to obs | `[]` |
| `randomization_events` | List of `EventSpec` for domain randomization | `None` |
| `return_numpy` | Convert outputs to NumPy before returning | `False` |

### Linear integrator (default for RC models)

When `integrator="linear"` (or `"auto"` with an RC model), exact zero-order-hold discretization is used:

```
x[t+1] = Ad @ x[t] + Bd * u[t] + Cd @ p[t]
```

Matrices `(Ad, Bd, Cd)` are computed once at init via GPU-parallel matrix exponential (`jax.vmap(jax.scipy.linalg.expm)`). Updating building parameters (domain randomization) recomputes matrices but reuses the compiled XLA kernel as long as array shapes are unchanged.

---

## Scan-Based Rollout API

Calling `env.step()` in a Python loop pays ~1 ms Python→JAX dispatch overhead per call regardless of `num_envs`. For large rollouts, use `env.rollout()` instead: it compiles the entire loop into a single XLA kernel via `jax.lax.scan`, issuing one dispatch for any rollout length.

```python
import jax.numpy as jnp

# actions_seq: (n_rollout_steps, n_envs)
actions_seq = jnp.zeros((32, 4096))

obs_seq, reward_seq, info = env.rollout(actions_seq)
# obs_seq:    (32, 4096, obs_dim)
# reward_seq: (32, 4096)
```

Throughput comparison at `n_steps=32` on a single GPU:

| n_envs | step-by-step | scan rollout | speedup |
|--------|-------------|-------------|---------|
| 1 024 | 1.3M/s | 68M/s | 52x |
| 4 096 | 5.3M/s | 267M/s | 50x |
| 16 384 | 21M/s | 746M/s | 36x |
| 65 536 | 58M/s | **1.16B/s** | 20x |

GPU vs CPU at peak (scan, 65K envs): **~30x faster**.

---

## Domain Randomization

Parameter randomization can be applied at episode start, on reset, or at fixed intervals using `EventSpec` objects.

```python
from src.randomization import EventSpec
from src.randomization.events import uniform, loguniform

events = [
    # Randomize insulation at episode start
    EventSpec(when="on_start",    fn=uniform("H_tr", 50, 300)),
    # Randomize occupancy gain at every reset
    EventSpec(when="on_reset",    fn=loguniform("c_bldg", 50, 200)),
    # Randomize COP scale every 96 steps (~1 day at 15-min timestep)
    EventSpec(when="on_interval", fn=uniform("cop_scale", 0.8, 1.2), interval=96),
]

env = RoomHeatVecEnv(
    num_envs=1024,
    ...,
    randomization_events=events,
    randomization_seed=42,
)
```

Parameters that only affect cost/control logic (`T_amb_lim`, `T_offset`, `T_room_set_lower`, `T_room_set_upper`, `cop_scale`) do not require matrix recomputation when changed — the engine detects this automatically.

---

## Framework Export

JAX GPU arrays can be shared with PyTorch, TensorFlow, or CuPy **without any CPU round-trip** via the DLPack protocol.

```python
from src.gym_interface import jax_to_torch, jax_to_tf, jax_to_cupy

obs_seq, reward_seq, info = env.rollout(actions_seq)

# Zero-copy to PyTorch CUDA tensor
obs_torch   = jax_to_torch(obs_seq)     # torch.Tensor on cuda:0
reward_torch = jax_to_torch(reward_seq)

# Zero-copy to TensorFlow GPU tensor
obs_tf = jax_to_tf(obs_seq)

# Zero-copy to CuPy array
obs_cupy = jax_to_cupy(obs_seq)
```

This makes i4b compatible with any GPU-native RL framework (CleanRL, TorchRL, sample-factory, etc.) without framework lock-in.

---

## Benchmarking

Two benchmark scripts are provided. Both isolate JAX backends in subprocesses to avoid cross-contamination.

### Env-level parallelization (step throughput)

```bash
# GPU
JAX_PLATFORMS=cuda python examples/benchmark_sim.py --mode gpu

# CPU
JAX_PLATFORMS=cpu python examples/benchmark_sim.py --mode cpu

# Both (default, uses subprocesses)
python examples/benchmark_sim.py
```

### Scan rollout vs step-by-step

```bash
# Extended env sweep at fixed rollout length
JAX_PLATFORMS=cuda python examples/benchmark_rollout.py \
  --mode gpu \
  --env-counts 1024,4096,16384,65536 \
  --rollout-steps 32 \
  --repeats 5
```

---

## Single-Env Gymnasium Interface

For single-environment use, `RoomHeatEnv` provides a standard Gymnasium interface:

```python
from src.gym_interface import make_room_heat_env

env = make_room_heat_env(
    building="sfh_2016_now_0_soc",
    hp_model="Heatpump_AW",
    method="4R3C",
    mdot_HP=0.25,
    internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
    delta_t=900,
    days=30,
)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

- **Action space**: `Box([-1], [1])` — mapped to supply temperature in [20, 65] °C
- **Observation**: thermal states + `[T_amb, Qdot_gains]` + optional T_amb forecast steps
- **Reward**: negative electrical energy consumption per step (kWh)

---

## YAML Configuration

Environments and domain randomization can be fully specified in YAML:

```yaml
# examples/configs/roomheat_vec.yaml
env:
  num_envs: 1024
  building: sfh_2016_now_0_soc
  hp_model: Heatpump_AW
  method: 4R3C
  mdot_HP: 0.25
  internal_gain_profile: data/profiles/InternalGains/ResidentialDetached.csv
  delta_t: 900
  days: 30
  device: gpu
  performance_mode: true

domain_randomization:
  seed: 42
  events:
    - when: on_reset
      target: H_tr
      distribution: uniform
      low: 50
      high: 300
    - when: on_interval
      interval: 96
      target: cop_scale
      distribution: uniform
      low: 0.8
      high: 1.2
```

```python
from src.gym_interface.config import make_env_from_config

env = make_env_from_config("examples/configs/roomheat_vec.yaml")
```

`make_env_from_config` dispatches to `RoomHeatVecEnv` or `RoomHeatEnv` based on whether `num_envs` is present.

---

## GPU-Native PPO Training

The `src/rl/` module provides a self-contained, RSL-RL-compatible training stack that runs entirely on GPU. No CPU staging occurs between the JAX environment and the PyTorch policy — observations are transferred via DLPack zero-copy.

### Reward function

The default environment reward is negative electrical energy consumption per step (`-E_el` in kWh). The wrapper supports composing a comfort penalty:

```
reward = -energy - comfort_penalty_weight * comfort_deviation
```

where `comfort_deviation` (`dev_sum`) measures how far the room temperature falls outside the comfort band [20, 22] °C. Tuning `comfort_penalty_weight` controls the energy-vs-comfort tradeoff.

### Minimal training example

```python
from src.gym_interface.vec_env import RoomHeatVecEnv
from src.rl.wrappers.rsl_rl import RslRlVecEnvWrapper
from src.rl.algorithms.actor_critic import ActorCritic
from src.rl.algorithms.ppo import PPO, PPOConfig
from src.rl.runners.on_policy_runner import OnPolicyRunner, RunnerConfig

# 1. JAX environment on GPU
jax_env = RoomHeatVecEnv(
    num_envs=4096,
    building="sfh_2016_now_0_soc",
    hp_model="Heatpump_AW",
    method="4R3C",
    mdot_HP=0.25,
    internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
    delta_t=900,
    days=30,
    device="gpu",
    performance_mode=True,
)

# 2. Wrap for PyTorch with composite reward
env = RslRlVecEnvWrapper(
    jax_env,
    torch_device="cuda:0",
    comfort_penalty_weight=0.1,  # energy + 0.1 * comfort cost
)

# 3. Actor-critic + PPO
actor_critic = ActorCritic(
    num_obs=env.num_obs,
    num_actions=env.num_actions,
    actor_hidden_dims=(256, 256, 256),
    critic_hidden_dims=(256, 256, 256),
)
ppo = PPO(actor_critic, cfg=PPOConfig(learning_rate=3e-4), device="cuda:0")

# 4. Train
runner = OnPolicyRunner(
    env, ppo,
    cfg=RunnerConfig(num_steps_per_env=32, save_dir="runs/ppo_comfort"),
)
runner.learn(num_learning_iterations=200)
```

### Command-line script

```bash
# Default: 4096 envs, comfort_weight=0.1, 200 iterations
JAX_PLATFORMS=cuda python examples/train_ppo_comfort.py

# Energy-only reward (no comfort penalty)
JAX_PLATFORMS=cuda python examples/train_ppo_comfort.py --comfort-weight 0.0

# Larger batch, longer training
JAX_PLATFORMS=cuda python examples/train_ppo_comfort.py \
    --num-envs 8192 --iterations 500 --comfort-weight 0.5
```

Install RL extras first: `pip install -r requirements-rl.txt`.

---

## Training with Stable-Baselines3

```bash
python -m examples.train_sb3_ppo --config examples/configs/roomheat_sb3.yaml
```

Evaluation:

```bash
python -m examples.eval_sb3_policy --config examples/configs/roomheat_sb3.yaml
```

Install RL extras first: `pip install -r requirements-rl.txt`.

---

## Model Predictive Control (MPC)

CasADi-based MPC minimizes energy consumption subject to comfort constraints over a receding horizon.

```bash
python -m examples.run_mpc
```

Key parameters in the script: building, heat pump model, method, timestep (`delta_t`), MPC horizon (`nk`), mass flow rate (`mdot_hp`), comfort setpoint (`T_room_set_lower`).

Results are saved to `results_mpc/`. Summary statistics (energy, cost, comfort deviation) are printed when `mpc_steps >= 24`.

Install CasADi: `pip install casadi`.

---

## License

Licensed under the BSD 3-Clause License.
