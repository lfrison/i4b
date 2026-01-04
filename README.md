# i4b - Intelligence for Buildings 

## Advanced Building Heat Pump Control Testing Framework

This project features a light-weight Python-based thermal simulation framework for heat pump operation in buildings. It is particularly useful for the following tasks:
- Serves as evaluation framework for testing different building heat pump control strategies
- Features different detailed reduced-order models for implementing building heat pump controllers (e.g., MPC)
- Serves as synthetic data generation framework, e.g. for ML-based controllers or anomaly detection

This Python project facilitates the quick generation of reduced order building models. It features a simulator class providing a high-level interface for one-step and multi-step simulations. These simulations return the next state(s) of the building (temperatures), indicators for comfort levels, and energy demand. This interface can be used to evaluate and test different control strategies. Interface to RL, MPC, and reference heat curve controller is provided. Implementation for MPC and reference heat curve is given. The project includes simple heat pump models based on performance curves for heating systems, and disturbance profiles for ambient temperature, internal heat gains by occupancy, and solar heat gains.

![I4C_Grafik](https://github.com/lfrison/i4b/assets/104891971/65cce2cf-8801-45ba-811d-a965a0115c08)

## Table of Contents

1. [Install Dependencies](#install-dependencies)
2. [Building Model](#building-models)
3. [Disturbances](#disturbances)
4. [Controller](#controller)
5. [Model Predictive Control (MPC)](#model-predictive-control-mpc)
6. [Gymnasium Interface (RL)](#gymnasium-interface-rl)
7. [Training with Stable-Baselines3](#training-with-stable-baselines3)
8. [Using Saferl with i4b (Safe RL)](#using-saferl-with-i4b-safe-rl)
9. [Evaluation](#evaluation)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Install Dependencies

Ensure you have the following dependencies installed:

- `numpy`
- `pandas`
- `pvlib`
- `casadi` (for MPC)

You can install these dependencies using pip:

`pip install -r requirements.txt`

For reinforcement learning examples and tools, install the optional extras:

`pip install -r requirements-rl.txt`

For Jupyter notebooks (located in `/notebooks`), install the optional notebook dependencies:

`pip install -r notebooks/requirements-notebooks.txt`

## Building Models

Geometrical and physical parameters that specify different buildings are available in the `/data/buildings` directory, including:

- `i4c_building`: an energy-efficient KFW 40+ house. Source: EnEV-Nachweis.
- `sfh_58_68_geg`: a single-family home constructed between 1958-68, with an envelope refurbished according to GEG regulations. Source: Tabula.
- `sfh_84_94_soc`: a single-family home constructed between 1984-94, with an envelope in the state of construction. Source: Tabula.

Building models are reduced order models based on RC networks. Different degrees of modelling depths can be selected, e.g., 2R2C, 4R3C.

Building specifications can be added by creating separate `.py` files containing a dictionary with the following parameters, which are accessible via the [TABULA web tool](https://webtool.building-typology.eu):

- `H_ve`: Heat transfer coefficient for ventilation (indoors --> ambient) [W/K]
- `H_tr`: Heat transfer coefficient for transmission (indoors --> ambient) [W/K]
- `H_tr_light`: Heat transfer coefficient for light building components (indoors --> ambient) [W/K]
- `c_bldg`: Specific heat capacity of the building [Wh/m²/K]
- `area_floor`: Conditioned floor area [m²]
- `height_room`: Average height of the heated zone [m³]
- `T_offset`: Optional parameter, used for heating curve control [°C]
- `windows`: List of dictionaries containing:
  - `area`: Absolute window area [m²]
  - `tilt`: Tilt angle [degree]
  - `azimuth`: Azimuth angle, 0 = North, 180 = South [degree]
  - `g-value`: Total solar heat gain factor [-]
  - `c_frame`: Fraction of window that is opaque due to the frame [-]
  - `c_shade`: Shading factor due to external influences, e.g., trees [-]
- `position`: Dictionary containing:
  - `lat`: Latitude [degree]
  - `long`: Longitude [degree]
  - `altitude`: Altitude above sea level [m]
  - `timezone`: Timezone as defined [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

## Disturbances

Functions are provided to generate disturbance profiles for:

- Ambient Temperature [°C]
- Internal heat gains by occupancy and appliances [W]
- Solar heat gains through transparent building elements [W]

These functions generate `pandas` dataframes, where the columns correspond to individual disturbances, and the index is a `pandas.DatetimeIndex`.

To manually generate disturbance profiles, start with the weather data and give each entry a `pandas.DatetimeIndex`. The datetime index is required to generate the internal and solar heat gain profiles. For the solar heat gain profiles, the weather dataframe should also contain information about solar irradiation.

## Controller

Heating curves are a widely used approach to control heating temperatures based on outdoor temperature to maintain constant room temperature. In this framework, the heating curve controller can be used as a baseline.
MPC controller using [CasADi](https://web.casadi.org/) is available.
Gymnasium-compatible RL interface is provided (see below).

![i4c-cntroller](https://github.com/lfrison/i4b/assets/104891971/87e45eff-9fea-4771-a8bc-ead693e322ee)

## Model Predictive Control (MPC)

An MPC controller implementation using [CasADi](https://web.casadi.org/) is provided. The MPC controller optimizes the heat pump supply temperature to minimize energy consumption while maintaining comfort constraints.

### Running MPC Experiments

To run an MPC simulation example, execute:

```bash
python -m examples.run_mpc
```

Key parameters to configure in the script:

- **Building model**: Select from available buildings (e.g., `sfh_1984_1994_1_enev`, `sfh_1958_1968_0_soc`, `i4c`)
- **Heat pump model**: Choose `Heatpump_AW` (air-water) or `Heatpump_Vitocal` (ground-source)
- **Method**: Building thermal model (`2R2C`, `4R3C`, or `5R4C`)
- **Timestep** (`h` or `delta_t`): Simulation timestep in seconds (e.g., `3600` for 1 hour, `900` for 15 minutes)
- **MPC steps** (`mpc_steps`): Total number of MPC iterations to run
- **Prediction horizon** (`nk`): Optimization horizon in timesteps (default: `24*int(3600/h)` for 1 day)
- **Mass flow** (`mdot_hp`): Heat pump mass flow rate [kg/s] (default: `0.25`)
- **Comfort setpoint**: `T_room_set_lower`[°C] Lowest room temperature to be defined as comfortable

Results are saved to `results_mpc/` directory. The script automatically evaluates and prints summary statistics if `mpc_steps >= 24`, including:
- Thermal and electrical energy consumption (kWh)
- Total cost (grid impact)
- Average and maximum comfort deviation (K)

## Gymnasium Interface (RL)

The module `src/gym_interface/room_env.py` exposes the building thermal simulation as a Gymnasium environment `RoomHeatEnv`. A convenience factory and registration helpers are provided in `src/gym_interface/__init__.py`.

- **Action space**: `Box([-1], [1])` (normalized). Mapped to supply flow temperature setpoint in °C within `[action_low, action_high]`, where `action_low = 20` degrees and `action_high = 65` degrees.
- **Observation**: Building state vector + current `T_amb`, `Qdot_gains` and optional future `T_amb` forecasts.
- **API**: Returns `(obs, reward, terminated, truncated, info)` following Gymnasium.

Quickstart (Python):

```python
from gym_interface import make_room_heat_env

env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    weather_forecast_steps=[],  # e.g., [1,2,3] to append future T_amb steps
    delta_t=900,  # timestep in seconds (900 = 15 minutes, 3600 = 1 hour)
    days=30,
    random_init=False,
    noise_level=0.0,
)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Direct instantiation (advanced users) is also supported by importing `RoomHeatEnv` and passing explicit constructor args.

## Training with Stable-Baselines3

We provide a minimal PPO training script:

```bash
python -m examples.train_sb3_ppo \
  --building sfh_2016_now_0_soc \
  --hp_model Heatpump_AW \
  --method 4R3C \
  --mdot_hp 0.25 \
  --internal_gain_profile data/profiles/InternalGains/ResidentialDetached.csv \
  --delta_t 900 \
  --days 30 \
  --total_timesteps 200000 \
  --logdir runs/ppo_roomheat
```

Notes:
- Install RL extras first: `pip install -r requirements-rl.txt`.
- Adjust `--forecast` to include simple ambient temperature forecasts.

## Using Saferl with i4b (Safe RL)

We provide a small integration layer in `examples/saferl_support/` that lets you train Safe RL agents from the
Saferl library on the `RoomHeatEnv` simulator while keeping i4b self-contained.

### 1. Clone Saferl and set up its environment

```bash
git clone git@github.com:nrgrp/saferl-lib.git
cd saferl-lib
conda create -n saferl python=3.10  # or use venv
conda activate saferl
```

Following the instruction in saferl to install it.

### 2. Clone i4b under Saferl `third_party`

From the `saferl-lib` repo direcotry:

```bash
mkdir -p saferl/third_party
cd saferl/third_party
git clone https://github.com/lfrison/i4b.git i4b
cd ../../
```

### 3. Install additional i4b dependencies for Safe RL

Install the core i4b requirements plus the Saferl-specific extras:

```bash
pip install -r saferl/third_party/i4b/examples/saferl_support/requirement_saferl.txt
```

### 4. Copy Saferl experiment configs

The directory `saferl/third_party/i4b/examples/saferl_support/` contains example Hydra env configs for Saferl.
Copy them into Saferl’s env config directory:

```bash
cp saferl/third_party/i4b/examples/saferl_support/RoomHeat*.yaml \
   saferl/examples/configs/env/
```

This exposes `RoomHeat1-v0`, `RoomHeat1-v1`, `RoomHeat2-v0`, and `RoomHeat2-v1` as selectable `env=...` options in
the Saferl CLI.

### 5. Train a CSAC-LB agent on `RoomHeat1-v0`

From the Saferl repo root:

```bash
cd saferl-lib
export PYTHONPATH=$PWD:$PYTHONPATH

python -m saferl.third_party.i4b.examples.train_saferl_csac_lb \
  env=RoomHeat1-v0 \
  algorithm=csac_lb \
  num_env=5 \
  seed=1 \
  env.total_timesteps=960000 \
  algorithm.model.cost_constraint="[10.0]" \
  algorithm.model.log_barrier_factor=10 \
  algorithm.model.log_barrier_multipier=0.1 \
  save_video=False \
  eval_freq=96000 \
  norm_obs=true norm_act=false norm_reward=false norm_cost=false
```

This launches distributed CSAC-LB training on `RoomHeat1-v0` using 5 parallel environments. Logs and models are
stored under `saferl/exp/local/.../RoomHeat1-v0/csac_lb/...`.

To customize new building, please create a new yaml config under `saferl/examples/configs/env` referring to other room configs. Available buildings can be found under `saferl/third_party/i4b/src/gym_interface/__init__.py`. Make sure also change it both in train_env and eval_env under the yaml file and give a new corresponding env_name for saving.

### 6. Evaluate the trained Saferl policy

To evaluate a trained run (reporting the same metrics as the built-in PPO and MPC scripts):

```bash
python -m saferl.third_party.i4b.examples.eval_saferl_policy \
  --exp_dir exp/local/<date>/Benchmark/RoomHeat1-v0/csac_lb/<run_id>/Seed1_Cost[10.0] \
  --num_episodes 1 \
  --device auto
```

### RL Policy Evaluation

Evaluate a saved PPO policy:

```bash
python -m examples.eval_sb3_policy \
  --building sfh_2016_now_0_soc \
  --hp_model Heatpump_AW \
  --method 4R3C \
  --mdot_hp 0.25 \
  --internal_gain_profile data/profiles/InternalGains/ResidentialDetached.csv \
  --delta_t 900 \
  --days 30 \
  --model_path runs/ppo_roomheat/ppo_roomheat.zip \
  --num_episodes 5
```

## License

Licensed under the terms of the BSD 3-Clause License.
