# i4b - Intelligence for Buildings 

## Advanced Building Heat Pump Control Testing Framework

This Python project facilitates the quick generation of reduced order building models. It features a simulator class providing a high-level interface for one-step and multi-step simulations. These simulations return the next state(s) of the building (temperatures), indicators for comfort levels, and energy demand. This interface can be used to evaluate and test different control strategies. Interface to RL, MPC, and reference heat curve controller is provided. Implementation for MPC and reference heat curve is given. The project includes simple heat pump models based on performance curves for heating systems, and disturbance profiles for ambient temperature, internal heat gains by occupancy, and solar heat gains.

![I4C_Grafik](https://github.com/lfrison/i4b/assets/104891971/65cce2cf-8801-45ba-811d-a965a0115c08)

## Table of Contents

1. [Install Dependencies](#install-dependencies)
2. [Building Model](#building-models)
3. [Disturbances](#disturbances)
4. [Controller](#controller)
5. [Gymnasium Interface (RL)](#gymnasium-interface-rl)
6. [Training with Stable-Baselines3](#training-with-stable-baselines3)
7. [Evaluation](#evaluation)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

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

## Gymnasium Interface (RL)

The module `src/gym_interface/room_env.py` exposes the building thermal simulation as a Gymnasium environment `RoomHeatEnv`. A convenience factory and registration helpers are provided in `src/gym_interface/__init__.py`.

- **Action space**: `Box([-1], [1])` (normalized). Mapped to supply flow temperature setpoint in °C within `[action_low, action_high]`.
- **Observation**: Building state vector + current `T_amb`, `Qdot_gains` and optional future `T_amb` forecasts.
- **API**: Returns `(obs, reward, terminated, truncated, info)` following Gymnasium.

Quickstart (Python):

```python
from gym_interface import make_room_heat_env

env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='HPbasic',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    weather_forecast_steps=[],  # e.g., [1,2,3] to append future T_amb steps
    timestep=3600,
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
python examples/train_sb3_ppo.py \
  --building sfh_2016_now_0_soc \
  --hp_model HPbasic \
  --method 4R3C \
  --mdot_hp 0.25 \
  --internal_gain_profile data/profiles/InternalGains/ResidentialDetached.csv \
  --timestep 3600 \
  --days 30 \
  --total_timesteps 200000 \
  --logdir runs/ppo_roomheat
```

Notes:
- Install RL extras first: `pip install -r requirements-rl.txt`.
- Adjust `--forecast` to include simple ambient temperature forecasts.

## Evaluation

Evaluate a saved PPO policy:

```bash
python examples/eval_sb3_policy.py \
  --building sfh_2016_now_0_soc \
  --hp_model HPbasic \
  --method 4R3C \
  --mdot_hp 0.25 \
  --internal_gain_profile data/profiles/InternalGains/ResidentialDetached.csv \
  --timestep 3600 \
  --days 30 \
  --model_path runs/ppo_roomheat/ppo_roomheat.zip
```

The script reports average return and total electricity use (kWh) across episodes.

## License

Licensed under the terms of the BSD 3-Clause License.
