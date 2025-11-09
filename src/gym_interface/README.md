# Gym Interface - i4b Building Control Environment

## Overview

The `gym_interface` provides a Gymnasium-compatible environment for training reinforcement learning agents to control building heating systems. The environment simulates a heat pump controlled building thermal model with realistic weather data and disturbances.

## Recent Updates (Refactored)

The codebase has been recently refactored with the following improvements:

### 1. Code Simplification
- Consolidated observation space creation into a single method
- Improved code organization with helper methods
- Better documentation and clearer parameter descriptions
- Removed redundant code and improved maintainability

### 2. Goal-Based Learning Support
- New `goal_based` parameter to enable goal-conditioned RL
- Random goal temperature sampling for each episode
- Goal temperature range: 19-28°C with 0.1°C precision
- Goal temperature included in observation space when enabled
- Useful for training agents that can adapt to different comfort preferences

### 3. Temperature Deviation Reward
- New `temp_deviation_weight` parameter for reward shaping
- Allows balancing energy efficiency vs. thermal comfort
- Default value of 0 maintains backward compatibility (energy-only optimization)
- Positive values penalize deviation from goal temperature

## Quick Start

### Basic Usage

```python
from gym_interface import make_room_heat_env

# Create a basic environment
env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    timestep=3600,
    days=30,
)

# Standard gym interface
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Goal-Based Learning

```python
# Create an environment with goal-based learning
env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    timestep=3600,
    days=30,
    goal_based=True,  # Enable goal-based mode
    goal_temp_range=(19.0, 28.0),  # Goal temperature range
    temp_deviation_weight=1.0,  # Penalize temp deviation in reward
)

obs, info = env.reset()
print(f"Goal temperature for this episode: {env.goal_temperature}°C")

# The goal temperature is included in the observation
# Last element of obs is the goal temperature
print(f"Goal from observation: {obs[-1]}°C")
```

### With Weather Forecast

```python
# Include weather forecast in observations
env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    weather_forecast_steps=[1, 2, 3],  # 1, 2, 3 hours ahead
    timestep=3600,
    days=30,
)
```

### Training with Stable-Baselines3

```bash
# Basic training
python examples/train_sb3_ppo.py \
    --building sfh_2016_now_0_soc \
    --days 30 \
    --total_timesteps 200000

# Goal-based training with temperature deviation penalty
python examples/train_sb3_ppo.py \
    --building sfh_2016_now_0_soc \
    --days 30 \
    --goal_based \
    --goal_temp_min 19.0 \
    --goal_temp_max 28.0 \
    --temp_deviation_weight 1.0 \
    --total_timesteps 200000
```

## Parameters

### Core Parameters

- **hp_model** (str): Heat pump model name (e.g., 'Heatpump_AW')
- **building** (str): Building configuration name
- **method** (str): Building model method (e.g., '4R3C')
- **mdot_HP** (float): Mass flow rate in kg/s
- **internal_gain_profile** (str): Path to internal gains profile CSV

### Simulation Parameters

- **timestep** (int): Simulation timestep in seconds (default: 3600)
- **days** (int): Episode length in days (default: None = full data)
- **random_init** (bool): Randomize initial state and start time (default: False)

### Goal-Based Learning Parameters

- **goal_based** (bool): Enable goal-based learning mode (default: False)
- **goal_temp_range** (tuple): Min and max goal temperature in °C (default: (19.0, 28.0))
- **temp_deviation_weight** (float): Weight for temperature deviation penalty (default: 0.0)

### Other Parameters

- **weather_forecast_steps** (list): Forecast steps to include in observations (default: [])
- **noise_level** (float): Observation noise std dev (default: 0.0)
- **cost_dim** (int): Cost dimension (1 or 2) for info dict (default: 1)

## Observation Space

The observation vector contains (in order):

1. **Building states** (4 values):
   - `T_room`: Room temperature [°C]
   - `T_wall`: Wall temperature [°C]
   - `T_hp_ret`: Heat pump return temperature [°C]
   - `T_hp_sup`: Heat pump supply temperature [°C]

2. **Current disturbances** (2 values):
   - `T_amb`: Ambient temperature [°C]
   - `Qdot_gains`: Total heat gains [W]

3. **Weather forecast** (optional, N values):
   - Future ambient temperatures if `weather_forecast_steps` is specified

4. **Goal temperature** (optional, 1 value):
   - Target room temperature [°C] if `goal_based=True`

### Observation Space Shapes

- Basic (no forecast, no goal): (6,)
- With 3-step forecast: (9,)
- Goal-based (no forecast): (7,)
- Goal-based with 3-step forecast: (10,)

## Action Space

- **Type**: Continuous, Box(-1, 1)
- **Physical mapping**: Normalized action [-1, 1] → Supply temperature [20, 65]°C
- **Description**: Target supply flow temperature for the heat pump

## Reward Function

The reward function varies based on configuration:

### Basic Mode (energy-only)
```
reward = -E_el_kWh
```
Minimizes electricity consumption.

### Goal-Based Mode (with temperature deviation)
```
reward = -E_el_kWh - temp_deviation_weight * |T_room - T_goal|
```
Balances energy efficiency with thermal comfort.

### Setting the Deviation Weight

- **temp_deviation_weight = 0**: Pure energy optimization (default)
- **temp_deviation_weight > 0**: Mixed optimization (energy + comfort)
- **temp_deviation_weight >> E_el**: Comfort prioritized over energy

Typical values: 0.1 - 2.0 for balanced optimization.

## Info Dictionary

Each step returns an info dictionary with:

- `Q_el_kWh`: Electricity consumption [kWh]
- `dev_sum`: Sum of negative temperature deviations
- `dev_max`: Maximum negative temperature deviation
- `t`: Current timestep index
- `u`: Applied control action (supply temperature) [°C]
- `T_room`: Current room temperature [°C]

Additional fields when `goal_based=True`:
- `goal_temperature`: Goal temperature for this episode [°C]
- `temp_deviation`: Absolute temperature deviation from goal [°C]

## Example Use Cases

### 1. Standard Energy Optimization
Train an agent to minimize energy consumption while maintaining comfort:
```python
env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    days=30,
)
```

### 2. Adaptive Comfort Control
Train an agent that can adapt to different user preferences:
```python
env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    days=30,
    goal_based=True,
    goal_temp_range=(19.0, 25.0),  # Range of user preferences
    temp_deviation_weight=1.0,
)
```

### 3. Predictive Control with Forecast
Train an agent that uses weather forecasts for better planning:
```python
env = make_room_heat_env(
    building='sfh_2016_now_0_soc',
    hp_model='Heatpump_AW',
    method='4R3C',
    mdot_HP=0.25,
    internal_gain_profile='data/profiles/InternalGains/ResidentialDetached.csv',
    weather_forecast_steps=[1, 2, 3, 6, 12],  # 1-12 hours ahead
    days=30,
    goal_based=True,
    temp_deviation_weight=0.5,
)
```

## Implementation Details

### Goal Temperature Sampling

When `goal_based=True`, a new goal temperature is sampled at each episode reset:
- Range: Specified by `goal_temp_range` (default: 19.0-28.0°C)
- Precision: 0.1°C (e.g., 19.0, 19.1, 19.2, ..., 27.9, 28.0)
- Distribution: Uniform random sampling

### Reward Shaping

The temperature deviation weight allows tuning the trade-off:
- Higher weight → More emphasis on comfort (maintain goal temperature)
- Lower weight → More emphasis on energy efficiency
- Weight of 0 → Pure energy optimization (backward compatible)

## Migration Guide

If you have existing code using the old environment:

### What Stayed the Same
- Basic API (reset, step) unchanged
- Observation and action spaces unchanged (when goal_based=False)
- Default behavior is backward compatible

### What Changed
- Some internal methods reorganized (should not affect external usage)
- New optional parameters added (all have backward-compatible defaults)
- Legacy parameters (action_deviation_factor, dev_sum_weight, dev_max_weight) kept for compatibility but not actively used

### To Enable New Features
Simply add the new parameters to your environment creation:
```python
# Old code (still works)
env = make_room_heat_env(building='...', hp_model='...', ...)

# New code with goal-based learning
env = make_room_heat_env(
    building='...', 
    hp_model='...', 
    goal_based=True,
    temp_deviation_weight=1.0,
    ...
)
```

## Files

- `room_env.py`: Main environment implementation (RoomHeatEnv class)
- `__init__.py`: Factory functions and building configurations
- `constant.py`: Observation space limits and constants
- `README.md`: This documentation

## Citation

If you use this environment in your research, please cite the i4b project.

