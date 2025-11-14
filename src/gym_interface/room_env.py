from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import src.models.model_hvac as model_hvac
from src.simulator import Model_simulator
from src.models.model_buildings import Building
from src.gym_interface import BUILDING_NAMES2CLASS
from src.gym_interface.constant import OBSERVATION_SPACE_LIMIT
from src.disturbances import load_weather, get_solar_gains, get_int_gains


class RoomHeatEnv(gym.Env):   
    """Gymnasium environment for room heating control using the i4b simulator.

    This environment exposes a heat-pump controlled building thermal model to
    RL agents. The action is a normalized setpoint for supply flow temperature
    in the range [-1, 1], mapped to a physical range [20, 65] degrees Celsius.

    Observations contain:
    - Building states (T_room, T_wall, T_hp_ret, T_hp_sup)
    - Current disturbances (T_amb, Qdot_gains)
    - Optional weather forecasts
    - Optional goal temperature (when goal_based=True)

    Episode termination uses standard Gymnasium semantics:
    - terminated: always False (no terminal states by default)
    - truncated: True if the configured horizon is reached

    Note: Cost metrics (comfort deviations, energy use) are reported via the
    info dictionary for logging/evaluation.
    """
    def __init__(self,
        hp_model: str,
        building: str,
        method: str,
        mdot_HP: float,
        internal_gain_profile: str,
        weather_forecast_steps: List[int] = [],
        # Simulation parameters
        timestep: int = 3600,
        days: int = None,
        random_init: bool = False,
        # Goal-based learning parameters
        goal_based: bool = False,
        goal_temp_range: Tuple[float, float] = (19.0, 28.0),
        # Reward shaping parameters
        temp_deviation_weight: float = 0.0,
        # Observation noise
        noise_level: float = 0.0,
        # Legacy parameters (kept for compatibility)
        grid_profile: str = None,
        cost_dim: int = 1,
        action_deviation_factor: float = 10,
        dev_sum_weight: float = 100,
        dev_max_weight: float = 100,
    ):
        """Initialize the RoomHeatEnv.

        Parameters
        ----------
        hp_model : str
            Name of the heat pump model class in `models.model_hvac`.
        building : str
            Key in `gym_interface.BUILDING_NAMES2CLASS` selecting building params.
        method : str
            Building model structure, e.g., '4R3C'.
        mdot_HP : float
            Mass flow rate of the heat supply system in kg/s.
        internal_gain_profile : str
            Relative path to internal gains profile CSV under the repo root.
        weather_forecast_steps : List[int]
            Steps ahead (in multiples of timestep) to append T_amb forecasts.
        timestep : int
            Sampling interval in seconds.
        days : int
            Episode length in days. If None, uses full available length.
        random_init : bool
            Whether to randomize initial state and start index.
        goal_based : bool
            If True, enables goal-based learning with randomized target temperature.
        goal_temp_range : Tuple[float, float]
            Range for goal temperature sampling (min, max) in degrees Celsius.
        temp_deviation_weight : float
            Weight for temperature deviation penalty in reward function (default 0).
        noise_level : float
            Standard deviation of Gaussian noise added to observations.
        grid_profile : str
            Optional grid price/signal profile name (unused, for future).
        cost_dim : int
            If 1, only log dev_neg_max; if 2, also log dev_neg_sum and a bool flag.
        action_deviation_factor : float
            Legacy parameter, kept for compatibility.
        dev_sum_weight : float
            Legacy parameter, kept for compatibility.
        dev_max_weight : float
            Legacy parameter, kept for compatibility.
        """
        super(RoomHeatEnv, self).__init__()

        # Core simulation parameters
        self.timestep = timestep
        self.method = method
        self.cost_dim = cost_dim
        self.noise_level = noise_level
        
        # Goal-based learning
        self.goal_based = goal_based
        self.goal_temp_range = goal_temp_range
        self.goal_temperature = 20.0  # Default, will be randomized in reset()
        
        # Reward shaping
        self.temp_deviation_weight = temp_deviation_weight
        
        # Initialize building model
        if building not in BUILDING_NAMES2CLASS.keys():
            raise ValueError(f"Building {building} not in the list of available buildings")
        
        print(f"Building {building} is selected")
        self.building = BUILDING_NAMES2CLASS[building]
        self.bldg_model = Building(
            params=self.building,
            mdot_hp=mdot_HP,
            method=self.method
        )
        
        # Initialize heat pump model
        self.hp_model = getattr(model_hvac, hp_model)(mdot_HP=mdot_HP)
        self.hp_model_name = hp_model
        
        # Initialize simulator
        self.simulator = Model_simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=self.timestep,
        )
        
        # Define action space: normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        self.action_low = 20.0  # Minimum supply temperature [°C]
        self.action_high = 65.0  # Maximum supply temperature [°C]

        # Load weather and disturbances
        self._load_disturbances(internal_gain_profile)
        
        # Define observation space
        self.obs_keys = self.bldg_model.state_keys
        self.p_keys = ["T_amb", "Qdot_gains"]
        self.weather_forecast_steps = weather_forecast_steps
        self.observation_space = self._create_observation_space()
        
        # Episode management
        self.days = days
        self.max_t = self._calculate_max_timesteps()
        self.random_init = random_init
        self.t = 0  # Current timestep in weather data
        self._cur_steps = 0  # Steps in current episode
        self.state = None
        self.prev_action = None

        self.reset()

    def _load_disturbances(self, internal_gain_profile: str):
        """Load weather data and calculate total disturbances."""
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        
        pos = self.building["position"]
        self.weather_data = load_weather(
            pos["lat"], pos["long"], pos["altitude"],
            tz=pos['timezone'],
            repo_filepath=str(repo_root)
        )
        
        # Generate internal gains
        self.internal_gains_df = get_int_gains(
            time=self.weather_data.index,
            profile_path=str(repo_root / internal_gain_profile),
            bldg_area=self.building['area_floor']
        )
        
        # Calculate total gains (solar + internal)
        Qdot_sol = get_solar_gains(weather=self.weather_data, bldg_params=self.building)
        Qdot_gains = pd.DataFrame(
            Qdot_sol + self.internal_gains_df['Qdot_tot'],
            columns=['Qdot_gains']
        )
        
        # Combine ambient temperature and gains
        self.p = pd.concat(
            [self.weather_data['T_amb'], Qdot_gains],
            axis=1
        ).astype(np.float32).resample(f'{self.timestep}s').ffill()

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space based on configuration."""
        # Building states
        obs_low = [OBSERVATION_SPACE_LIMIT[key][0] for key in self.obs_keys]
        obs_high = [OBSERVATION_SPACE_LIMIT[key][1] for key in self.obs_keys]
        
        # Current disturbances
        obs_low.extend([
            OBSERVATION_SPACE_LIMIT['T_amb'][0],
            OBSERVATION_SPACE_LIMIT['Qdot_gains'][0]
        ])
        obs_high.extend([
            OBSERVATION_SPACE_LIMIT['T_amb'][1],
            OBSERVATION_SPACE_LIMIT['Qdot_gains'][1]
        ])
        
        # Weather forecast
        if len(self.weather_forecast_steps) > 0:
            print(f"Enabling Weather Forecast: {len(self.weather_forecast_steps)} steps")
            obs_low.extend([OBSERVATION_SPACE_LIMIT['T_amb'][0]] * len(self.weather_forecast_steps))
            obs_high.extend([OBSERVATION_SPACE_LIMIT['T_amb'][1]] * len(self.weather_forecast_steps))
        else:
            print("Disabling Weather Forecast")
        
        # Goal temperature (if goal-based)
        if self.goal_based:
            print("Enabling Goal-Based Mode")
            obs_low.append(self.goal_temp_range[0])
            obs_high.append(self.goal_temp_range[1])
        
        return spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )

    def _calculate_max_timesteps(self) -> int:
        """Calculate maximum timesteps per episode."""
        if self.days is None:
            return len(self.p) - 1
        
        max_t = self.days * 24 * int(3600 / self.timestep)
        if max_t >= len(self.p):
            raise ValueError(
                f"Maximum timesteps ({max_t}) exceeds available data length ({len(self.p)})"
            )
        return max_t

    def _sample_goal_temperature(self) -> float:
        """Sample a goal temperature with 1 decimal precision."""
        # Generate temperature between min and max with 0.1 degree precision
        min_temp, max_temp = self.goal_temp_range
        n_steps = int((max_temp - min_temp) * 10) + 1
        temp = np.random.choice(np.linspace(min_temp, max_temp, n_steps))
        return round(float(temp), 1)

    def _build_observation(self, state_dict: Dict) -> np.ndarray:
        """Build observation vector from state dictionary."""
        obs = []
        
        # Building states
        obs.extend([state_dict[key] for key in self.obs_keys])
        
        # Current disturbances
        pk = self.get_cur_p()
        obs.extend([pk[key] for key in self.p_keys])
        
        # Weather forecast
        if len(self.weather_forecast_steps) > 0:
            obs.extend(self.get_cur_weather_forecast())
        
        # Goal temperature
        if self.goal_based:
            obs.append(self.goal_temperature)
        
        return np.array(obs, dtype=np.float32)

    def restore_action(self, a: np.ndarray) -> float:
        """Map normalized action in [-1, 1] to physical setpoint in [20, 65]°C."""
        return float(a * (self.action_high - self.action_low) / 2 + 
                     (self.action_low + self.action_high) / 2)

    def normalize_action(self, a: np.ndarray) -> np.ndarray:
        """Map physical setpoint to normalized action in [-1, 1]."""
        return (a - (self.action_low + self.action_high) / 2) * 2 / (self.action_high - self.action_low)

    def _reward_func(self, costs: Dict, T_room: float) -> float:
        """Compute scalar reward.

        Returns negative electricity consumption plus optional temperature deviation penalty.
        
        Parameters
        ----------
        costs : Dict
            Cost dictionary containing 'E_el' (kWh) and deviation metrics.
        T_room : float
            Current room temperature in degrees Celsius.
            
        Returns
        -------
        float
            Reward value (to be maximized).
        """
        # Energy cost (negative to minimize)
        energy_penalty = -float(costs['E_el'])
        
        # Temperature deviation penalty (if weighted)
        temp_deviation_penalty = 0.0
        if self.temp_deviation_weight > 0:
            temp_deviation = abs(T_room - self.goal_temperature)
            temp_deviation_penalty = -self.temp_deviation_weight * temp_deviation
        
        return energy_penalty + temp_deviation_penalty

    def step(self, a: np.ndarray):
        """Advance the simulation by one step.

        Returns (obs, reward, terminated, truncated, info) following Gymnasium API.
        Cost/comfort metrics are included in info.
        """
        # Get current state
        state_dict = {key: value for key, value in zip(self.obs_keys, self.state[:len(self.obs_keys)])}
        pk = self.get_cur_p()
        
        # Convert normalized action to physical setpoint
        T_hp_sup_set = self.restore_action(a)
        
        # Apply heating logic
        if pk['T_amb'] < self.bldg_model.params['T_amb_lim']:
            # Normal heating: apply offset and ensure supply > return
            T_hp_sup_set = max(
                T_hp_sup_set + self.bldg_model.params['T_offset'],
                state_dict['T_hp_ret']
            )
        else:
            # Too warm outside: no heating
            T_hp_sup_set = state_dict['T_hp_ret']

        # Check heat pump constraints
        T_hp_sup_set = self.hp_model.check_hp(T_hp_sup_set, state_dict['T_hp_ret'])
        self.prev_action = T_hp_sup_set

        # Simulate one step
        res = self.simulator.get_next_state(state_dict, T_hp_sup_set, pk)
        next_state, costs = res['state'], res['cost']
        
        # Convert electricity use to kWh
        costs['E_el'] = costs['E_el'] / 1000
        
        # Update time and state
        self.t += 1
        self._cur_steps += 1
        self.state = self._build_observation(next_state)
        
        # Calculate reward
        reward = self._reward_func(costs, next_state['T_room'])
        
        # Check termination
        truncated = (self.t >= len(self.p) - 1 or self._cur_steps >= self.max_t)
        
        # Build info dictionary
        info = {
            "Q_el_kWh": float(costs["E_el"]),
            "dev_sum": float(costs["dev_neg_sum"]),
            "dev_max": float(costs["dev_neg_max"]),
            "t": int(self.t),
            "u": float(T_hp_sup_set),
            "T_room": float(next_state['T_room']),
        }
        
        if self.goal_based:
            info["goal_temperature"] = float(self.goal_temperature)
            info["temp_deviation"] = float(abs(next_state['T_room'] - self.goal_temperature))
        
        if self.cost_dim == 2:
            info["dev_sum_hourly"] = float(costs["dev_neg_sum"])
            info["comfort_violated"] = bool(costs["dev_neg_max"] > 0.01)
        
        # Add noise to observation
        obs = self.state.copy()
        if self.noise_level > 0:
            obs += np.random.normal(0, self.noise_level, obs.shape)
        
        return obs, float(reward), False, truncated, info

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial observation and info."""
        super().reset(seed=seed)
        
        # Sample new goal temperature if goal-based
        if self.goal_based:
            self.goal_temperature = self._sample_goal_temperature()
        
        # Determine starting timestep
        if self.random_init:
            max_offset = max(self.weather_forecast_steps) if self.weather_forecast_steps else 0
            max_start = self.p.shape[0] - self.max_t - 1 - max_offset
            self.t = np.random.randint(0, max_start)
        else:
            self.t = 0
        
        # Initialize state
        self.state = self._create_initial_state()
        self.prev_action = None
        self._cur_steps = 0
        
        return self.state.copy(), {}
    
    def _create_initial_state(self) -> np.ndarray:
        """Create initial state observation."""
        if self.random_init:
            # Random initialization within observation bounds
            obs_low = [OBSERVATION_SPACE_LIMIT[key][0] for key in self.obs_keys]
            obs_high = [OBSERVATION_SPACE_LIMIT[key][1] for key in self.obs_keys]
            state_dict = {
                key: np.random.uniform(low, high)
                for key, low, high in zip(self.obs_keys, obs_low, obs_high)
            }
        else:
            # Initialize all temperatures to goal (or 20°C if not goal-based)
            init_temp = self.goal_temperature if self.goal_based else 20.0
            state_dict = {key: init_temp for key in self.obs_keys}
        
        return self._build_observation(state_dict)

    # Utility methods
    def get_obs(self) -> np.ndarray:
        """Return a copy of the current observation vector."""
        return self.state.copy()
    
    def get_cur_T_amb(self) -> float:
        """Get current ambient temperature in degrees Celsius."""
        return float(self.p.iloc[self.t]['T_amb'])
    
    def get_cur_Qdot_gains(self) -> float:
        """Get current total heat gains in W."""
        return float(self.p.iloc[self.t]['Qdot_gains'])

    def get_cur_p(self) -> Dict:
        """Return current disturbances as a dict (T_amb, Qdot_gains)."""
        return self.p.iloc[self.t].to_dict()

    def get_cur_weather_forecast(self) -> List[float]:
        """Return list of ambient temperature forecasts for configured steps."""
        return [float(self.p.iloc[self.t + i]['T_amb']) for i in self.weather_forecast_steps]
    
    def get_cur_time(self):
        """Return current pandas timestamp from disturbances index."""
        return self.p.index[self.t]

    def get_building_info(self) -> Tuple[str, str, float]:
        """Return (hp_model_name, building_name, mass_flow_rate)."""
        return (self.hp_model_name, self.bldg_model.params['name'], self.bldg_model.mdot_hp)

    def get_info_wt(self):
        """Return tuple of (current_time, hp_model_name, building_name, mass_flow_rate)."""
        return (self.get_cur_time(), *self.get_building_info())

    def get_p_by_t(self, t: int) -> Dict:
        """Return disturbances at index t as a dict."""
        return self.p.iloc[t].to_dict()

    def reset_env(self, building: Dict, mdot_HP: float, hp_model: str, 
                  weather_profile: pd.DataFrame, internal_gain_profile: str, 
                  weather_forecast_profile: pd.DataFrame = None):
        """Reset environment with new building and weather configuration.
        
        This method allows dynamic reconfiguration of the environment.
        Kept for backward compatibility.
        """
        self.building = building
        self.bldg_model = Building(
            params=self.building,
            mdot_hp=mdot_HP,
            method=self.method
        )
        self.hp_model = getattr(model_hvac, hp_model)(mdot_HP=mdot_HP)
        self.hp_model_name = hp_model
        self.simulator = Model_simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=self.timestep,
        )
        
        # Reload disturbances
        self.weather_data = weather_profile
        self.internal_gains_df = get_int_gains(
            time=self.weather_data.index,
            profile_path=internal_gain_profile,
            bldg_area=self.building['area_floor']
        )
        Qdot_sol = get_solar_gains(weather=self.weather_data, bldg_params=self.building)
        Qdot_gains = pd.DataFrame(
            Qdot_sol + self.internal_gains_df['Qdot_tot'],
            columns=['Qdot_gains']
        )
        self.p = pd.concat(
            [self.weather_data['T_amb'], Qdot_gains],
            axis=1
        ).astype(np.float32).resample(f'{self.timestep}s').ffill()
        
        self.reset()
            
    def render(self, mode='human'):
        raise NotImplementedError
