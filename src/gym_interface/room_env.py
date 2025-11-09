from typing import Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import models.model_hvac as model_hvac
from simulator import Model_simulator
from models.model_buildings import Building
from gym_interface import BUILDING_NAMES2CLASS
import pandas as pd
from gym_interface.constant import OBSERVATION_SPACE_LIMIT
import os
from disturbances import load_weather, get_solar_gains, get_int_gains


class RoomHeatEnv(gym.Env):   
    """Gymnasium environment for room heating control using the i4b simulator.

    This environment exposes a heat-pump controlled building thermal model to
    RL agents. The action is a normalized setpoint for supply flow temperature
    in the range [-1, 1], mapped to a physical range [action_low=20,
    action_high=65] in degrees Celsius.
    Observations contain the building states and current disturbances 
    (ambient temperature and total gains), with optional weather
    forecasts appended.

    Episode termination uses standard Gymnasium semantics:
    - terminated: always False (no terminal states by default)
    - truncated: True if the configured horizon is reached

    Note: Cost metrics (comfort deviations, energy use) are reported via the
    info dictionary for logging/evaluation, not as a separate return item.
    """
    def __init__(self,
        hp_model: str,
        building: str,
        method: str,
        mdot_HP: float,
        internal_gain_profile: str,
        weather_forecast_steps: List[int] = [],
        grid_profile: str = None,
        cost_dim: int = 1,
        # Simulation parameters
        timestep: int = 3600,
        days: int = None,
        random_init: bool = False,
        action_deviation_factor: float = 10,
        dev_sum_weight: float = 100,
        dev_max_weight: float = 100,
        reward_function_idx: int = 0,
        goal_based: bool = False,
        noise_level: float = 0.0,
        # TODO: allow flexible goal
        goal_constraint_limit: float = None,
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
        grid_profile : str
            Optional grid price/signal profile name (unused for now).
        cost_dim : int
            If 1, only log dev_neg_max; if 2, also log dev_neg_sum and a bool flag.
        timestep : int
            Sampling interval in seconds.
        days : int
            Episode length in days. If None, uses full available length.
        random_init : bool
            Whether to randomize initial state and start index.
        action_deviation_factor, dev_sum_weight, dev_max_weight : float
            Weights for custom reward shaping (reserved).
        reward_function_idx : int
            Selector for reward function variants (reserved).
        goal_based : bool
            If True, enables goal-constraint logic (reserved).
        noise_level : float
            Standard deviation of Gaussian noise added to observations.
        goal_constraint_limit : float
            Non-negative constraint limit if goal_based is True.
        """
        super(RoomHeatEnv, self).__init__()

        self.timestep = timestep # sampling interval in s
        self.method = method
        self.cost_dim = cost_dim
        if building not in BUILDING_NAMES2CLASS.keys():
            raise ValueError(f"Building {building} not in the list of available buildings")
        else:
            print(f"Building {building} is selected")
            self.building = BUILDING_NAMES2CLASS[building]
        self.bldg_model = Building(params = self.building, # More example buildings can be found in data/buildings/.
                                  mdot_hp = mdot_HP,       # Massflow of the heat supply system. [kg/s]
                                  method = self.method) 
        self.hp_model = getattr(model_hvac, hp_model)(mdot_HP = mdot_HP)
        self.hp_model_name = hp_model
        self.noise_level = noise_level
        self.simulator = Model_simulator(
            hp_model = self.hp_model,
            bldg_model = self.bldg_model,
            timestep = self.timestep,
        )
        
        self.action_space = spaces.Box(
            low = (-1),
            high = (1),
            shape = (1,),
            dtype = np.float32
        )
        # Action space normalized to (-1, 1) mapped to [action_low, action_high]
        self.action_low = 20
        self.action_high = 65

        self.t = None # starting timestep corresponding to the weather data

        self.num_steps = 0 # number of steps of the current episode
        
        # Load weather data and 
        pos = self.building["position"]
        # Get repo root: go up 2 parent levels from src/gym_interface/room_env.py
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        self.weather_data = load_weather(pos["lat"], pos["long"], pos["altitude"], tz=pos['timezone'],
                                         repo_filepath=str(repo_root))
        # self.T_amb = weather_data
        # Generate absolut heat gain profiles, based on datetime, building usage and floor area. 
        self.internal_gains_df = get_int_gains(
            time=self.weather_data.index,
            profile_path=str(repo_root / internal_gain_profile),
            bldg_area=self.building['area_floor'])
        Qdot_sol = get_solar_gains(weather = self.weather_data, bldg_params = self.building)
        Qdot_gains = pd.DataFrame(Qdot_sol + self.internal_gains_df['Qdot_tot'] , columns = ['Qdot_gains']) # calculate total gains
        self.p = pd.concat([self.weather_data['T_amb'], Qdot_gains], axis = 1).astype(np.float32).resample(f'{timestep}s').ffill() # Disturbances

        self.action_deviation_factor = action_deviation_factor
        self.dev_sum_weight = dev_sum_weight
        self.dev_max_weight = dev_max_weight
        
        self.goal_based = goal_based
        self.goal_constraint_limit = goal_constraint_limit
        
        self.obs_keys = self.bldg_model.state_keys
        self.p_keys = ["T_amb", "Qdot_gains"]
        self.obs_len = len(self.obs_keys)

        # TODO: so far still using the real weather data
        self.weather_forecast_data = self.weather_data

        self.weather_forecast_steps = weather_forecast_steps
        if len(self.weather_forecast_steps) == 0:
            print("Disabling Weather Forecast")
            self.obs_lim_low = [OBSERVATION_SPACE_LIMIT[key][0] for key in self.obs_keys]
            self.obs_lim_high = [OBSERVATION_SPACE_LIMIT[key][1] for key in self.obs_keys]
            
            self.observation_space = spaces.Box(
                low = np.array(self.obs_lim_low + [OBSERVATION_SPACE_LIMIT['T_amb'][0]] + [OBSERVATION_SPACE_LIMIT['Qdot_gains'][0]]),
                high = np.array(self.obs_lim_high + [OBSERVATION_SPACE_LIMIT['T_amb'][1]] + [OBSERVATION_SPACE_LIMIT['Qdot_gains'][1]]),
                shape = (len(self.obs_lim_low) + 2,),
                dtype = np.float32
            )
        else:
            print("Enabling Weather Forecast")
            self.obs_lim_low = [OBSERVATION_SPACE_LIMIT[key][0] for key in self.obs_keys]
            self.obs_lim_high = [OBSERVATION_SPACE_LIMIT[key][1] for key in self.obs_keys]
            
            self.observation_space = spaces.Box(
                low = np.array(self.obs_lim_low + [OBSERVATION_SPACE_LIMIT['T_amb'][0]] + \
                    [OBSERVATION_SPACE_LIMIT['Qdot_gains'][0]] + [OBSERVATION_SPACE_LIMIT['T_amb'][0]] * len(self.weather_forecast_steps)),
                high = np.array(self.obs_lim_high + [OBSERVATION_SPACE_LIMIT['T_amb'][1]] + \
                    [OBSERVATION_SPACE_LIMIT['Qdot_gains'][1]] + [OBSERVATION_SPACE_LIMIT['T_amb'][1]] * len(self.weather_forecast_steps)),
                shape = (len(self.obs_lim_low) + 2 + len(self.weather_forecast_steps),),
                dtype = np.float32
            )
        self.days = days
        self._cur_steps = 0
        max_t = None if days is None else days * 24 * int(3600 / self.timestep)
        if max_t is not None:
            assert max_t < len(self.p), \
                "Maximaum timesteps (%d) is larger than the length (%d) of the ambient temperature" \
                    %(max_t, len(self.p))
            self.max_t = max_t
        else:
            self.max_t = len(self.p) - 1
        
        self.dev_max = None
        self.cur_action = None
        self.prev_action = None
        self.random_init = random_init

        if goal_constraint_limit is not None and goal_based:
            assert goal_constraint_limit >= 0, "Goal constraint limit must be positive"
        
        self.reward_function_idx = reward_function_idx
        self.reset()

    def reset_env(self, building: Dict, mdot_HP: float, hp_model: str, weather_profile: pd.DataFrame, internal_gain_profile: str, weather_forecast_profile: pd.DataFrame):
        self.building = building
        self.bldg_model = Building(params = self.building, # More example buildings can be found in data/buildings/.
                                  mdot_hp = mdot_HP,       # Massflow of the heat supply system. [kg/s]
                                  method = self.method) 
        self.hp_model = getattr(model_hvac, hp_model)(mdot_HP = mdot_HP)
        self.hp_model_name = hp_model
        self.simulator = Model_simulator(
            hp_model = self.hp_model,
            bldg_model = self.bldg_model,
            timestep = self.timestep,
        )
        self.weather_data = weather_profile
        # Generate absolut heat gain profiles, based on datetime, building usage and floor area. 
        self.internal_gains_df = get_int_gains(
            time=self.weather_data.index,
            profile_path=internal_gain_profile,
            bldg_area=self.building['area_floor'])
        Qdot_sol = get_solar_gains(weather = self.weather_data, bldg_params = self.building)
        Qdot_gains = pd.DataFrame(Qdot_sol + self.internal_gains_df['Qdot_tot'] , columns = ['Qdot_gains']) # calculate total gains
        self.p = pd.concat([self.weather_data['T_amb'], Qdot_gains], axis = 1).astype(np.float32).resample(f'{self.timestep}s').ffill() # Disturbances
        self.hp_model_name = hp_model

        self.weather_forecast_data = weather_forecast_profile
        self.reset()

    def get_obs(self) -> np.ndarray:
        """Return a copy of the current observation vector."""
        return self.state.copy()
    
    def get_cur_T_amb(self) -> float:
        """Get current ambient temperature in degC."""
        return float(self.p.iloc[self.t]['T_amb'])
    
    def get_cur_Qdot_gains(self) -> float:
        """Get current total heat gains in W."""
        return float(self.p.iloc[self.t]['Qdot_gains'])

    def get_building_info(self) -> Tuple[str, str, float]:
        """Return (hp_model_name, building_name, mass_flow_rate)."""
        return (self.hp_model_name, self.bldg_model.params['name'], self.bldg_model.mdot_hp)

    def get_cur_time(self):
        """Return current pandas timestamp from disturbances index."""
        return self.p.index[self.t]

    def get_info_wt(self):
        """Return tuple of (current_time, hp_model_name, building_name, mass_flow_rate)."""
        return (self.get_cur_time(), *self.get_building_info())

    def get_cur_p(self) -> Dict:
        """Return current disturbances as a dict (T_amb, Qdot_gains)."""
        return self.p.iloc[self.t].to_dict()

    def get_cur_weather_forecast(self) -> List[float]:
        """Return list of ambient temperature forecasts for configured steps."""
        return [float(self.p.iloc[self.t + i]['T_amb']) for i in self.weather_forecast_steps]
    
    def get_p_by_t(self, t: int) -> Dict:
        """Return disturbances at index t as a dict."""
        return self.p.iloc[t].to_dict()

    def _reward_func(self,
        action_deviation: np.ndarray,
        costs: Dict
    ) -> float:
        """Compute scalar reward.

        Currently returns negative electricity consumption (kWh) to minimize energy use.
        """
        Q_el_kWh = costs['E_el']
        return -float(Q_el_kWh)

    
    def restore_action(self, a: np.ndarray) -> np.ndarray:
        """Map normalized action in [-1, 1] to physical setpoint in [low, high]."""
        return a * (self.action_high - self.action_low) / 2 + (self.action_low + self.action_high) / 2

    def normalize_action(self, a: np.ndarray) -> np.ndarray:
        """Map physical setpoint to normalized action in [-1, 1]."""
        return (a - (self.action_low + self.action_high) / 2) * 2 / (self.action_high - self.action_low)

    def step(self, a: np.ndarray):
        """Advance the simulation by one step.

        Returns (obs, reward, terminated, truncated, info) following Gymnasium API.
        Cost/comfort metrics are included in info.
        """
        prev_obs = self.get_obs()
        pk = self.get_cur_p()
        
        x_init = {key : value for key, value in zip(self.obs_keys, prev_obs)}
        self.cur_action = a
        a = self.restore_action(a)
        
        # Only if upper heating limit temp. is not exceeded by ambient temp.
        if pk['T_amb'] < self.bldg_model.params['T_amb_lim']:
            # Set control variable supply flow temperature according to heating curve
            T_hp_sup_set = max(a + self.bldg_model.params['T_offset'], 
                                            x_init['T_hp_ret'])
        else:
            # Set supply flow temperature to return flow temperature -> no heating
            T_hp_sup_set = x_init['T_hp_ret']

        T_hp_sup_set = self.hp_model.check_hp(
            T_hp_sup_set,
            x_init['T_hp_ret'])

        if self.prev_action is None: 
            self.prev_action = T_hp_sup_set.copy()
        action_deviation = T_hp_sup_set - self.prev_action
        self.prev_action = T_hp_sup_set.copy()
        res = self.simulator.get_next_state(x_init, T_hp_sup_set, pk)
        next_state, costs = res['state'], res['cost']
        # convert electricity use to kWh
        costs['E_el'] = costs['E_el'] / 1000
        self.t += 1

        self.state[:self.obs_len] = [next_state[key] for key in self.obs_keys]
        self.state[self.obs_len:self.obs_len+len(self.p_keys)] = [pk[key] for key in self.p_keys]
        self.state[self.obs_len+len(self.p_keys):self.obs_len+len(self.p_keys)+len(self.weather_forecast_steps)] = self.get_cur_weather_forecast()

        reward = self._reward_func(action_deviation, costs)
        self._cur_steps += 1
        truncated = True if self.t >= len(self.p) - 1 or self._cur_steps >= self.max_t else False
        info = {
            "Q_el_kWh": float(costs["E_el"]),
            "dev_sum": float(costs["dev_neg_sum"]),
            "dev_max": float(costs["dev_neg_max"]),
            "t": int(self.t),
            "u": float(T_hp_sup_set),
        }
        if self.cost_dim == 2:
            info["dev_sum_hourly"] = float(costs["dev_neg_sum"])  # alias for clarity
            info["comfort_violated"] = bool(costs["dev_neg_max"] > 0.01)

        obs = self.state.copy() + np.random.normal(0, self.noise_level, self.state.shape)
        return obs, float(reward), False, truncated, info

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial observation and info."""
        super().reset(seed=seed)
        if self.random_init:
            if len(self.weather_forecast_steps) > 0:
                self.t = np.random.randint(0, self.p.shape[0] - self.max_t - 1 - max(self.weather_forecast_steps))
            else:
                self.t = np.random.randint(0, self.p.shape[0] - self.max_t - 1)
        else:
            self.t = 0
        self.state = self.create_init_state()
        self.dev_max = None
        self.prev_action = None
        self._cur_steps = 0
        return self.state.copy(), {}
    
    def create_init_state(self, t: int = 0, date=None) -> np.ndarray:
        if self.random_init:
            init_state = np.random.uniform(self.obs_lim_low, self.obs_lim_high).astype(np.float32)
        else:
            init_state = np.array([20] * self.obs_len, dtype=np.float32)
        
        p_init = self.p.iloc[t].values
        weather_forecast = [self.p.iloc[t + i]['T_amb'] for i in self.weather_forecast_steps]
        init_state = np.concatenate([init_state, p_init, weather_forecast])
        return init_state

            
    def render(self, mode='human'):
        raise NotImplementedError

