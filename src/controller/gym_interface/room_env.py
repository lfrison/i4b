from typing import Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint
import model_hvac
from simulator import Model_simulator
import Utilities as util 
from model_buildings import Building
import scipy.stats as stats
from gym_interface import BUILDING_NAMES2CLASS
import pandas as pd
import os
from disturbances import load_weather, get_solar_gains, get_int_gains

OBSERVATION_SPACE_LIMIT = {
    'T_room': (5, 60),
    'T_wall': (5, 60),
    'T_hp_ret': (5, 65),
    'T_hp_sup': (5, 65),
    'T_amb': (-25, 45),
    'T_forecast' : (-25, 45),
    'Qdot_gains': (0, 8000),
    'goal_temperature': (15, 30),
}

class RoomHeatEnv(gym.Env):   
    """ Convert the I4B simulation environment to Gymnasium fasion
    """
    def __init__(self,
        hp_model,
        building,
        method,
        mdot_HP,
        internal_gain_profile,
        # weather_profile,
        # weather_forecast_profile,
        weather_forecast_steps: list = [],
        grid_profile: str = None,
        cost_dim : int = 1,
        # Simulation parameters
        timestep : int = 3600,
        days : int = None,
        random_init: bool = False,
        action_deviation_factor: float = 10,
        dev_sum_weight: float = 100,
        dev_max_weight: float = 100,
        reward_function_idx: int = 0,
        goal_based: bool = False,
        noise_level: float = 0.0,
    ):
        """This model implements a OpenAI gym wrapper for a Room temperature simulator.
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
            low = (-1), # TODO: get the minimal action/heat pump temperature
            high = (1),
            shape = (1,),
            dtype = np.float32
        )
        # TODO: Now action space normalized to (-1, 1)
        self.action_low = 20
        self.action_high = 65

        self.t = None # starting timestep corresponding to the weather data

        self.num_steps = 0 # number of steps of the current episode
        
        # Load weather data and 
        pos = self.building["position"]
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.weather_data = load_weather(pos["lat"], pos["long"], pos["altitude"], tz=pos['timezone'],
                                         repo_filepath=parent_dir)
        # self.T_amb = weather_data
        # Generate absolut heat gain profiles, based on datetime, building usage and floor area. 
        self.internal_gains_df = get_int_gains(
            time = self.weather_data.index,
            profile_path = os.path.join(parent_dir, internal_gain_profile),
            bldg_area = self.building['area_floor'])
        Qdot_sol = get_solar_gains(weather = self.weather_data, bldg_params = self.building)
        Qdot_gains = pd.DataFrame(Qdot_sol + self.internal_gains_df['Qdot_tot'] , columns = ['Qdot_gains']) # calculate total gains
        self.p = pd.concat([self.weather_data['T_amb'], Qdot_gains], axis = 1).astype(np.float32).resample(f'{timestep}s').ffill() # Disturbances

        self.action_deviation_factor = action_deviation_factor
        self.dev_sum_weight = dev_sum_weight
        self.dev_max_weight = dev_max_weight
        
        self.goal_based = goal_based
        
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
        max_t = days * 24 * int(3600 / self.timestep)
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
        
        self.reward_function_idx = reward_function_idx
        self.reset()

    def reset_env(self, building, mdot_HP, hp_model, weather_profile, internal_gain_profile, weather_forecast_profile):
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
            time = self.weather_data.index,
            profile = internal_gain_profile,
            bldg_area = self.building['area_floor'])
        Qdot_sol = get_solar_gains(weather = self.weather_data, bldg_params = self.building)
        Qdot_gains = pd.DataFrame(Qdot_sol + self.internal_gains_df['Qdot_tot'] , columns = ['Qdot_gains']) # calculate total gains
        self.p = pd.concat([self.weather_data['T_amb'], Qdot_gains], axis = 1).astype(np.float32).resample(f'{self.timestep}s').ffill() # Disturbances
        self.hp_model_name = hp_model

        self.weather_forecast_data = weather_forecast_profile
        self.reset()

    def get_obs(self):
        return self.state.copy()
    
    def get_cur_T_amb(self):
        return self.p.iloc[self.t]['T_amb']
    
    def get_cur_Qdot_gains(self):
        return self.p.iloc[self.t]['Qdot_gains']

    def get_building_info(self):
        # which building, heat pump type and mass flow rate
        return (self.hp_model_name, self.bldg_model.params['name'], self.bldg_model.mdot_hp)

    def get_cur_time(self):
        return self.p.index[self.t]

    def get_info_wt(self):
        # return T (n_day of the year), H (hour of the day), W (if it's a weekday or weekend day)
        return (self.get_cur_time(), *self.get_building_info())

    def get_cur_p(self):
        return self.p.iloc[self.t].to_dict()

    def get_cur_weather_forecast(self):
        return [self.p.iloc[self.t + i]['T_amb'] for i in self.weather_forecast_steps]
    
    def get_p_by_t(self, t):
        return self.p.iloc[t].to_dict()

    def _reward_func(self,
        action_deviation: np.ndarray,
        costs: dict
    ) -> float:
        """Customized reward function for Room Temperature Simulation

        Args:
            prev_obs (np.ndarray): previous observation
            cur_obs (np.ndarray): current observation
            u (np.ndarray): control action
            action_deviation (np.ndarry): control diviation from the last action
            Q_el (float): used HP electricity [kWh]
            deviation (tuple):
                dev_sum_neg (float): absolute room temperature comfort deviation from the lower bound (21 degrees) hourly on average [Kh]
                dev_max_neg (float): absolute maximal room temperature comfort deviation from the lower bound (21 degrees) [K]
                dev_sum_pos (float): absolute room temperature comfort deviation hourly from the high bound (26 degrees) on average [Kh]
                dev_max_pos (float): absolute maximal room temperature comfort deviation from the high bound (26 degrees) [K]
    
        Returns:
            reward (float): received reward sigmal
        """
        dev_sum, dev_max = costs['dev_neg_sum'], costs['dev_neg_max']
        Q_el = costs['E_el']

        return - Q_el

    
    def restore_action(self, a):
        return a * (self.action_high - self.action_low) / 2 + (self.action_low + self.action_high) / 2

    def normalize_action(self, a):
        return (a - (self.action_low + self.action_high) / 2) * 2 / (self.action_high - self.action_low)

    def step(self, a):
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
        # convert to kwh
        costs['E_el'] = costs['E_el'] / 1000
        self.t += 1

        self.state[:self.obs_len] = [next_state[key] for key in self.obs_keys]
        self.state[self.obs_len:self.obs_len+len(self.p_keys)] = [pk[key] for key in self.p_keys]
        self.state[self.obs_len+len(self.p_keys):self.obs_len+len(self.p_keys)+len(self.weather_forecast_steps)] = self.get_cur_weather_forecast()

        reward = self._reward_func(action_deviation, costs).item()
        self._cur_steps += 1
        if self.cost_dim == 1:
            cost = costs["dev_neg_max"].item()
        elif self.cost_dim == 2:
            # cost = [costs["dev_neg_sum"].item(), costs["dev_neg_max"].item()]
            cost = [costs["dev_neg_sum"].item(), costs["dev_neg_max"].item() > 0.01] # 0.01 is the tolerance threshold
        return (
            self.state.copy() + np.random.normal(0, self.noise_level, self.state.shape),
            reward,
            cost,
            False,
            True if self.t >= len(self.p) - 1  or self._cur_steps >= self.max_t else False,
            {
                "cost" : cost,
                "Q_el"    : costs["E_el"].item() / 1000,
                "dev_sum" : costs["dev_neg_sum"].item(),
                "dev_max" : costs["dev_neg_max"].item(),
                "t" : self.t,
                "u" : T_hp_sup_set
            },
        )

    def reset(self, seed=None, **kwargs):
        # self.seed(seed)
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
    
    def create_init_state(self, t=0, date=None):
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

