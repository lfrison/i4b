from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import src.models.model_hvac as model_hvac
from src.simulator import Model_simulator
from src.models.model_buildings import Building
from src.gym_interface import BUILDING_NAMES2CLASS
from src.gym_interface.constant import OBSERVATION_SPACE_LIMIT
from src.disturbances import load_weather, load_weather_for_city, get_solar_gains, get_int_gains
from src.core.params import compute_derived_params
from src.core.sim import JaxSimulator, LINEAR_METHODS
from src.core.hvac import hp_params
from src.randomization import EventManager, EventSpec


class RoomHeatEnv(gym.Env):   
    """Gymnasium environment for room heating control using the i4b simulator.

    This environment exposes a heat-pump controlled building thermal model to
    RL agents. The action is a normalized setpoint for supply flow temperature
    in the range [-1, 1], mapped to a physical range [20, 65] degrees Celsius.

    Observations contain:
    - Building thermal states (varies by method, e.g. T_room, T_wall, T_hp_ret for 4R3C)
    - Current disturbances: T_amb, Qdot_gains
    - Optional weather forecast: T_amb at future timesteps (weather_forecast_steps)
    - Optional goal temperature appended when goal_based=True

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
        weather_forecast_steps: Optional[List[int]] = None,
        # Simulation parameters
        delta_t: int = 900,
        days: int = None,
        random_init: bool = False,
        backend: str = "jax",
        device: str = "cpu",
        return_numpy: bool = False,
        termination_fn: Optional[Callable] = None,
        randomization_events: Optional[List[EventSpec]] = None,
        randomization_seed: Optional[int] = None,
        allow_weather_download: bool = False,
        city: Optional[str] = None,
        batch_mode: str = "vmap",
        saveat_mode: str = "ts",
        norm_mode: str = "rms",
        fast_cost: bool = False,
        integration_steps: int = 10,
        step_mode: str = "adaptive",
        integrator: str = "diffrax",
        # Goal-based learning parameters
        goal_based: bool = False,
        goal_temp_range: Tuple[float, float] = (19.0, 28.0),
        # Reward shaping parameters
        temp_deviation_weight: float = 0.0,
        # Observation noise
        noise_level: float = 0.0,
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
            Steps ahead (in multiples of delta_t) to append T_amb forecasts.
        delta_t : int
            Sampling interval in seconds (default: 900).
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
        batch_mode : str
            JAX batch mode ("vmap" or "stacked") for Diffrax integration.
        saveat_mode : str
            Diffrax save mode ("ts" for intermediate states or "t1" for final only).
        norm_mode : str
            Norm used by the adaptive stepsize controller ("rms" or "max").
        fast_cost : bool
            If True and saveat_mode is "t1", compute costs from final state only.
        integration_steps : int
            Number of fixed integration steps per env step.
        step_mode : str
            Integration step mode ("adaptive" or "fixed").
        integrator : str
            Integration backend ("diffrax" or "jax_rk4").
        """
        super(RoomHeatEnv, self).__init__()

        # Core simulation parameters
        self.delta_t = delta_t  # Store internally as timestep for compatibility
        self.days = days
        self.method = method
        self.noise_level = noise_level
        self.backend = backend
        self.device = device
        self.return_numpy = return_numpy
        self.termination_fn = termination_fn
        self.event_manager = EventManager(randomization_events, seed=randomization_seed) if randomization_events else None
        self.allow_weather_download = allow_weather_download
        self.city = city
        self.batch_mode = batch_mode
        self.saveat_mode = saveat_mode
        self.norm_mode = norm_mode
        self.fast_cost = fast_cost
        self.integration_steps = integration_steps
        self.step_mode = step_mode
        self.integrator = integrator
        self.internal_gain_profile = internal_gain_profile
        self._resolve_runtime_options()
        
        # Goal-based learning
        self.goal_based = goal_based
        self.goal_temp_range = goal_temp_range
        self.goal_temperature = 20.0  # Default, will be randomized in reset()
        
        # Reward shaping
        self.temp_deviation_weight = temp_deviation_weight
        
        # Initialize building model
        if building not in BUILDING_NAMES2CLASS.keys():
            raise ValueError(f"Building {building} not in the list of available buildings")
        
        self.building = BUILDING_NAMES2CLASS[building]
        self.env_params = dict(self.building)
        self.env_params['mdot_hp'] = mdot_HP
        if self.event_manager:
            self.env_params = self.event_manager.apply("on_start", [self.env_params])[0]
            self.building = dict(self.env_params)
        self.bldg_model = Building(
            params=self.building,
            mdot_hp=self.env_params.get('mdot_hp', mdot_HP),
            method=self.method
        )
        
        # Initialize heat pump model
        self.hp_model = getattr(model_hvac, hp_model)(mdot_HP=mdot_HP)
        self.hp_model_name = hp_model
        
        # Initialize simulator
        if self.backend == "jax":
            try:
                import jax
                import jax.numpy as jnp
            except Exception as e:
                raise ImportError("JAX backend requested but jax is not installed.") from e
            self.jax = jax
            self.jnp = jnp
            if self.device == "gpu":
                devices = jax.devices("gpu")
                if not devices:
                    raise RuntimeError("GPU device requested but no GPU devices available.")
                self.jax_device = devices[0]
            else:
                self.jax_device = jax.devices("cpu")[0]
            self.simulator = JaxSimulator(
                method=self.method,
                timestep=self.delta_t,
                batch_mode=self.batch_mode,
                saveat_mode=self.saveat_mode,
                norm_mode=self.norm_mode,
                integration_steps=self.integration_steps,
                step_mode=self.step_mode,
                integrator=self.integrator,
            )
            self.hp_params = hp_params(self.hp_model_name, mdot_HP)
        elif self.backend == "legacy":
            self.simulator = Model_simulator(
                hp_model=self.hp_model,
                bldg_model=self.bldg_model,
                timestep=self.delta_t,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
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
        self._load_disturbances(self.internal_gain_profile)
        
        # Define observation space
        self.obs_keys = self.bldg_model.state_keys
        self.p_keys = ["T_amb", "Qdot_gains"]
        self.p_keys_dyn = list(self.p_keys)
        if self.method in ("6R4C", "7R5C"):
            self.p_keys_dyn += ["Qdot_int", "Qdot_sol"]
        self.weather_forecast_steps = weather_forecast_steps or []
        self.observation_space = self._create_observation_space()
        
        # Episode management
        self.max_t = self._calculate_max_timesteps()
        self.random_init = random_init
        self.t = 0  # Current timestep in weather data
        self._cur_steps = 0  # Steps in current episode
        self.state = None
        self.prev_action = None
        self._linear_param_signature = None
        self._noise_rng_key = None

        self.reset()

    def _resolve_runtime_options(self):
        self.integrator = str(self.integrator).lower()
        self.saveat_mode = str(self.saveat_mode).lower()
        self.step_mode = str(self.step_mode).lower()
        self.norm_mode = str(self.norm_mode).lower()
        self.batch_mode = str(self.batch_mode).lower()

        valid_integrators = {"auto", "diffrax", "jax_rk4", "linear"}
        if self.integrator not in valid_integrators:
            raise ValueError(f"Unknown integrator: {self.integrator}")

        if self.integrator == "auto":
            self.integrator = "linear" if self.method in LINEAR_METHODS else "jax_rk4"

        if self.integrator == "linear":
            if self.method not in LINEAR_METHODS:
                raise ValueError(f"Integrator 'linear' is not available for method {self.method}")
            self.saveat_mode = "t1"
            self.step_mode = "fixed"
            self.fast_cost = True

    def _load_disturbances(self, internal_gain_profile: str):
        """Load weather data and calculate total disturbances."""
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        
        pos = self.building["position"]
        if self.city:
            self.weather_data = load_weather_for_city(
                self.city,
                repo_filepath=str(repo_root),
                allow_download=self.allow_weather_download,
            )
        else:
            self.weather_data = load_weather(
                pos["lat"], pos["long"], pos["altitude"],
                tz=pos['timezone'],
                repo_filepath=str(repo_root),
                allow_download=self.allow_weather_download,
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
        ).astype(np.float32).resample(f'{self.delta_t}s').ffill()
        if self.method in ("6R4C", "7R5C"):
            # Append Qdot_int and Qdot_sol for higher-order models
            q_int = self.internal_gains_df['Qdot_tot'].rename('Qdot_int')
            q_sol = pd.Series(Qdot_sol, index=self.weather_data.index, name='Qdot_sol')
            self.p = pd.concat([self.p, q_int, q_sol], axis=1).astype(np.float32).resample(f'{self.delta_t}s').ffill()
        if self.days is not None:
            max_steps = int(self.days * 24 * (3600 / self.delta_t))
            if max_steps > 0 and max_steps < len(self.p):
                self.p = self.p.iloc[:max_steps]

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
            obs_low.extend([OBSERVATION_SPACE_LIMIT['T_amb'][0]] * len(self.weather_forecast_steps))
            obs_high.extend([OBSERVATION_SPACE_LIMIT['T_amb'][1]] * len(self.weather_forecast_steps))

        # Goal temperature (if goal-based)
        if self.goal_based:
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
        
        max_t = self.days * 24 * int(3600 / self.delta_t)
        if max_t >= len(self.p):
            max_t = len(self.p) - 1
        return max_t

    def _sample_goal_temperature(self) -> float:
        """Sample a goal temperature with 1 decimal precision."""
        # Generate temperature between min and max with 0.1 degree precision
        min_temp, max_temp = self.goal_temp_range
        n_steps = int((max_temp - min_temp) * 10) + 1
        temp = np.random.choice(np.linspace(min_temp, max_temp, n_steps))
        return round(float(temp), 1)

    def _build_observation(self, state_dict: Dict):
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
        
        if self.backend == "jax" and not self.return_numpy:
            return self.jnp.array(obs, dtype=self.jnp.float32)
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
        # Apply interval randomization
        if self.event_manager:
            self.env_params = self.event_manager.apply("on_interval", [self.env_params], step=self._cur_steps)[0]
            self.building = dict(self.env_params)
            self.bldg_model.params.update(self.building)

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
        if self.backend == "jax":
            T_hp_sup_set = float(self.simulator.apply_hp_constraints(self.jnp.array(T_hp_sup_set),
                                                                     self.jnp.array(state_dict['T_hp_ret']),
                                                                     {**self.hp_params, 'mdot_hp': self.bldg_model.mdot_hp}))
        else:
            T_hp_sup_set = self.hp_model.check_hp(T_hp_sup_set, state_dict['T_hp_ret'])
        self.prev_action = T_hp_sup_set

        # Simulate one step
        if self.backend == "jax":
            with self.jax.default_device(self.jax_device):
                x = self.jnp.array([state_dict[k] for k in self.obs_keys], dtype=self.jnp.float32)[None, :]
                p_vec = self.jnp.array([pk[k] for k in self.p_keys_dyn], dtype=self.jnp.float32)[None, :]
                u = self.jnp.array([T_hp_sup_set], dtype=self.jnp.float32)
            params = compute_derived_params(self.building)
            params['mdot_hp'] = self.bldg_model.mdot_hp
            with self.jax.default_device(self.jax_device):
                params_batch = {k: self.jnp.array([v], dtype=self.jnp.float32) for k, v in params.items() if isinstance(v, (int, float))}
                if self.simulator.integrator == "linear":
                    linear_signature = tuple(
                        (k, float(params_batch[k][0]))
                        for k in sorted(params_batch.keys())
                    )
                    if linear_signature != self._linear_param_signature:
                        self.simulator.prepare_linear(
                            params_batch,
                            state_dim=x.shape[1],
                            p_dim=p_vec.shape[1],
                            dtype=self.jnp.float32,
                        )
                        self._linear_param_signature = linear_signature
                ys, next_state_arr = self.simulator.step(x, u, p_vec, params_batch)
            next_state = {k: float(next_state_arr[0, i]) for i, k in enumerate(self.obs_keys)}
            # compute operative temp for methods 6R4C and 7R5C
            if "T_op" in self.obs_keys:
                if "T_surf_floor" in self.obs_keys:
                    f_floor = params['area_floor'] / params['A_surf']
                    next_state["T_surf"] = f_floor * next_state['T_surf_floor'] + (1 - f_floor) * next_state['T_surf_wall']
                next_state["T_op"] = 0.7 * next_state["T_room"] + 0.3 * next_state.get("T_surf", next_state.get("T_surf_wall", 0.0))
            if self.fast_cost and self.saveat_mode == "t1":
                T_room = next_state_arr[0, 0]
                dev_neg = self.jnp.maximum(self.bldg_model.T_room_set_lower - T_room, 0.0)
                dev_pos = self.jnp.maximum(T_room - self.bldg_model.T_room_set_upper, 0.0)
                costs = {
                    'dev_neg_sum': float(dev_neg * self.delta_t / 3600.0),
                    'dev_neg_max': float(dev_neg),
                    'dev_pos_sum': float(dev_pos * self.delta_t / 3600.0),
                    'dev_pos_max': float(dev_pos),
                }
                T_hp_ret = next_state_arr[0, -1]
                hp_cost = self.simulator.calc_cost(self.jnp.array([T_hp_ret]), u[0], self.jnp.array(pk['T_amb']), {**self.hp_params, 'mdot_hp': self.bldg_model.mdot_hp})
            else:
                # comfort deviation
                T_room_series = ys[0, :, 0]
                dev_neg = self.jnp.maximum(self.bldg_model.T_room_set_lower - T_room_series, 0.0)
                dev_pos = self.jnp.maximum(T_room_series - self.bldg_model.T_room_set_upper, 0.0)
                costs = {
                    'dev_neg_sum': float(self.jnp.mean(dev_neg) * self.delta_t / 3600.0),
                    'dev_neg_max': float(self.jnp.max(dev_neg)),
                    'dev_pos_sum': float(self.jnp.mean(dev_pos) * self.delta_t / 3600.0),
                    'dev_pos_max': float(self.jnp.max(dev_pos)),
                }
                T_hp_ret_series = ys[0, :, -1]
                hp_cost = self.simulator.calc_cost(T_hp_ret_series, u[0], self.jnp.array(pk['T_amb']), {**self.hp_params, 'mdot_hp': self.bldg_model.mdot_hp})
            costs.update({k: float(v) for k, v in hp_cost.items()})
        else:
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
        terminated = False

        # Build info dictionary
        info = {
            "cost": float(costs["dev_neg_max"]), # Reserved for SafeRL
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

        if self.termination_fn is not None:
            terminated, truncated, extra = self.termination_fn(next_state, self.building, self.t, info)
            if extra:
                info.update(extra)
        
        # Add noise to observation
        obs = self.state.copy()
        if self.noise_level > 0:
            if self.backend == "jax" and not self.return_numpy:
                self._noise_rng_key, subkey = self.jax.random.split(self._noise_rng_key)
                obs = obs + self.jax.random.normal(subkey, obs.shape) * self.noise_level
            else:
                obs += np.random.normal(0, self.noise_level, obs.shape)
        
        if self.backend == "jax" and self.return_numpy:
            obs = np.array(obs)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial observation and info."""
        super().reset(seed=seed)

        if self.event_manager:
            self.env_params = self.event_manager.apply("on_reset", [self.env_params])[0]
            self.building = dict(self.env_params)
            self.bldg_model = Building(
                params=self.building,
                mdot_hp=self.env_params.get('mdot_hp', self.bldg_model.mdot_hp),
                method=self.method
            )
            self.hp_model = getattr(model_hvac, self.hp_model_name)(mdot_HP=self.env_params.get('mdot_hp', self.bldg_model.mdot_hp))
            if self.backend == "legacy":
                self.simulator = Model_simulator(
                    hp_model=self.hp_model,
                    bldg_model=self.bldg_model,
                    timestep=self.delta_t,
                )
            self._load_disturbances(self.internal_gain_profile)
        
        # Sample new goal temperature if goal-based
        if self.goal_based:
            self.goal_temperature = self._sample_goal_temperature()
        
        # Determine starting timestep
        if self.random_init:
            max_offset = max(self.weather_forecast_steps) if self.weather_forecast_steps else 0
            max_start = self.p.shape[0] - self.max_t - 1 - max_offset
            if max_start > 0:
                self.t = np.random.randint(0, max_start)
            else:
                self.t = 0
        else:
            self.t = 0
        
        # Initialize state
        self.state = self._create_initial_state()
        self.prev_action = None
        self._cur_steps = 0
        if self.backend == "jax" and self.noise_level > 0:
            noise_seed = seed if seed is not None else 0
            self._noise_rng_key = self.jax.random.PRNGKey(noise_seed)
        
        obs = self.state.copy()
        if self.backend == "jax" and self.return_numpy:
            obs = np.array(obs)
        return obs, {}
    
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
        obs = self.state.copy()
        if self.backend == "jax" and self.return_numpy:
            obs = np.array(obs)
        return obs
    
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
            timestep=self.delta_t,
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
        ).astype(np.float32).resample(f'{self.delta_t}s').ffill()
        
        self.reset()
            
    def render(self, mode='human'):
        raise NotImplementedError
