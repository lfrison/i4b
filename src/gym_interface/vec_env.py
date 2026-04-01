from __future__ import annotations

from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
import gymnasium as gym

from src.gym_interface import BUILDING_NAMES2CLASS
from src.gym_interface.constant import OBSERVATION_SPACE_LIMIT
from src.disturbances import load_weather, load_weather_for_city, get_solar_gains, get_int_gains
from src.core.params import compute_derived_params, stack_params
from src.core.sim import JaxSimulator, LINEAR_METHODS
from src.core.hvac import hp_params
from src.randomization import EventManager, EventSpec
from src.constants import C_WATER_SPEC

# Params that affect only cost/control logic, NOT the RC dynamics matrices.
# When domain randomization touches only these keys the linear matrices (Ad, Bd, Cd)
# stay valid and _build_step_fn() does not need to be called again.
_NON_DYNAMICS_KEYS: frozenset = frozenset({
    "T_amb_lim", "T_offset", "T_room_set_lower", "T_room_set_upper", "cop_scale",
})


class RoomHeatVecEnv(gym.vector.VectorEnv):
    """Batched room-heating simulator with JAX-first execution.

    Defaults are tuned for high-throughput parallel rollout:
    - `integrator="auto"` resolves to `linear` for linear RC models.
    - Time/state stay on device between calls.
    - `performance_mode=True` forces cheapest safe settings for large batches.
    """
    @property
    def num_envs(self):
        return self._num_envs

    def __init__(
        self,
        num_envs: int,
        hp_model: str,
        building: str | List[str],
        method: str,
        mdot_HP: float,
        internal_gain_profile: str,
        weather_forecast_steps: Optional[List[int]] = None,
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
        cities: Optional[List[str]] = None,
        steps_per_call: int = 1,
        batch_mode: str = "stacked",
        saveat_mode: str = "t1",
        norm_mode: str = "max",
        fast_cost: bool = True,
        integration_steps: int = 10,
        step_mode: str = "fixed",
        integrator: str = "auto",
        return_trajectories: bool = False,
        compute_metrics: bool = False,
        sync_time: bool = True,
        performance_mode: bool = False,
        diagnostics: bool = False,
    ):
        self._num_envs = num_envs
        self.backend = backend
        self.device = device
        self.return_numpy = return_numpy
        self.termination_fn = termination_fn
        self.event_manager = EventManager(randomization_events, seed=randomization_seed) if randomization_events else None
        self.allow_weather_download = allow_weather_download
        self.weather_forecast_steps = list(weather_forecast_steps or [])
        self.delta_t = delta_t
        self.method = method
        self.days = days
        self.random_init = random_init
        self.internal_gain_profile = internal_gain_profile
        self.cities = cities
        self.steps_per_call = max(1, int(steps_per_call))
        self.batch_mode = batch_mode
        self.saveat_mode = saveat_mode
        self.norm_mode = norm_mode
        self.fast_cost = fast_cost
        self.integration_steps = integration_steps
        self.step_mode = step_mode
        self.integrator = integrator
        self.return_trajectories = return_trajectories
        self.compute_metrics = compute_metrics
        self.sync_time = sync_time
        self.performance_mode = performance_mode
        self.diagnostics = diagnostics

        self._resolve_runtime_options()

        if self.backend != "jax":
            raise ValueError("RoomHeatVecEnv currently supports only JAX backend.")
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
            try:
                self.jax_device = jax.devices("cpu")[0]
            except Exception as e:
                raise RuntimeError("CPU backend not available. Set JAX_PLATFORMS=cpu to run CPU benchmarks.") from e

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
        self.hp_model_name = hp_model
        self.hp_params = hp_params(self.hp_model_name, mdot_HP)
        self._cop_fn = self.hp_params["cop_fn"]
        with self.jax.default_device(self.jax_device):
            self._env_idx = self.jnp.arange(self._num_envs, dtype=self.jnp.int32)
            self._env_idx_row = self._env_idx[None, :]

        if isinstance(building, list):
            if len(building) != num_envs:
                raise ValueError("If building is a list, its length must match num_envs.")
            building_keys = building
        else:
            building_keys = [building] * num_envs

        self.buildings = [BUILDING_NAMES2CLASS[b] for b in building_keys]
        self.env_params = [dict(b) for b in self.buildings]
        for p in self.env_params:
            p['mdot_hp'] = mdot_HP

        self._load_disturbances(self.internal_gain_profile, self.cities)
        self._build_param_arrays()

        self.obs_keys = self._state_keys()
        self.p_keys = ["T_amb", "Qdot_gains"]
        self.p_keys_dyn = list(self.p_keys)
        if self.method in ("6R4C", "7R5C"):
            self.p_keys_dyn += ["Qdot_int", "Qdot_sol"]
        self._py_step_count: int = 0
        self._is_linear: bool = (self.integrator == "linear")
        self._build_step_fn()
        self._build_auto_reset_fn()

        obs_low = [OBSERVATION_SPACE_LIMIT[key][0] for key in self.obs_keys]
        obs_high = [OBSERVATION_SPACE_LIMIT[key][1] for key in self.obs_keys]
        obs_low.extend([OBSERVATION_SPACE_LIMIT['T_amb'][0], OBSERVATION_SPACE_LIMIT['Qdot_gains'][0]])
        obs_high.extend([OBSERVATION_SPACE_LIMIT['T_amb'][1], OBSERVATION_SPACE_LIMIT['Qdot_gains'][1]])
        if len(self.weather_forecast_steps) > 0:
            obs_low.extend([OBSERVATION_SPACE_LIMIT['T_amb'][0]] * len(self.weather_forecast_steps))
            obs_high.extend([OBSERVATION_SPACE_LIMIT['T_amb'][1]] * len(self.weather_forecast_steps))

        self.single_observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.action_low = 20.0
        self.action_high = 65.0
        self.t = None
        self._cur_steps = None
        self.state = None
        self.reset()

    # ---------------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------------

    def _resolve_runtime_options(self):
        """Normalize integration options for throughput-oriented vector simulation."""
        self.integrator = str(self.integrator).lower()
        self.batch_mode = str(self.batch_mode).lower()
        self.saveat_mode = str(self.saveat_mode).lower()
        self.step_mode = str(self.step_mode).lower()
        self.norm_mode = str(self.norm_mode).lower()
        valid_integrators = {"auto", "diffrax", "jax_rk4", "linear"}
        if self.integrator not in valid_integrators:
            raise ValueError(f"Unknown integrator: {self.integrator}")

        auto_integrator = "linear" if self.method in LINEAR_METHODS else "jax_rk4"
        if self.integrator == "auto":
            self.integrator = auto_integrator

        if self.performance_mode:
            self.saveat_mode = "t1"
            self.fast_cost = True
            self.compute_metrics = False
            self.sync_time = True
            self.step_mode = "fixed"
            if self.integrator == "diffrax":
                self.batch_mode = "stacked"
                self.norm_mode = "max"

        if self.integrator == "linear":
            if self.method not in LINEAR_METHODS:
                raise ValueError(f"Integrator 'linear' is not available for method {self.method}")
            self.saveat_mode = "t1"
            self.step_mode = "fixed"
            self.fast_cost = True
            self.batch_mode = "stacked"

    def _state_keys(self):
        if self.method == "2R2C":
            return ("T_room", "T_hp_ret")
        if self.method == "4R3C":
            return ("T_room", "T_wall", "T_hp_ret")
        if self.method == "5R4C":
            return ("T_room", "T_int", "T_wall", "T_hp_ret")
        if self.method == "6R4C":
            return ("T_room", "T_surf", "T_op", "T_mass", "T_hp_ret")
        if self.method == "7R5C":
            return ("T_room", "T_surf_wall", "T_op", "T_mass", "T_surf_floor", "T_hp_ret")
        raise ValueError(f"Unknown method: {self.method}")

    # ---------------------------------------------------------------------------
    # Disturbance loading
    # ---------------------------------------------------------------------------

    def _load_disturbances(self, internal_gain_profile: str, cities: Optional[List[str]]):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        self.p_list = []
        p_cache = {}
        unique_profiles = []
        key_to_unique_idx: Dict[tuple, int] = {}
        env_to_unique_idx: List[int] = []
        for i in range(self.num_envs):
            bldg = self.buildings[i]
            pos = bldg["position"]
            city = None
            if cities and len(cities) == self.num_envs and cities[i]:
                city = cities[i]
            key = self._disturbance_key(bldg, city)
            p = p_cache.get(key)
            if p is None:
                if city:
                    weather = load_weather_for_city(city, repo_filepath=str(repo_root),
                                                    allow_download=self.allow_weather_download)
                else:
                    weather = load_weather(pos["lat"], pos["long"], pos["altitude"],
                                           tz=pos["timezone"], repo_filepath=str(repo_root),
                                           allow_download=self.allow_weather_download)
                int_gains = get_int_gains(time=weather.index,
                                          profile_path=str(repo_root / internal_gain_profile),
                                          bldg_area=bldg['area_floor'])
                q_sol = get_solar_gains(weather=weather, bldg_params=bldg)
                q_gains = pd.DataFrame(q_sol + int_gains['Qdot_tot'], columns=['Qdot_gains'])
                p = pd.concat([weather['T_amb'], q_gains], axis=1).astype(np.float32).resample(f'{self.delta_t}s').ffill()
                if self.method in ("6R4C", "7R5C"):
                    q_int = int_gains['Qdot_tot'].rename('Qdot_int')
                    q_sol_s = pd.Series(q_sol, index=weather.index, name='Qdot_sol')
                    p = pd.concat([p, q_int, q_sol_s], axis=1).astype(np.float32).resample(f'{self.delta_t}s').ffill()
                if self.days is not None:
                    max_steps = int(self.days * 24 * (3600 / self.delta_t))
                    if max_steps > 0 and max_steps < len(p):
                        p = p.iloc[:max_steps]
                p_cache[key] = p
                key_to_unique_idx[key] = len(unique_profiles)
                unique_profiles.append(p)
            self.p_list.append(p)
            env_to_unique_idx.append(key_to_unique_idx[key])
        with self.jax.default_device(self.jax_device):
            # Store only unique profiles: shape (T, n_unique, F) instead of (T, n_envs, F).
            # When all envs share one building+location, n_unique=1 and VRAM drops by n_envs×.
            p_np = np.stack([p.values for p in unique_profiles], axis=0)  # (n_unique, T, F)
            p_np = np.transpose(p_np, (1, 0, 2))                          # (T, n_unique, F)
            self.p_arr = self.jnp.array(p_np, dtype=self.jnp.float32)
            # p_env_idx[i] = which unique profile env i uses.
            self.p_env_idx = self.jnp.array(env_to_unique_idx, dtype=self.jnp.int32)
            self.p_env_idx_row = self.p_env_idx[None, :]  # (1, n_envs) for seq indexing

    def _disturbance_key(self, bldg: Dict, city: Optional[str]):
        pos = bldg.get("position", {})
        if city:
            weather_key = ("city", str(city))
        else:
            weather_key = (
                "pos",
                float(pos.get("lat", 0.0)),
                float(pos.get("long", 0.0)),
                float(pos.get("altitude", 0.0)),
                str(pos.get("timezone", "")),
            )
        windows = bldg.get("windows", [])
        windows_key = tuple(
            (
                float(w.get("area", 0.0)),
                float(w.get("tilt", 0.0)),
                float(w.get("azimuth", 0.0)),
                float(w.get("g_value", 0.0)),
                float(w.get("c_frame", 0.0)),
                float(w.get("c_shade", 0.0)),
            )
            for w in windows
        )
        return (
            weather_key,
            float(pos.get("lat", 0.0)),
            float(pos.get("long", 0.0)),
            float(pos.get("altitude", 0.0)),
            str(pos.get("timezone", "")),
            float(bldg.get("area_floor", 0.0)),
            windows_key,
            str(self.internal_gain_profile),
            int(self.delta_t),
            str(self.method),
            None if self.days is None else int(self.days),
        )

    # ---------------------------------------------------------------------------
    # Parameter arrays
    # ---------------------------------------------------------------------------

    def _build_param_arrays(self):
        if self.event_manager:
            self.env_params = self.event_manager.apply("on_start", self.env_params)
        derived = [compute_derived_params(p) for p in self.env_params]
        for i, d in enumerate(derived):
            d['mdot_hp'] = self.env_params[i].get('mdot_hp', self.hp_params['mdot_hp'])
            d['cop_scale'] = self.env_params[i].get('cop_scale', 1.0)
            d['T_room_set_lower'] = self.env_params[i].get('T_room_set_lower', 20.0)
            d['T_room_set_upper'] = self.env_params[i].get('T_room_set_upper', 22.0)
        self.params = stack_params(derived)
        with self.jax.default_device(self.jax_device):
            self.params = {k: self.jnp.array(v) for k, v in self.params.items()}

    def _dynamics_changed(self, old_env_params: List[Dict]) -> bool:
        """Return True if any dynamics-relevant param changed vs old_env_params.

        Compares Python-level dicts (no GPU sync needed).
        """
        for old, new in zip(old_env_params, self.env_params):
            for k, v in new.items():
                if k in _NON_DYNAMICS_KEYS:
                    continue
                if old.get(k) != v:
                    return True
        return False

    # ---------------------------------------------------------------------------
    # JIT compilation — step / rollout kernels
    # ---------------------------------------------------------------------------

    def _build_step_fn(self):
        """Compile per-step and rollout JAX functions for the current configuration.

        For the linear integrator, ``prepare_linear`` is called first to compute
        exact discrete-time matrices (Ad, Bd, Cd) via GPU-parallel matrix
        exponential. The XLA kernel is reused when shapes are unchanged; only
        values (params, matrices) may vary without recompilation.
        """
        if getattr(self.simulator, "integrator", None) == "linear":
            self.simulator.prepare_linear(
                self.params,
                state_dim=len(self.obs_keys),
                p_dim=len(self.p_keys_dyn),
                dtype=self.jnp.float32,
            )
        n = self.num_envs
        p_env_idx = self.p_env_idx
        p_keys_len = len(self.p_keys)
        p_dyn_len = len(self.p_keys_dyn)
        delta_t = float(self.delta_t)
        is_linear = (self.integrator == "linear")

        def _one_step_core(state, t, actions, p_arr, params, linear_mats):
            """Single environment step kernel, shared by _step_fn and rollout scan.

            ``params`` and ``linear_mats`` are explicit JIT arguments so updating
            their values on domain-randomization events never triggers recompilation.
            ``p_arr`` has shape (T, n_unique, F); ``p_env_idx`` (closed over) maps
            each env to its unique disturbance profile.
            """
            actions = actions.reshape((n,))
            T_hp_sup_set = actions * (self.action_high - self.action_low) / 2 + (self.action_low + self.action_high) / 2

            p_vec = p_arr[t, p_env_idx, :p_dyn_len]
            p_cur = p_vec[:, :p_keys_len]
            T_amb = p_cur[:, 0]
            T_hp_ret = state[:, -1]

            T_hp_sup_set = self.jnp.where(
                T_amb < params['T_amb_lim'],
                T_hp_sup_set,
                T_hp_ret,
            )
            T_hp_sup_set = self.simulator.apply_hp_constraints(T_hp_sup_set, T_hp_ret, params)

            ys, next_state = self.simulator.step(state, T_hp_sup_set, p_vec, params, linear_mats)

            if self.method == "6R4C":
                T_room = next_state[:, 0]
                T_surf = next_state[:, 1]
                T_op = 0.7 * T_room + 0.3 * T_surf
                next_state = next_state.at[:, 2].set(T_op)
            elif self.method == "7R5C":
                f_floor = params['area_floor'] / params['A_surf']
                T_surf = f_floor * next_state[:, 4] + (1 - f_floor) * next_state[:, 1]
                T_op = 0.7 * next_state[:, 0] + 0.3 * T_surf
                next_state = next_state.at[:, 2].set(T_op)

            if self.fast_cost and self.saveat_mode == "t1":
                T_room = next_state[:, 0]
                if self.compute_metrics:
                    dev_neg = self.jnp.maximum(params['T_room_set_lower'] - T_room, 0.0)
                    dev_pos = self.jnp.maximum(T_room - params['T_room_set_upper'], 0.0)
                    dev_neg_sum = dev_neg * delta_t / 3600.0
                    dev_neg_max = dev_neg
                    dev_pos_sum = dev_pos * delta_t / 3600.0
                    dev_pos_max = dev_pos
                else:
                    dev_neg_sum = self.jnp.zeros((n,), dtype=next_state.dtype)
                    dev_neg_max = self.jnp.zeros((n,), dtype=next_state.dtype)
                    dev_pos_sum = self.jnp.zeros((n,), dtype=next_state.dtype)
                    dev_pos_max = self.jnp.zeros((n,), dtype=next_state.dtype)

                T_hp_ret = next_state[:, -1]
                cop = self._cop_fn(T_hp_sup_set, T_amb) * params['cop_scale']
                Qdot_th = params['mdot_hp'] * C_WATER_SPEC * (T_hp_sup_set - T_hp_ret)
            else:
                T_room_series = ys[:, :, 0]
                if self.compute_metrics:
                    dev_neg = self.jnp.maximum(params['T_room_set_lower'][:, None] - T_room_series, 0.0)
                    dev_pos = self.jnp.maximum(T_room_series - params['T_room_set_upper'][:, None], 0.0)
                    dev_neg_sum = self.jnp.mean(dev_neg, axis=1) * delta_t / 3600.0
                    dev_neg_max = self.jnp.max(dev_neg, axis=1)
                    dev_pos_sum = self.jnp.mean(dev_pos, axis=1) * delta_t / 3600.0
                    dev_pos_max = self.jnp.max(dev_pos, axis=1)
                else:
                    dev_neg_sum = self.jnp.zeros((n,), dtype=next_state.dtype)
                    dev_neg_max = self.jnp.zeros((n,), dtype=next_state.dtype)
                    dev_pos_sum = self.jnp.zeros((n,), dtype=next_state.dtype)
                    dev_pos_max = self.jnp.zeros((n,), dtype=next_state.dtype)

                T_hp_ret_series = ys[:, :, -1]
                cop = self._cop_fn(T_hp_sup_set, T_amb) * params['cop_scale']
                Qdot_th = self.jnp.mean(params['mdot_hp'][:, None] * C_WATER_SPEC * (T_hp_sup_set[:, None] - T_hp_ret_series), axis=1)
            Qdot_th = self.jnp.maximum(Qdot_th, 0.0)
            P_el = Qdot_th / cop
            E_el = P_el * delta_t / 3600.0

            return next_state, p_cur, E_el, dev_neg_sum, dev_neg_max, dev_pos_sum, dev_pos_max

        # Build outer _step_fn with the right signature depending on integrator type.
        # For linear: linear_mats is an explicit arg (dynamic values, no recompile on update).
        # For non-linear: linear_mats is always None (Python constant, elided from JIT args).
        if is_linear:
            if self.return_trajectories:
                def _step_fn(state, t, actions, p_arr, params, linear_mats):
                    def _scan_fn(carry, _):
                        s, tt = carry
                        ns, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = _one_step_core(s, tt, actions, p_arr, params, linear_mats)
                        next_t = tt + 1
                        return (ns, next_t), (ns, next_t, p_cur, E_el, dn_s, dn_m, dp_s, dp_m)

                    (final_state, final_t), outs = self.jax.lax.scan(
                        _scan_fn, (state, t), None, length=self.steps_per_call
                    )
                    state_seq, t_seq, p_seq, E_el, dn_s, dn_m, dp_s, dp_m = outs
                    obs_seq = self._build_observation_seq(state_seq, t_seq, p_seq, p_arr_full=p_arr)
                    return final_state, final_t, obs_seq, t_seq, E_el, dn_s, dn_m, dp_s, dp_m
            else:
                def _step_fn(state, t, actions, p_arr, params, linear_mats):
                    zero_vec = self.jnp.zeros((n,), dtype=state.dtype)
                    p_zero = self.jnp.zeros((n, p_keys_len), dtype=state.dtype)
                    init = (state, t, p_zero, zero_vec, zero_vec, zero_vec, zero_vec, zero_vec)

                    def _body(_, carry):
                        s, tt, _, _, _, _, _, _ = carry
                        ns, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = _one_step_core(s, tt, actions, p_arr, params, linear_mats)
                        return (ns, tt + 1, p_cur, E_el, dn_s, dn_m, dp_s, dp_m)

                    final_state, final_t, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = self.jax.lax.fori_loop(
                        0, self.steps_per_call, _body, init
                    )
                    obs = self._build_observation(final_state, p_cur, p_arr_full=p_arr, t=t)
                    return final_state, final_t, obs, E_el, dn_s, dn_m, dp_s, dp_m
        else:
            if self.return_trajectories:
                def _step_fn(state, t, actions, p_arr, params):
                    def _scan_fn(carry, _):
                        s, tt = carry
                        ns, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = _one_step_core(s, tt, actions, p_arr, params, None)
                        next_t = tt + 1
                        return (ns, next_t), (ns, next_t, p_cur, E_el, dn_s, dn_m, dp_s, dp_m)

                    (final_state, final_t), outs = self.jax.lax.scan(
                        _scan_fn, (state, t), None, length=self.steps_per_call
                    )
                    state_seq, t_seq, p_seq, E_el, dn_s, dn_m, dp_s, dp_m = outs
                    obs_seq = self._build_observation_seq(state_seq, t_seq, p_seq, p_arr_full=p_arr)
                    return final_state, final_t, obs_seq, t_seq, E_el, dn_s, dn_m, dp_s, dp_m
            else:
                def _step_fn(state, t, actions, p_arr, params):
                    zero_vec = self.jnp.zeros((n,), dtype=state.dtype)
                    p_zero = self.jnp.zeros((n, p_keys_len), dtype=state.dtype)
                    init = (state, t, p_zero, zero_vec, zero_vec, zero_vec, zero_vec, zero_vec)

                    def _body(_, carry):
                        s, tt, _, _, _, _, _, _ = carry
                        ns, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = _one_step_core(s, tt, actions, p_arr, params, None)
                        return (ns, tt + 1, p_cur, E_el, dn_s, dn_m, dp_s, dp_m)

                    final_state, final_t, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = self.jax.lax.fori_loop(
                        0, self.steps_per_call, _body, init
                    )
                    obs = self._build_observation(final_state, p_cur, p_arr_full=p_arr, t=t)
                    return final_state, final_t, obs, E_el, dn_s, dn_m, dp_s, dp_m

        self._step_fn = self.jax.jit(_step_fn, donate_argnums=(0, 1))
        self._is_linear = is_linear
        # Expose the core for reuse in the scan rollout.
        self._one_step_core_fn = _one_step_core
        self._build_rollout_fn()

    def _build_rollout_fn(self):
        """Build a JIT-compiled scan rollout that runs n_rollout_steps entirely on device.

        Unlike step(), rollout() accepts a *sequence* of actions (one per step),
        runs them all inside a single jax.lax.scan kernel, and returns the full
        trajectory.  This replaces n_rollout_steps Python→GPU dispatch calls with
        one, eliminating the ~1 ms-per-call Python overhead for every step.
        """
        one_step = self._one_step_core_fn
        is_linear = self._is_linear

        if is_linear:
            def _rollout_fn(state, t, actions_seq, p_arr, params, linear_mats):
                # actions_seq: (n_rollout_steps, n_envs)
                def _body(carry, action):
                    s, tt = carry
                    ns, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = one_step(
                        s, tt, action, p_arr, params, linear_mats
                    )
                    obs = self._build_observation(ns, p_cur, p_arr_full=p_arr, t=tt)
                    return (ns, tt + 1), (obs, E_el)

                (final_state, final_t), traj = self.jax.lax.scan(
                    _body, (state, t), actions_seq
                )
                return final_state, final_t, traj
        else:
            def _rollout_fn(state, t, actions_seq, p_arr, params):
                def _body(carry, action):
                    s, tt = carry
                    ns, p_cur, E_el, dn_s, dn_m, dp_s, dp_m = one_step(
                        s, tt, action, p_arr, params, None
                    )
                    obs = self._build_observation(ns, p_cur, p_arr_full=p_arr, t=tt)
                    return (ns, tt + 1), (obs, E_el)

                (final_state, final_t), traj = self.jax.lax.scan(
                    _body, (state, t), actions_seq
                )
                return final_state, final_t, traj

        self._rollout_fn = self.jax.jit(_rollout_fn, donate_argnums=(0, 1))

    def rollout(self, actions_seq):
        """Run a full rollout on device via jax.lax.scan — no Python loop.

        Instead of calling step() n_rollout_steps times (n dispatches to GPU),
        this compiles the entire loop into a single XLA kernel and issues one
        dispatch, eliminating ~1 ms overhead per step.

        Args:
            actions_seq: array of shape ``(n_rollout_steps, n_envs)`` or
                ``(n_rollout_steps, n_envs, 1)`` — one action per env per step.

        Returns:
            obs_seq:    JAX array ``(n_rollout_steps, n_envs, obs_dim)``
            reward_seq: JAX array ``(n_rollout_steps, n_envs)``
            info:       dict with ``Q_el_kWh`` trajectory

        Side-effects:
            Updates ``self.state`` and ``self.t`` to the final state in-place.
        """
        with self.jax.default_device(self.jax_device):
            actions_j = self.jnp.asarray(actions_seq, dtype=self.jnp.float32)
            if actions_j.ndim == 3:
                actions_j = actions_j[:, :, 0]

            if self._is_linear:
                final_state, final_t, (obs_seq, E_el_seq) = self._rollout_fn(
                    self.state, self.t, actions_j, self.p_arr,
                    self.params, self.simulator._linear_mats
                )
            else:
                final_state, final_t, (obs_seq, E_el_seq) = self._rollout_fn(
                    self.state, self.t, actions_j, self.p_arr, self.params
                )

        self.state = final_state
        self.t = final_t
        n_steps = actions_j.shape[0]
        self._cur_steps = self._cur_steps + n_steps
        self._py_step_count += n_steps

        reward_seq = -E_el_seq
        info = {"Q_el_kWh": E_el_seq}
        if self.return_numpy:
            return np.array(obs_seq), np.array(reward_seq), {"Q_el_kWh": np.array(E_el_seq)}
        return obs_seq, reward_seq, info

    # ---------------------------------------------------------------------------
    # Observation building
    # ---------------------------------------------------------------------------

    def _build_observation(self, state_arr, p_arr, p_arr_full=None, t=None):
        obs = self.jnp.concatenate([state_arr, p_arr], axis=-1)
        if len(self.weather_forecast_steps) > 0:
            if p_arr_full is None:
                p_arr_full = self.p_arr
            if t is None:
                t = self.t
            t_arr = self.jnp.asarray(t, dtype=self.jnp.int32)
            forecasts = []
            for step in self.weather_forecast_steps:
                idx = self.jnp.clip(t_arr + step, 0, p_arr_full.shape[0] - 1)
                forecasts.append(p_arr_full[idx, self.p_env_idx, 0])
            obs = self.jnp.concatenate([obs, self.jnp.stack(forecasts, axis=-1)], axis=-1)
        return obs.astype(self.jnp.float32)

    def _build_observation_seq(self, state_seq, t_seq, p_seq, p_arr_full=None):
        obs = self.jnp.concatenate([state_seq, p_seq], axis=-1)
        if len(self.weather_forecast_steps) > 0:
            if p_arr_full is None:
                p_arr_full = self.p_arr
            forecasts = []
            for step in self.weather_forecast_steps:
                idx = self.jnp.clip(t_seq + step, 0, p_arr_full.shape[0] - 1)
                forecasts.append(p_arr_full[idx, self.p_env_idx_row, 0])
            obs = self.jnp.concatenate([obs, self.jnp.stack(forecasts, axis=-1)], axis=-1)
        return obs.astype(self.jnp.float32)

    # ---------------------------------------------------------------------------
    # Public API — reset / step / rollout / auto_reset
    # ---------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if self.event_manager:
            old_env_params = [dict(p) for p in self.env_params]
            new_env_params, changed = self.event_manager.apply_inplace("on_reset", self.env_params)
            if changed:
                self.env_params = new_env_params
                self.buildings = [dict(p) for p in self.env_params]
                self._load_disturbances(self.internal_gain_profile, self.cities)
                self._build_param_arrays()
                # Rebuild step fn when dynamics params change or always for safety on reset.
                if self.integrator == "linear" and not self._dynamics_changed(old_env_params):
                    # Only non-dynamics params changed: update linear mats (fast GPU expm)
                    # then rebuild step fn so the new mats are captured.
                    self.simulator.prepare_linear(
                        self.params,
                        state_dim=len(self.obs_keys),
                        p_dim=len(self.p_keys_dyn),
                        dtype=self.jnp.float32,
                    )
                    # _step_fn still valid (same structure) — no rebuild needed.
                else:
                    self._build_step_fn()
        self._py_step_count = 0
        init_temp = 20.0
        with self.jax.default_device(self.jax_device):
            self.t = self.jnp.zeros((self.num_envs,), dtype=self.jnp.int32)
            self._cur_steps = self.jnp.zeros((self.num_envs,), dtype=self.jnp.int32)
            state = self.jnp.zeros((self.num_envs, len(self.obs_keys)), dtype=self.jnp.float32)
            state = state + init_temp
        self.state = state
        p_cur = self.p_arr[0, self.p_env_idx, :len(self.p_keys)]
        obs = self._build_observation(self.state, p_cur, p_arr_full=self.p_arr, t=self.t)
        if self.return_numpy:
            return np.array(obs), {}
        return obs, {}

    def _build_auto_reset_fn(self):
        """Build a JIT-compiled masked reset that stays entirely on device."""
        n_obs = len(self.obs_keys)
        max_t = self.p_arr.shape[0] - 1

        @self.jax.jit
        def _masked_reset(state, t, cur_steps, done, init_temps, start_times):
            # Per-env randomized initial temperature and start time.
            init_state = self.jnp.broadcast_to(
                init_temps[:, None], (self.num_envs, n_obs)
            )
            new_state = self.jnp.where(done[:, None], init_state, state)
            # Clamp start times to valid range.
            clamped_t = self.jnp.clip(start_times, 0, max_t)
            new_t = self.jnp.where(done, clamped_t, t)
            new_cur_steps = self.jnp.where(
                done, self.jnp.zeros_like(cur_steps), cur_steps
            )
            return new_state, new_t, new_cur_steps

        self._auto_reset_fn = _masked_reset

    def auto_reset(self, done):
        """Reset done environments with domain randomization.

        Applies ``on_reset`` events to randomize building parameters for the
        done environments, then resets their state with randomized initial
        temperature and start time on device.

        Args:
            done: bool array of shape (num_envs,) — True for envs to reset.
        """
        done_j = self.jnp.asarray(done, dtype=bool)
        done_np = np.asarray(done_j)
        done_indices = np.where(done_np)[0]

        if len(done_indices) == 0:
            return

        # --- Apply domain randomization for done envs ---
        if self.event_manager:
            done_params = [self.env_params[i] for i in done_indices]
            updated, changed = self.event_manager.apply_inplace(
                "on_reset", done_params
            )
            if changed:
                for j, i in enumerate(done_indices):
                    self.env_params[i] = updated[j]
                # Rebuild param arrays and update JAX params for changed envs.
                self._rebuild_params_for_envs(done_indices)

        # --- Randomized initial temperature and start time ---
        rng = self.event_manager.rng if self.event_manager else np.random.default_rng()
        init_temps_np = np.full(self.num_envs, 20.0, dtype=np.float32)
        start_times_np = np.zeros(self.num_envs, dtype=np.int32)
        # Randomize only the done envs.
        n_done = len(done_indices)
        init_temps_np[done_indices] = rng.uniform(15.0, 25.0, size=n_done).astype(
            np.float32
        )
        max_t = int(self.p_arr.shape[0]) - 1
        if max_t > 0:
            start_times_np[done_indices] = rng.integers(
                0, max_t, size=n_done
            ).astype(np.int32)

        with self.jax.default_device(self.jax_device):
            init_temps_j = self.jnp.array(init_temps_np)
            start_times_j = self.jnp.array(start_times_np)

        self.state, self.t, self._cur_steps = self._auto_reset_fn(
            self.state, self.t, self._cur_steps, done_j, init_temps_j, start_times_j
        )

    def _rebuild_params_for_envs(self, indices):
        """Rebuild JAX param arrays after DR changed specific env params.

        Recomputes derived params for the given env indices and updates
        the corresponding rows in the JAX param dict in-place.
        """
        # Build updated numpy arrays from current params, patch changed envs.
        updated_arrays = {k: np.array(v, copy=True) for k, v in self.params.items()}
        for i in indices:
            d = compute_derived_params(self.env_params[i])
            d['mdot_hp'] = self.env_params[i].get('mdot_hp', self.hp_params['mdot_hp'])
            d['cop_scale'] = self.env_params[i].get('cop_scale', 1.0)
            d['T_room_set_lower'] = self.env_params[i].get('T_room_set_lower', 20.0)
            d['T_room_set_upper'] = self.env_params[i].get('T_room_set_upper', 22.0)
            for key, val in d.items():
                if key in updated_arrays:
                    updated_arrays[key][i] = val
        with self.jax.default_device(self.jax_device):
            self.params = {k: self.jnp.array(v) for k, v in updated_arrays.items()}

        # Rebuild linear mats if using linear integrator.
        if self.integrator == "linear":
            self.simulator.prepare_linear(
                self.params,
                state_dim=len(self.obs_keys),
                p_dim=len(self.p_keys_dyn),
                dtype=self.jnp.float32,
            )

    def _to_action_tensor(self, actions):
        actions_j = self.jnp.asarray(actions, dtype=self.jnp.float32)
        if actions_j.ndim == 1:
            if actions_j.shape[0] != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} actions, got {actions_j.shape[0]}")
            return actions_j
        if actions_j.ndim == 2:
            if actions_j.shape[0] != self.num_envs:
                raise ValueError(f"Expected actions shape ({self.num_envs}, *), got {tuple(actions_j.shape)}")
            return actions_j[:, 0]
        raise ValueError(f"Unsupported action shape: {tuple(actions_j.shape)}")

    def step(self, actions):
        if self.event_manager:
            # Pure-Python counter — no GPU sync.
            self._py_step_count += self.steps_per_call
            old_env_params = [dict(p) for p in self.env_params]
            new_env_params, changed = self.event_manager.apply_inplace(
                "on_interval", self.env_params, step=self._py_step_count
            )
            if changed:
                self.env_params = new_env_params
                self.buildings = [dict(p) for p in self.env_params]
                self._build_param_arrays()
                dynamics_changed = self._dynamics_changed(old_env_params)
                if dynamics_changed and self.integrator == "linear":
                    # Dynamics changed: recompute matrices (GPU expm) + rebuild step fn.
                    self._build_step_fn()
                elif dynamics_changed:
                    # Non-linear: params are explicit args, no step fn rebuild needed.
                    pass
                # Non-dynamics only: params JAX arrays updated, no rebuild needed.
        if self.diagnostics:
            import time
            t_start = time.perf_counter()
        with self.jax.default_device(self.jax_device):
            if self.diagnostics:
                t_prep = time.perf_counter()
            actions_j = self._to_action_tensor(actions)
            # Pass params (and linear_mats for linear integrator) as explicit JIT args.
            if self._is_linear:
                linear_mats = self.simulator._linear_mats
                if self.return_trajectories:
                    (next_state, next_t, obs, t_seq, E_el, dev_neg_sum, dev_neg_max, dev_pos_sum, dev_pos_max) = self._step_fn(
                        self.state, self.t, actions_j, self.p_arr, self.params, linear_mats
                    )
                else:
                    next_state, next_t, obs, E_el, dev_neg_sum, dev_neg_max, dev_pos_sum, dev_pos_max = self._step_fn(
                        self.state, self.t, actions_j, self.p_arr, self.params, linear_mats
                    )
            else:
                if self.return_trajectories:
                    (next_state, next_t, obs, t_seq, E_el, dev_neg_sum, dev_neg_max, dev_pos_sum, dev_pos_max) = self._step_fn(
                        self.state, self.t, actions_j, self.p_arr, self.params
                    )
                else:
                    next_state, next_t, obs, E_el, dev_neg_sum, dev_neg_max, dev_pos_sum, dev_pos_max = self._step_fn(
                        self.state, self.t, actions_j, self.p_arr, self.params
                    )

            reward = -E_el
            if self.return_trajectories:
                step_ids = self.jnp.arange(1, self.steps_per_call + 1)[:, None]
                cur_steps_seq = self._cur_steps[None, :] + step_ids
                truncated = (t_seq >= self.p_arr.shape[0] - 1) | (cur_steps_seq >= (self.p_arr.shape[0] - 1))
                terminated = self.jnp.zeros((self.steps_per_call, self.num_envs), dtype=bool)
                info = {"dev_sum": dev_neg_sum + dev_pos_sum, "dev_max": self.jnp.maximum(dev_neg_max, dev_pos_max), "Q_el_kWh": E_el}
            else:
                cur_steps_next = self._cur_steps + self.steps_per_call
                truncated = (next_t >= self.p_arr.shape[0] - 1) | (cur_steps_next >= (self.p_arr.shape[0] - 1))
                terminated = self.jnp.zeros((self.num_envs,), dtype=bool)
                info = {"dev_sum": dev_neg_sum + dev_pos_sum, "dev_max": self.jnp.maximum(dev_neg_max, dev_pos_max), "Q_el_kWh": E_el}

            self.t = next_t
            self._cur_steps = self._cur_steps + self.steps_per_call
            self.state = next_state
            if self.diagnostics:
                t_device_end = time.perf_counter()
                try:
                    obs.block_until_ready()
                except Exception:
                    pass
                t_block = time.perf_counter()
        if self.termination_fn is not None:
            terminated, truncated, extra = self.termination_fn(self.state, self.env_params, self.t, info)
            if extra:
                info.update(extra)

        if self.diagnostics:
            info = dict(info)
            info["timing"] = {
                "total_s": t_block - t_start,
                "prep_s": t_prep - t_start,
                "device_s": t_device_end - t_prep,
                "block_s": t_block - t_device_end,
            }
        if self.return_numpy:
            info_np = {}
            for k, v in info.items():
                if isinstance(v, dict):
                    info_np[k] = v
                else:
                    info_np[k] = np.array(v)
            return np.array(obs), np.array(reward), np.array(terminated), np.array(truncated), info_np
        return obs, reward, terminated, truncated, info
