"""MPC simulation platform for building HVAC control.

This script implements a Model Predictive Control (MPC) simulation:
    1. Initialize problem definition and MPC setup
    2. Read input parameters
    3. Loop over all time steps:
        - Retrieve last output values y from plant
        - Call MPC.get_next_step(y)
        - Apply first control out of the sequence
        - Simulate system dynamics for next time step x = x_next
"""
import argparse
import importlib
import sys
import warnings
from argparse import BooleanOptionalAction
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure the root of your package is in the PYTHONPATH
root_path = str(Path(__file__).resolve().parent)
sys.path.insert(0, root_path)

import src.disturbances as disturbances
import src.models.model_buildings as model_buildings
import src.models.model_hvac as model_hvac
import src.simulator as simulator
from src.controller.mpc.casadi_framework import MPC_solver
import src.controller.mpc.mpc_utility as util


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for MPC simulation."""
    parser = argparse.ArgumentParser(
        description="MPC simulation platform for building HVAC"
    )
    parser.add_argument(
        "--h", type=int, default=900,
        help="Timestep in seconds"
    )
    parser.add_argument(
        "--days", type=int, default=394,
        help="Number of simulation days"
    )
    parser.add_argument(
        "--nk_hours", type=int, default=24,
        help="Prediction horizon in hours"
    )
    parser.add_argument(
        "--step_length", type=int, default=1,
        help="MPC iteration step length in steps"
    )
    parser.add_argument(
        "--method", type=str, choices=["2R2C", "4R3C", "5R4C"],
        default="4R3C", help="Building thermal model"
    )
    parser.add_argument(
        "--mdot_hp", type=float, default=0.25,
        help="Heat pump massflow [kg/s]"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Std dev of Gaussian noise for observations/disturbances"
    )
    parser.add_argument(
        "--offset_days", type=int, default=0,
        help="Offset days for starting point in weather/gains"
    )
    parser.add_argument(
        "--grid_on", action=BooleanOptionalAction, default=False,
        help="Enable grid supportive operation"
    )
    parser.add_argument(
        "--simulator_on", action=BooleanOptionalAction, default=True,
        help="Use simulator for next state (slower)"
    )
    parser.add_argument(
        "--hp_type", type=str, choices=["AW", "Vitocal"], default="AW",
        help="Heat pump type"
    )
    parser.add_argument(
        "--building", type=str,
        default="sfh_1958_1968_0_soc",
        help="Building name (e.g., 'sfh_1958_1968_0_soc', 'i4c')"
    )
    parser.add_argument(
        "--t_lower_night", type=float, default=20.0,
        help="Night setback lower room setpoint [C]"
    )
    parser.add_argument(
        "--t_lower_day", type=float, default=20.0,
        help="Day lower room setpoint [C]"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for noise"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results_mpc",
        help="Directory to store results"
    )
    return parser.parse_args()


def select_building(building_name: str):
    """Load building parameters from data.buildings module.
    
    Parameters
    ----------
    building_name : str
        Building name (e.g., 'sfh_1958_1968_0_soc', 'i4c')
        
    Returns
    -------
    dict
        Building parameters dictionary
        
    Raises
    ------
    AttributeError
        If building does not exist
        
    Notes
    -----
    Available buildings include:
    - i4c
    - sfh_XXXX_YYYY_Z_type where:
      - XXXX_YYYY: construction period
      - Z: 0 (soc), 1 (enev), 2 (kfw)
    """
    import data.buildings
    
    # Support legacy format "module:attribute" for backward compatibility
    if ":" in building_name:
        _, building_name = building_name.split(":", 1)
    
    if not hasattr(data.buildings, building_name):
        available = sorted([b for b in data.buildings.__all__ 
                          if not b.startswith('_')])
        raise AttributeError(
            f"Building '{building_name}' not found.\n"
            f"Available buildings: {', '.join(available[:5])}... "
            f"(total: {len(available)})"
        )
    
    return getattr(data.buildings, building_name)


def get_column_names(method: str, include_noisy: bool = False) -> list:
    """Get CSV column names based on building model method.
    
    Parameters
    ----------
    method : str
        Building thermal model type ('2R2C', '4R3C', or '5R4C')
    include_noisy : bool
        If True, include noisy observation columns
        
    Returns
    -------
    list
        Column names for the results CSV file
    """
    # True state columns
    base_cols = ['time', 'T_room']
    if method == '5R4C':
        state_cols = ['T_int', 'T_wall']
    elif method == '4R3C':
        state_cols = ['T_wall']
    elif method == '2R2C':
        state_cols = []
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    true_state_cols = base_cols + state_cols + ['T_return']
    
    # Noisy observation columns (what MPC sees)
    noisy_cols = []
    if include_noisy:
        noisy_cols = ['T_room_obs']
        if method == '5R4C':
            noisy_cols += ['T_int_obs', 'T_wall_obs']
        elif method == '4R3C':
            noisy_cols += ['T_wall_obs']
        noisy_cols += ['T_return_obs']
    
    # Control and disturbances
    remaining_cols = [
        'T_HP', 'T_amb', 'Qdot_gains',
        'T_room_set_lower', 'grid_signal', 'P_el_kWh'
    ]
    
    return true_state_cols + noisy_cols + remaining_cols


def compute_power_consumption(uk, xk, P, hp_model, h):
    """Compute electrical power consumption for current timestep.
    
    Parameters
    ----------
    uk : array
        Control input (T_HP)
    xk : array
        State vector
    P : array
        Parameter matrix
    hp_model : Heatpump
        Heat pump model instance
    h : int
        Timestep in seconds
        
    Returns
    -------
    float
        Electrical energy consumption in kWh
    """
    T_HP_current = uk[0]
    T_return_current = xk[-1]
    T_amb_current = P[0, 0]
    
    COP = hp_model.COP(T_HP_current, T_amb_current)
    Qth_kW = (hp_model.mdot_HP * hp_model.c_water *
              (T_HP_current - T_return_current) / 1000)
    P_el_kW = Qth_kW / COP if COP > 0 else 0
    E_el_kWh = P_el_kW * h / 3600
    
    return E_el_kWh


def add_noise_to_disturbances(P, noise_std):
    """Add Gaussian noise to disturbances (T_amb and Qdot_gains).
    
    Parameters
    ----------
    P : ndarray
        Parameter matrix
    noise_std : float
        Standard deviation of Gaussian noise
        
    Returns
    -------
    ndarray
        Noisy parameter matrix
    """
    P_noisy = P.copy()
    P_noisy[:, 0] += np.random.normal(0, noise_std, size=P_noisy.shape[0])
    P_noisy[:, 1] += np.random.normal(0, noise_std, size=P_noisy.shape[0])
    return P_noisy


def main() -> None:
    """Main MPC simulation loop."""
    args = parse_args()

    # Derived values
    h = args.h
    nk = args.nk_hours * int(3600 / h)
    mpc_steps = args.days * 24 * int(3600 / h)

    # Random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize the building model
    building_params = select_building(args.building)
    method = args.method
    mdot_hp = args.mdot_hp

    building_model = model_buildings.Building(
        params=building_params,
        mdot_hp=mdot_hp,
        method=method,
        T_room_set_lower=args.t_lower_day,
        T_room_set_upper=26
    )

    # Heat pump model
    if args.hp_type == "Vitocal":
        hp_model = model_hvac.Heatpump_Vitocal(mdot_HP=mdot_hp)
    else:
        hp_model = model_hvac.Heatpump_AW(mdot_HP=mdot_hp)

    # Simulator
    sim_env = None
    if args.simulator_on:
        sim_env = simulator.Model_simulator(hp_model, building_model, h)

    # MPC problem dimension parameters
    npar = 4
    nc = 4
    ns = 2
    nx_map = {'2R2C': 2, '4R3C': 3, '5R4C': 4}
    nx = nx_map.get(method)
    if nx is None:
        raise ValueError(f"Unsupported method: {method}")

    # Set-up result file
    include_noisy = args.noise > 0
    columns = get_column_names(method, include_noisy=include_noisy)
    resultdir = args.results_dir
    resultfile = (
        f"results_{building_params['name']}_"
        f"days{int(mpc_steps/(24*int(3600/h)))}_"
        f"prediction{int(nk/int(3600/h))}h_"
        f"h{h}_noise_{args.noise:f}_seed{args.seed}"
    )
    util.init_file(columns, resultdir, resultfile)

    # Define MPC solver
    mpc = MPC_solver(
        resultdir, resultfile, hp_model, building_model,
        nx=nx, npar=npar, h=h, nk=nk, nc=nc, ns=ns, ws=0.1
    )

    # Load weather and disturbances
    pos = building_params['position']
    weather_df = disturbances.load_weather(
        latitude=pos['lat'],
        longitude=pos['long'],
        altitude=pos['altitude']
    )[0:8760]

    # Internal gains
    int_gains_df = disturbances.get_int_gains(
        time=weather_df.index,
        profile_path='data/profiles/InternalGains/ResidentialDetached.csv',
        bldg_area=building_params['area_floor']
    )

    # Solar and total gains
    Qdot_sol = disturbances.get_solar_gains(
        weather=weather_df,
        bldg_params=building_params
    )
    Qdot_gains = pd.DataFrame(
        Qdot_sol / 2 + int_gains_df['Qdot_tot'] / 100,
        columns=['Qdot_gains']
    )
    p_hourly = pd.concat([weather_df['T_amb'], Qdot_gains], axis=1)

    # Grid signal
    data_grid = np.ones(Qdot_sol.shape[0])
    if args.grid_on:
        grid_df = pd.read_csv(
            'data/grid/grid_signals.csv',
            sep=',',
            header='infer'
        )
        data_grid = grid_df['EEX2015'].values * 0.001 * 5

    # Lower setpoint profile (24h template)
    T_lower = np.ones(24) * building_model.T_room_set_lower
    T_lower[:7] = args.t_lower_night
    T_lower[21:] = args.t_lower_night
    n_hours = Qdot_sol.shape[0]
    n_days = int(np.ceil(n_hours / 24))
    p_hourly['T_room_set_lower'] = np.tile(
        T_lower,
        n_days
    )[:n_hours]
    p_hourly['grid'] = data_grid[:n_hours]

    # Offset days
    if args.offset_days > 0:
        p_hourly = p_hourly[args.offset_days * 24:]

    # Resample to control step
    p = p_hourly.resample(f'{h}S').ffill()

    # Initial state vector (matching gym environment: 20°C for all temperatures)
    xk_dict = {key: 20.0 for key in building_model.state_keys}
    xk_next = np.array(list(xk_dict.values()))

    # MPC iteration loop
    start = timer()
    for i in range(mpc_steps):
        # Extract prediction horizon parameters
        idx_start = i * args.step_length
        idx_end = idx_start + mpc.nk + 1
        P = p[idx_start:idx_end].values

        # Add noise to disturbances if specified
        xk_obs = None  # Noisy observation (what MPC sees)
        if args.noise > 0:
            P_noisy = add_noise_to_disturbances(P, args.noise)
            xk_mpc = xk_next + np.random.normal(
                0, args.noise, size=xk_next.shape
            )
            xk_obs = xk_mpc.copy()  # Store noisy observation for logging
        else:
            P_noisy = P
            xk_mpc = xk_next

        # Solve MPC optimization problem
        xk = xk_next  # Store true previous state
        mpc.update_NLP(xk_mpc)
        uk, xk_next = mpc.solve_NLP(P_noisy, mpc_steps <= 1)

        # Update state using simulator if enabled
        if args.simulator_on and sim_env is not None:
            pk = p.iloc[i].to_dict()
            results = sim_env.get_next_state(
                x_init=xk_dict, uk=uk, pk=pk
            )
            xk_dict = results['state']
            xk_next = np.array(list(xk_dict.values()))

        # Compute and log power consumption
        E_el_kWh = compute_power_consumption(uk, xk, P, hp_model, h)
        
        # Log both true state and noisy observation (if noise enabled)
        util.update_file(
            i * h, mpc.resultfile, mpc.resultdir, P, uk, xk, E_el_kWh, xk_obs=xk_obs
        )

    # Print results
    elapsed_time = timer() - start
    print(f'Elapsed time is {elapsed_time:.2f}s.')

    if mpc_steps >= 24:
        util.evaluate_mpc(
            hp_model, building_model,
            mpc.resultfile, mpc.resultdir, h=h
        )


if __name__ == "__main__":
    main()
