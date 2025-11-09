'''
MPC simulation platform

initialization routines
    problem definition
    MPC setup
    read input parameters
loop over all time steps:
    retrieving last output values y from plant
    calling MPC.get_next_step(y)
    applying first control out of the sequence.
    simulation of system dynamics for next time step x = x_next
'''
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import argparse
from argparse import BooleanOptionalAction

# Ensure the root of your package is in the PYTHONPATH
root_path = str(Path(__file__).resolve().parent)
sys.path.insert(0, root_path)

from src.controller.mpc.casadi_framework import MPC_solver            
import src.controller.mpc.mpc_utility as util
from timeit import default_timer as timer

import src.simulator as simulator
import src.models.model_hvac as model_hvac
import src.models.model_buildings as model_buildings
import src.disturbances as disturbances
import importlib
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPC simulation platform for building HVAC")
    parser.add_argument("--h", type=int, default=900, help="Timestep in seconds")
    parser.add_argument("--days", type=int, default=394, help="Number of simulation days")
    parser.add_argument("--nk_hours", type=int, default=24, help="Prediction horizon in hours")
    parser.add_argument("--step_length", type=int, default=1, help="MPC iteration step length in steps")
    parser.add_argument("--method", type=str, choices=["2R2C","4R3C","5R4C"], default="4R3C", help="Building thermal model")
    parser.add_argument("--mdot_hp", type=float, default=0.25, help="Heat pump massflow [kg/s]")
    parser.add_argument("--noise", type=float, default=0.0, help="Std dev of Gaussian noise for observations/disturbances")
    parser.add_argument("--offset_days", type=int, default=0, help="Offset days for starting point in weather/gains")
    parser.add_argument("--grid_on", action=BooleanOptionalAction, default=False, help="Enable grid supportive operation")
    parser.add_argument("--simulator_on", action=BooleanOptionalAction, default=True, help="Use simulator for next state (slower)")
    parser.add_argument("--hp_type", type=str, choices=["AW","Vitocal"], default="AW", help="Heat pump type")
    parser.add_argument("--building", type=str, default="sfh_1958_1968:sfh_1958_1968_0_soc", help="Building as 'module:attribute' under data.buildings, e.g., 'sfh_1984_1994:sfh_1984_1994_1_enev'")
    parser.add_argument("--t_lower_night", type=float, default=18.0, help="Night setback lower room setpoint [C]")
    parser.add_argument("--t_lower_day", type=float, default=20.0, help="Day lower room setpoint [C]")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise")
    parser.add_argument("--results_dir", type=str, default="results_mpc", help="Directory to store results")
    return parser.parse_args()


def select_building(building_key: str):
    if ":" not in building_key:
        raise ValueError("--building must be provided as 'module:attribute' under data.buildings, e.g., 'sfh_1958_1968:sfh_1958_1968_0_soc'")
    module_name, attr_name = building_key.split(":", 1)
    module = importlib.import_module(f"data.buildings.{module_name}")
    if not hasattr(module, attr_name):
        raise AttributeError(f"Module data.buildings.{module_name} has no attribute {attr_name}")
    return getattr(module, attr_name)


def main() -> None:
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

    building_model = model_buildings.Building(params = building_params,
                                              mdot_hp = mdot_hp,
                                              method = method,
                                              T_room_set_lower = args.t_lower_day,
                                              T_room_set_upper = 26)

    # Heat pump model
    if args.hp_type == "Vitocal":
        hp_model = model_hvac.Heatpump_Vitocal(mdot_HP = mdot_hp)
    else:
        hp_model = model_hvac.Heatpump_AW(mdot_HP = mdot_hp)

    # Simulator
    if args.simulator_on:
        sim_env = simulator.Model_simulator(hp_model, building_model, h)
    else:
        sim_env = None

    # MPC problem dimension parameters
    npar = 4
    nc = 4
    ns = 2
    if method == '2R2C':
        nx = 2
        columns=['time','T_room','T_return','T_HP','T_amb','Qdot_gains','T_room_set_lower','grid_signal']
    elif method == '4R3C':
        nx = 3
        columns=['time','T_room','T_wall','T_return','T_HP','T_amb','Qdot_gains','T_room_set_lower','grid_signal']
    elif method == '5R4C':
        nx = 4
        columns=['time','T_room','T_int','T_wall','T_return','T_HP','T_amb','Qdot_gains','T_room_set_lower','grid_signal']
    else:
        raise ValueError("Unsupported method")

    # Set-up result file
    resultdir = args.results_dir
    resultfile = 'results_%s_days%d_prediction%dh_h%d_noise_%f' % (
        building_params['name'],
        mpc_steps/(24*int(3600/h)),
        nk/int(3600/h),
        h,
        args.noise
    )
    util.init_file(columns,resultdir,resultfile)

    # define MPC solver
    mpc = MPC_solver(resultdir, resultfile, hp_model, building_model, nx=nx, npar=npar, h=h, nk=nk, nc=nc, ns=ns)

    # DISTURBANCES
    pos = building_params['position']
    weather_df = disturbances.load_weather(latitude = pos['lat'],
                                           longitude = pos['long'],
                                           altitude  = pos['altitude'])[0:8760]

    int_gains_df = disturbances.get_int_gains(time = weather_df.index,
                                              profile_path = 'data/profiles/InternalGains/ResidentialDetached.csv',
                                              bldg_area = building_params['area_floor'] )

    Qdot_sol = disturbances.get_solar_gains(weather = weather_df, bldg_params = building_params)

    Qdot_gains = pd.DataFrame(Qdot_sol/2 + int_gains_df['Qdot_tot']/100 , columns = ['Qdot_gains'])
    p_hourly = pd.concat([weather_df['T_amb'], Qdot_gains], axis = 1)

    # Grid signal
    data_grid = np.ones(Qdot_sol.shape[0])
    if args.grid_on:
        data_grid = pd.read_csv('data\grid\grid_signals.csv',sep=',', header='infer')['EEX2015'].values*0.001*5
    _ = np.ones(Qdot_sol.shape[0])

    # Lower setpoint profile (24h template)
    T_lower = np.ones(24)*building_model.T_room_set_lower
    T_lower[:7] = args.t_lower_night
    T_lower[21:] = args.t_lower_night
    p_hourly['T_room_set_lower'] = np.reshape([T_lower]*(int(np.ceil(Qdot_sol.shape[0]/24))),int(np.ceil(Qdot_sol.shape[0]/24)*24))
    p_hourly['grid'] = data_grid[:Qdot_sol.shape[0]]

    # Offset days
    if args.offset_days > 0:
        p_hourly = p_hourly[args.offset_days*24:]

    # Resample to control step
    p = p_hourly.resample(f'{h}S').ffill()

    # Initial state vector
    xk_dict = {key : 18 for key in building_model.state_keys}
    if method != '2R2C':
        xk_dict['T_wall'] = 15
    xk_dict['T_hp_ret'] = 22
    xk_next = np.array(list(xk_dict.values()))

    # Iterate MPC
    start = timer()
    for i in range(mpc_steps):
        P = p[i*args.step_length:i*args.step_length+mpc.nk+1].values

        # Noise on disturbances (T_amb, Qdot_gains) only
        if args.noise > 0:
            P_noisy = P.copy()
            P_noisy[:, 0] = P_noisy[:, 0] + np.random.normal(0, args.noise, size=P_noisy.shape[0])
            P_noisy[:, 1] = P_noisy[:, 1] + np.random.normal(0, args.noise, size=P_noisy.shape[0])
        else:
            P_noisy = P

        xk = xk_next
        if args.noise > 0:
            xk_mpc = xk + np.random.normal(0, args.noise, size=xk.shape)
        else:
            xk_mpc = xk

        mpc.update_NLP(xk_mpc)
        uk, xk_next = mpc.solve_NLP(P_noisy, mpc_steps<=1)

        if args.simulator_on and sim_env is not None:
            pk = p.iloc[i].to_dict()
            results = sim_env.get_next_state(x_init = xk_dict, uk = uk, pk = pk)
            xk_dict = results['state']
            xk_next = np.array(list(xk_dict.values()))

        util.update_file(i*h, mpc.resultfile, mpc.resultdir, P, uk, xk)

    print ('Elapsed time is %.2fs.' %(timer() - start))
    if mpc_steps>=24:
        util.evaluate_mpc(hp_model, building_model, mpc.resultfile, mpc.resultdir, h=h)


if __name__ == "__main__":
    main()

