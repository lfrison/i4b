import math
import sys

import pandas as pd
import numpy as np
import copy

from pathlib import Path

# Ensure local i4b root is on the path when running from repo root
I4B_ROOT = Path(__file__).resolve().parents[1]
if str(I4B_ROOT) not in sys.path:
    sys.path.insert(0, str(I4B_ROOT))

import src.disturbances as dist
import src.models.model_buildings as m_bldg
import src.models.model_hvac as m_hvac
import src.controller.heatcurve.heatcurve as hc
import src.simulator as sim

from data.buildings.sfh_1958_1968 import sfh_1958_1968_0_soc

from pathlib import Path


# This dict serves as a demonstration and explanation configuration for the
# below defined method generate_heatcurve_control_dataset().
DEMO_CONFIG = {
    # Define simulation start and end times, the timezone,
    # as well as the geo-location of the simulated settings.
    'start_time': '2015-10-01 00:00:00',
    'end_time': '2016-03-31 23:59:59',
    'timezone': 'Europe/Berlin',
    'location': {
        'latitude': 49.142291,
        'longitude': 9.218655,
    },
    # The building simulation method and the simulation
    # timestep stay the same for all simulated settings.
    'building_simulation_method': '4R3C',
    'simulation_timestep_min': 30,
    # Define the basic simulation setting which is also
    # the initially simulated one. All keys used here
    # are mandatory to be given and filled.
    'base_setting': {
        'building': sfh_1958_1968_0_soc,
        # Refrigerant mass flow:
        'heatpump_mdot': 0.25,
        # Heating curve parameters:
        'heatcurve_t_lim_shift': 5.0,
        'heatcurve_slope': 32.0,
        'heatcurve_offset': 0.0,
        # Target room temperature (deg):
        'target_room_temperature_deg': 20.0,
        # Internal gain profile:
        'internal_gain_profile': './data/profiles/InternalGains/ResidentialDetached.csv',
        # Amplification values for internal, solar and combined gains:
        'internal_gain_amplification': 1.0,
        'solar_gain_amplification': 1.0,
        'gain_amplification': 0.25,
        # Ambient temperature sensor stuck at setting start temperature:
        'ambient_temperature_stuck': False,
        # Fault label for faulty setting changes:
        'faulty': False,
    },
    # Sequentially define all changes that should be done
    # at the specified switch times with respect to the
    # above defined base setting. 'switch_time' is mandatory,
    # all keys used in the base setting can be used here and
    # will override the last value (either from the base setting
    # or a previous setting change).
    'setting_changes': [
        {'switch_time': '2016-01-01 00:00:00', 'heatcurve_offset': 5.0, 'faulty': True},
    ],
}

OUTPUT_FEATURES = ['T_amb', 'ghi', 'T_room', 'T_hp_sup', 'T_hp_ret', ]


def generate_heatcurve_control_dataset(config: dict = DEMO_CONFIG, print_progress: bool = True, apply_setting_changes: bool = True) -> pd.DataFrame:
    '''
    Generates a configurable heating curve control dataset in form of a
    single pandas.DataFrame, as defined in the provided config dict.

    Refer to DEMO_CONFIG for a detailed overview of the configuration
    possibilities. Find examples below in this file.

    Parameters
    ----------
    config: dict, optional
        The configuration dict which specifies the heating curve
        control simulation to be executed. Refer to DEMO_CONFIG
        for a detailed overview of the configuration possibilities.
        Default is DEMO_CONFIG.
    print_progress: bool, optional
        If the simulation progress in percent should be printed
        to the console. Default is True.

    Returns
    -------
    pandas.DataFrame
        The generated heating curve control data.
    list
        The list of setting switch time indices.
    '''

    # Internal function to print the simulation progress to the console
    def prog_print(step: int, num_steps: int) -> None:
        perc = math.floor(float(step * 100) / float(num_steps))
        if ((step % 10) == 0) or (step == num_steps):
            print(f'\rSimulation progress: {perc}%     ', end='')
            if perc == 100:
                print()
            sys.stdout.flush()

    # Load & prepare weather data
    df_weather = dist.load_weather_pvgis(latitude=config['location']['latitude'],
                                         longitude=config['location']['longitude'],
                                         tz=config['timezone'])
    df_weather = df_weather.resample(f'{config["simulation_timestep_min"]}Min').mean()
    if config['simulation_timestep_min'] < 60:
        # If upsampled -> Interpolate NaN gaps
        df_weather = df_weather.interpolate(limit_area='inside')
    df_weather = df_weather[config['start_time'] : config['end_time']]

    # Prepare result dataframe
    pseudo_model_bldg = m_bldg.Building(params=config['base_setting']['building'],
                                        mdot_hp=config['base_setting']['heatpump_mdot'],
                                        method=config['building_simulation_method'])
    pseudo_cost_bldg = pseudo_model_bldg.calc_comfort_dev(T_room=[0,0], timestep=3600)
    pseudo_model_hp = m_hvac.Heatpump_AW(mdot_HP=config['base_setting']['heatpump_mdot'])
    pseudo_cost_hp = pseudo_model_hp.calc(T_hp_ret=[0,0], T_hp_sup=0, T_amb=0, timestep=3600)
    result_columns = ['T_hp_sup'] + list(pseudo_model_bldg.state_keys)
    result_columns += list(pseudo_cost_bldg.keys()) + list(pseudo_cost_hp.keys())
    result_columns += ['Fault']
    df_results = pd.DataFrame(index=df_weather.index, columns=result_columns)
    df_results.loc[df_results.index[-1] + (df_results.index[1] - df_results.index[0])] = [np.NaN] * len(result_columns)

    # Run simulation
    base_setting = copy.deepcopy(config['base_setting'])
    num_settings = len(config['setting_changes']) + 1 if apply_setting_changes else 1
    state = None
    steps_simulated = 0
    switch_times = []
    for i_setting in range(num_settings):
        i_change = i_setting - 1

        # Update setting to changes
        setting = copy.deepcopy(base_setting)
        if apply_setting_changes and i_change >= 0:
            for k in config['setting_changes'][i_change].keys():
                setting[k] = config['setting_changes'][i_change][k]
        setting['building']['T_offset'] = setting['heatcurve_offset']

        # Crop weather data to new setting time
        if num_settings == 1:
            # If only one setting -> No cropping needed
            df_w = df_weather
        else:
            if i_setting == 0:
                # If first setting -> From start until switch time of next setting
                df_w = df_weather[: config['setting_changes'][0]['switch_time']]
            elif i_setting == (num_settings - 1):
                # If last setting -> From last simulated timestep until end
                df_w = df_weather[df_results.index[steps_simulated] :]
            else:
                # Otherwise -> From last simulated timestep until switch time of next setting
                df_w = df_weather[df_results.index[steps_simulated] : config['setting_changes'][i_setting]['switch_time']]

        # Calculate internal gains
        df_int_gains = dist.get_int_gains(time=df_w.index,
                                          profile_path=setting['internal_gain_profile'],
                                          bldg_area=setting['building']['area_floor'])
        df_int_gains['Qdot_tot'] = df_int_gains['Qdot_tot'] * setting['internal_gain_amplification']

        # Calculate solar gains
        df_solar_gains = dist.get_solar_gains(weather=df_w, bldg_params=setting['building'])
        df_solar_gains = df_solar_gains * setting['solar_gain_amplification']

        # Get combined disturbances
        df_qdot_gains = pd.DataFrame((df_solar_gains + df_int_gains['Qdot_tot']), columns=['Qdot_gains'])
        df_qdot_gains['Qdot_gains'] = df_qdot_gains['Qdot_gains'] * setting['gain_amplification']
        df_disturbances = pd.concat([df_w['T_amb'], df_qdot_gains], axis=1)

        # Create models & simulator
        model_bldg = m_bldg.Building(params=setting['building'],
                                     mdot_hp=setting['building']['mdot_hp'],
                                     method=config['building_simulation_method'])
        model_hp = m_hvac.Heatpump_AW(mdot_HP=setting['building']['mdot_hp'])
        simulator = sim.Model_simulator(bldg_model=model_bldg,
                                        hp_model=model_hp,
                                        timestep=(config['simulation_timestep_min'] * 60))

        # Initialize state
        if i_setting == 0:
            state = {key : setting['target_room_temperature_deg'] for key in model_bldg.state_keys}
            for k in state:
                row_label = df_results.index[0]
                df_results.at[row_label, k] = state[k]      
                #df_results[k][0] = state[k]

        # Simulate setting
        heatcurve = hc.Heatingcurve(T_room_set=setting['target_room_temperature_deg'])
        for i_sim_step in range(len(df_disturbances)):
            i_total_step = steps_simulated + i_sim_step

            # Simulation progress
            if print_progress:
                prog_print(i_total_step, (len(df_results) - 2))

            # Disturbances
            d = df_disturbances.iloc[i_sim_step].to_dict()

            # Supply temperature
            t_amb = d['T_amb']
            if setting['ambient_temperature_stuck']:
                t_amb = df_disturbances['T_amb'][0]
            t_sup = heatcurve.calc(t_amb,
                                   shift_T_lim=setting['heatcurve_t_lim_shift'],
                                   offset_T_flow_nom=setting['heatcurve_slope'])[0]
            t_sup += setting['heatcurve_offset']

            # Supply temperature validity check
            t_ret = df_results['T_hp_ret'].iat[i_total_step]
            if d['T_amb'] < setting['building']['T_amb_lim']:
                # If heating limit not exceeded -> Set supply temperature according to heating curve
                t_sup = max(t_sup, t_ret)
            else:
                # Otherwise -> Set supply to return temperature, no heating
                t_sup = t_ret
            t_sup = model_hp.check_hp(t_sup, t_ret)

            # Do one-step integration
            r = simulator.get_next_state(x_init=state, uk=t_sup, pk=d)
            for k in r['state'].keys():
                state[k] = r['state'][k]

            # Save results
            df_results.loc[df_results.index[i_total_step], 'T_hp_sup'] = t_sup
            for k in r['state'].keys():
                df_results.loc[df_results.index[i_total_step+1], k] = r['state'][k]
            for k in r['cost'].keys():
                df_results.loc[df_results.index[i_total_step+1], k] = r['cost'][k]

        # Set fault labels
        df_results.loc[df_results.index[steps_simulated : steps_simulated + len(df_disturbances)], 'Fault'] = int(setting['faulty'])


        # Finalize
        steps_simulated += len(df_disturbances)
        if i_setting < (num_settings - 1):
            switch_times.append(df_results.index[steps_simulated-1])

    df_results = df_results[:-1]
    df_results = pd.concat([df_results, df_weather], axis=1)
    df_results = df_results[['T_amb', 'ghi', 'T_hp_sup', 'T_hp_ret', 'Qdot_th', 'P_el', 'T_room', 'Fault']]
    df_results = df_results.dropna()

    return df_results, switch_times


'''
The code below serves as executable example implementation of the above defined method
generate_heatcurve_control_dataset().
'''


import os
import copy
import matplotlib.pyplot as plt

from data.buildings.sfh_1958_1968 import sfh_1958_1968_0_soc


def _plot_results(df: pd.DataFrame, switch_times: list, title: str = '') -> None:
    fig = plt.figure()
    ax_temp = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2, fig=fig)
    ax_power = plt.subplot2grid(shape=(3, 1), loc=(2, 0), rowspan=1, fig=fig, sharex=ax_temp)

    # Plot scenario changing points
    labeled = False
    for st in switch_times:
        if (st >= df.index[0]) and (st <= df.index[-1]):
            if not labeled:
                ax_temp.axvline(st, label='Setting change', color='gray', linestyle='--', alpha=0.5)
                labeled = True
            else:
                ax_temp.axvline(st, color='gray', linestyle='--', alpha=0.5)
            ax_power.axvline(st, color='gray', linestyle='--', alpha=0.5)

    # Plot fault labels
    fault_start = None
    labeled = False
    for i, l in enumerate(df['Fault']):
        if (fault_start is None) and (l == 1):
            fault_start = df.index[i]
        if (fault_start is not None) and ((l == 0) or (i == (len(df) - 1))):
            if labeled:
                ax_temp.axvspan(fault_start, df.index[i], facecolor='pink', alpha=0.3)
                ax_power.axvspan(fault_start, df.index[i], facecolor='pink', alpha=0.3)
            else:
                ax_temp.axvspan(fault_start, df.index[i], label='Fault', facecolor='pink', alpha=0.3)
                ax_power.axvspan(fault_start, df.index[i], label='Fault', facecolor='pink', alpha=0.3)
                labeled = True
            fault_start = None

    # Plot temperature curves
    ax_temp.plot(df.index, df['T_hp_sup'], label='$T_{sup}$', color='firebrick')
    ax_temp.plot(df.index, df['T_hp_ret'], label='$T_{ret}$', color='royalblue')
    ax_temp.plot(df.index, df['T_room'], label='$T_{room}$', color='darkorange')
    ax_temp.plot(df.index, df['T_amb'], label='$T_{amb}$', color='forestgreen')
    ax_temp.set_ylabel(r'Temperature ($^\circ C$)')
    lines_temp, labels_temp = ax_temp.get_legend_handles_labels()

    # Plot solar irradiation
    sampling_rate_min = int((df.index[1] - df.index[0]).total_seconds() / 60.0)
    bar_width = 0.8 * (sampling_rate_min / (24 * 60))
    ax_si = ax_temp.twinx() 
    ax_si.bar(df.index, df['ghi'], width=bar_width, label='$I_{gh}$', color='gray', alpha=0.25)
    ax_si.set_ylabel(r'Global horizontal irradiance ($\frac{W}{m^2}$)')
    lines_si, labels_si = ax_si.get_legend_handles_labels()
    
    # Plot power curves
    ax_power.plot(df.index, df['Qdot_th'], label='$\dot{Q}_{bldg}$', color='mediumvioletred')
    ax_power.plot(df.index, df['P_el'], label='$P_{el}$', color='mediumturquoise')
    ax_power.set_ylabel(r'Power ($W$)')
    lines_power, labels_power = ax_power.get_legend_handles_labels()

    # Finish figure
    ax_temp.legend((lines_temp + lines_si), (labels_temp + labels_si))
    ax_power.legend(lines_power, labels_power)
    ax_power.set_xlabel('Time')
    fig.suptitle(title)
    fig.tight_layout()
    
    plt.show()


if not os.path.exists('datasets/'):
    os.mkdir('datasets/')


# ------------------------------------------------------------------------
# Fault case 1:
# 5 times window left open for several hours/days.
# ------------------------------------------------------------------------

print('\nSimulate example fault case 1: 5x window left open for several hours/days')

building_base = sfh_1958_1968_0_soc
building_window_open = copy.deepcopy(building_base)
building_window_open['H_tr'] += 150
config = {
    'start_time': '2015-10-01 00:00:00',
    'end_time': '2016-03-31 23:59:59',
    'timezone': 'Europe/Berlin',
    'location': {
        'latitude': 49.142291,
        'longitude': 9.218655,
    },
    'building_simulation_method': '4R3C',
    'simulation_timestep_min': 10,
    'base_setting': {
        'building': building_base,
        'heatpump_mdot': 0.25,
        'heatcurve_t_lim_shift': 5.0,
        'heatcurve_slope': 32.0,
        'heatcurve_offset': 0.0,
        'target_room_temperature_deg': 20.0,
        'internal_gain_profile': './data/profiles/InternalGains/ResidentialDetached.csv',
        'internal_gain_amplification': 1.0,
        'solar_gain_amplification': 1.0,
        'gain_amplification': 0.25,
        'ambient_temperature_stuck': False,
        'faulty': False,
    },
    'setting_changes': [],
}

times_window_open = [
                     # Window left open over working day
                     ('2015-11-02 07:00:00', '2015-11-02 17:00:00'),
                     # Window left open during absence over christmas holiday
                     ('2015-12-23 14:00:00', '2015-12-29 19:00:00'),
                     # Window left open over night
                     ('2016-01-30 22:00:00', '2016-01-31 09:00:00'),
                     # Window left open over night & working day
                     ('2016-02-15 21:00:00', '2016-02-16 17:00:00'),
                     # Window left open over weekend
                     ('2016-03-11 18:00:00', '2016-03-13 17:00:00'),
                    ]

for t in times_window_open:
    o = {'switch_time': t[0], 'building': building_window_open, 'faulty': True}
    c = {'switch_time': t[1], 'building': building_base, 'faulty': False}
    config['setting_changes'].append(o)
    config['setting_changes'].append(c)

'''
df_results, switch_times = generate_heatcurve_control_dataset(config=config)
df_results.to_csv('datasets/window_open_2015-10-01_2016-03-31.csv')

e_electrical = (df_results['P_el'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
e_thermal = (df_results['Qdot_th'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
print(f'-> Electrical energy: {e_electrical:.2f} kWh')
print(f'-> Thermal energy: {e_thermal:.2f} kWh')

_plot_results(df_results, switch_times, '5x window left open for several hours/days')
'''


# ------------------------------------------------------------------------
# Fault case 2:
# 3 times ambient temperature sensor stuck for several hours/days.
# ------------------------------------------------------------------------

print('\nSimulate example fault case 2: 5x ambient temperature sensor stuck for several hours/days')

building_base = sfh_1958_1968_0_soc
config = {
    'start_time': '2015-10-01 00:00:00',
    'end_time': '2016-03-31 23:59:59',
    'timezone': 'Europe/Berlin',
    'location': {
        'latitude': 49.142291,
        'longitude': 9.218655,
    },
    'building_simulation_method': '4R3C',
    'simulation_timestep_min': 10,
    'base_setting': {
        'building': building_base,
        'heatpump_mdot': 0.25,
        'heatcurve_t_lim_shift': 5.0,
        'heatcurve_slope': 32.0,
        'heatcurve_offset': 0.0,
        'target_room_temperature_deg': 20.0,
        'internal_gain_profile': './data/profiles/InternalGains/ResidentialDetached.csv',
        'internal_gain_amplification': 1.0,
        'solar_gain_amplification': 1.0,
        'gain_amplification': 0.25,
        'ambient_temperature_stuck': False,
        'faulty': False,
    },
    'setting_changes': [
        # Ambient temperature sensor stuck for single day
        {'switch_time': '2015-10-20 11:00:00', 'ambient_temperature_stuck': True, 'faulty': True},
        {'switch_time': '2015-10-20 20:00:00', 'ambient_temperature_stuck': False, 'faulty': False},
        # Ambient temperature sensor stuck for day & night
        {'switch_time': '2015-12-04 16:00:00', 'ambient_temperature_stuck': True, 'faulty': True},
        {'switch_time': '2015-12-05 19:00:00', 'ambient_temperature_stuck': False, 'faulty': False},
        # Ambient temperature sensor stuck for several days & nights
        {'switch_time': '2016-01-25 04:00:00', 'ambient_temperature_stuck': True, 'faulty': True},
        {'switch_time': '2016-02-01 13:00:00', 'ambient_temperature_stuck': False, 'faulty': False},
    ],
}

'''
df_results, switch_times = generate_heatcurve_control_dataset(config=config)
df_results.to_csv('datasets/temp-sens_stuck_2015-10-01_2016-03-31.csv')

e_electrical = (df_results['P_el'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
e_thermal = (df_results['Qdot_th'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
print(f'-> Electrical energy: {e_electrical:.2f} kWh')
print(f'-> Thermal energy: {e_thermal:.2f} kWh')

_plot_results(df_results, switch_times, '5x ambient temperature sensor stuck for several hours/days')
'''


# ------------------------------------------------------------------------
# Fault case 3:
# Heating curve parameters slope and offset accidentally reset to default values.
# ------------------------------------------------------------------------

print('\nSimulate example fault case 3: Heating curve parameters slope and offset accidentally reset to default values')

building_base = sfh_1958_1968_0_soc
config = {
    'start_time': '2015-10-01 00:00:00',
    'end_time': '2016-03-31 23:59:59',
    'timezone': 'Europe/Berlin',
    'location': {
        'latitude': 49.142291,
        'longitude': 9.218655,
    },
    'building_simulation_method': '4R3C',
    'simulation_timestep_min': 10,
    'base_setting': {
        'building': building_base,
        'heatpump_mdot': 0.25,
        'heatcurve_t_lim_shift': 5.0,
        'heatcurve_slope': 32.0,
        'heatcurve_offset': 0.0,
        'target_room_temperature_deg': 20.0,
        'internal_gain_profile': './data/profiles/InternalGains/ResidentialDetached.csv',
        'internal_gain_amplification': 1.0,
        'solar_gain_amplification': 1.0,
        'gain_amplification': 0.25,
        'ambient_temperature_stuck': False,
        'faulty': False,
    },
    'setting_changes': [
        {'switch_time': '2016-01-18 00:00:00', 'heatcurve_slope': 25.0, 'heatcurve_offset': 5.0, 'faulty': True},
    ],
}

'''
df_results, switch_times = generate_heatcurve_control_dataset(config=config)
df_results.to_csv('datasets/hc-params_reset_2015-10-01_2016-03-31.csv')

e_electrical = (df_results['P_el'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
e_thermal = (df_results['Qdot_th'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
print(f'-> Electrical energy: {e_electrical:.2f} kWh')
print(f'-> Thermal energy: {e_thermal:.2f} kWh')

_plot_results(df_results, switch_times, 'Heating curve parameters slope and offset accidentally reset to default values')
'''


# ------------------------------------------------------------------------
# Nominal data without faults
# ------------------------------------------------------------------------

print('\nSimulate nominal data without faults')

building_base = sfh_1958_1968_0_soc
config = {
    'start_time': '2014-07-01 00:00:00',
    'end_time': '2015-06-30 23:59:59',
    'timezone': 'Europe/Berlin',
    'location': {
        'latitude': 49.142291,
        'longitude': 9.218655,
    },
    'building_simulation_method': '4R3C',
    'simulation_timestep_min': 10,
    'base_setting': {
        'building': building_base,
        'heatpump_mdot': 0.25,
        'heatcurve_t_lim_shift': 5.0,
        'heatcurve_slope': 32.0,
        'heatcurve_offset': 0.0,
        'target_room_temperature_deg': 20.0,
        'internal_gain_profile': './data/profiles/InternalGains/ResidentialDetached.csv',
        'internal_gain_amplification': 1.0,
        'solar_gain_amplification': 1.0,
        'gain_amplification': 0.25,
        'ambient_temperature_stuck': False,
        'faulty': False,
    },
    'setting_changes': [],
}

df_results, _ = generate_heatcurve_control_dataset(config=config)
df_results.to_csv('datasets/nominal_2014-07-01_2015-06-30.csv')

e_electrical = (df_results['P_el'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
e_thermal = (df_results['Qdot_th'] * (config['simulation_timestep_min'] / 60)).sum() / 1000.0
print(f'-> Electrical energy: {e_electrical:.2f} kWh')
print(f'-> Thermal energy: {e_thermal:.2f} kWh')
