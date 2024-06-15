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

# Ensure the root of your package is in the PYTHONPATH
root_path = str(Path(__file__).resolve().parent)
print(root_path)
sys.path.insert(0, root_path)

from src.controller.mpc.casadi_framework import MPC_solver            
import src.controller.mpc.mpc_utility as util
from timeit import default_timer as timer

import src.simulator as simulator
import src.models.model_hvac as model_hvac
import src.models.model_buildings as model_buildings
from data.buildings.sfh_1958_1968 import sfh_1958_1968_0_soc # load example building data
from data.buildings.sfh_1984_1994 import sfh_1984_1994_1_enev # load example building data
from data.buildings.i4c_building import i4c # load example building data
import src.disturbances as disturbances

SIMULATOR_ON = True #False # flag whether simulation should be used to obtain next state (slower)
# GRID_ON = True  # flag whether grid supportive operation
GRID_ON = False # standard (energy efficient) operation (grid signal constant 1)

h = 3600  # length time step in s
#h = 900  # length time step in s
offset_days = 0  # in days

# Initialize the building model
mdot_hp = 0.25 # Massflow of the heat supply system. [kg/s]
method    = '4R3C' # model degree
#building = sfh_1958_1968_0_soc  # choose building
building = sfh_1984_1994_1_enev 
# building = i4c

# may change important parameters of building 
#building['c_bldg']=10 # thermal capacity
#building['H_tr']*=1 # heating loss of building envelope 

building_model = model_buildings.Building(params    = building, # More example buildings can be found in data/buildings/.
                                           mdot_hp   = mdot_hp,       # Massflow of the heat supply system. [kg/s]
                                           method    = method,        # For available calculation methods see: model_buildings.py.
                                           T_room_set_lower = 20,     # lower and upper comfort temperature
                                           T_room_set_upper = 26)
# additional building model (e.g. for mpc controller)
#building_model_mpc=building_model.copy()
#building_model_mpc.method='2R2C'


# Create HP model (uncomment selected heat pump)
#hp_model = model_hvac.Heatpump_Vitocal(mdot_HP = mdot_hp) # ground source heat pump
hp_model = model_hvac.Heatpump_AW(mdot_HP = mdot_hp) # air water heat pump

# flag whether to use simulation (default true) or mpc computation
if SIMULATOR_ON: 
   # define simulation environment with simulation model
   simulator = simulator.Model_simulator(hp_model, building_model, h)


# MPC settings
mpc_steps = 1#*24*int(3600/h) # total number of MPC iteration steps 
nk = 24*int(3600/h) # optimization/forecasting horizon (number of steps), default 1 day= 24*int(3600/h)
step_length = 1 #24*int(3600/h) # step length of mpc iteration (nk>=step_length), default 1
ws = .1 # weighting factor for temperature slack variables (balances comfort deviation vs. cost)

      
# MPC problem dimension parameters
npar = 4 # number of external parameters (time series)
nc = 4 # number of optimization constraints 
ns = 2 # number of slack variables
if method == '2R2C':
    nx = 2 # number of states
    columns=['time','T_room','T_return','T_HP','T_amb','Qdot_gains','T_room_set_lower','grid_signal'] # column names for result file (time,x,u,P)
elif method == '4R3C':
    nx = 3
    columns=['time','T_room','T_wall','T_return','T_HP','T_amb','Qdot_gains','T_room_set_lower','grid_signal'] # (time,x,u,P)
elif method == '5R4C':
    nx = 4
    columns=['time','T_room','T_int','T_wall','T_return','T_HP','T_amb','Qdot_gains','T_room_set_lower','grid_signal'] # (time,x,u,P)


# Set-up result file
resultdir = "results_mpc"
resultfile = 'results_%s_days%d_prediction%dh_h%d'%(building['name'],mpc_steps/(24*int(3600/h)),nk/int(3600/h),h)  # result file
util.init_file(columns,resultdir,resultfile)

# define MPC solver
mpc = MPC_solver(resultdir, resultfile, hp_model, building_model,nx=nx,npar=npar,h=h,nk=nk,nc=nc,ns=ns) 


# DISTURBANCES
# Extract building location from building parameters
pos = building['position']

# Load weather data as pandas df
weather_df = disturbances.load_weather(latitude = pos['lat'],
                              longitude = pos['long'],
                              altitude  = pos['altitude'])[0:8760]

# Generate absolut heat gain profiles, based on datetime, building usage and floor area. 
int_gains_df = disturbances.get_int_gains(time = weather_df.index,
                                profile_path = 'data/profiles/InternalGains/ResidentialDetached.csv',
                                bldg_area = building['area_floor'] )
#print(int_gains_df['Qdot_tot'][:24])

# Generate profiles of solar heat gains, based on datetime and irradiance data in the weather df, 
# and the window properties defined in the building parameters.
Qdot_sol = disturbances.get_solar_gains(weather = weather_df, bldg_params = building)

# Combine all disturbances (solar gain, internal gains) into one disturbance dataframe:
#int_gains_df['Qdot_tot'] = 0
Qdot_gains = pd.DataFrame(Qdot_sol/2 + int_gains_df['Qdot_tot']/100 , columns = ['Qdot_gains']) # calculate total gains
#print(Qdot_gains[:24])
# Concatenate parameter vector
p_hourly = pd.concat([weather_df['T_amb'], Qdot_gains], axis = 1) # create new df
#p_hourly['Qdot_gains'] = 0

# Define the grid signal
data_grid = np.ones(Qdot_sol.shape[0])
# read grid data
if GRID_ON: 
   data_grid = pd.read_csv('data\grid\grid_signals.csv',sep=',',
                           header='infer')['EEX2015'].values*0.001*5 # EUR/kWh
grid_signal = np.ones(Qdot_sol.shape[0])

# Set a lower set point temperature (e.g. night setback)
T_lower = np.ones(24)*building_model.T_room_set_lower
T_lower[:7] = 18
T_lower[21:] = 18
#p_hourly['T_amb'] = np.ones(Qdot_sol.shape[0])
p_hourly['T_room_set_lower'] = np.reshape([T_lower]*(int(np.ceil(Qdot_sol.shape[0]/24))),int(np.ceil(Qdot_sol.shape[0]/24)*24))
p_hourly['grid'] = data_grid[:Qdot_sol.shape[0]]

# set offset days in order to shift some days (default 0 offset)
p_hourly=p_hourly[offset_days*24:]

# resample parameter vector to new frequency
p = p_hourly.resample(f'{h}S').ffill() # resample disturbance

# example to set up an interestin scenario (24h)
#p.iloc[6:8,-1]-=.1


# Setup initial state vector
xk_dict = {key : 18 for key in building_model.state_keys} # in degC
if method != '2R2C': xk_dict['T_wall'] = 15
xk_dict['T_hp_ret'] = 22
xk_next = np.array(list(xk_dict.values())) 

# MPC iteration
start = timer()
for i in range(mpc_steps):  
   
   P = p[i*step_length:i*step_length+mpc.nk+1].values # 24 time step forecasts of [ambient temperature, heat gains]
   
   # extract initial values from dict and create np.array as required for mpc
   xk = xk_next #np.array(list(xk_dict.values())) 
   #print('Iteration',i,xk)

   # run optimizer
   mpc.update_NLP(xk)
   uk, xk_next = mpc.solve_NLP(P,mpc_steps<=1)

    
   pk = p.iloc[i].to_dict() # get disturbances of current step
   # run simulator
   if SIMULATOR_ON:
      pk = p.iloc[i].to_dict() # get disturbances of current step
      results = simulator.get_next_state(x_init = xk_dict, uk = uk, pk = pk)
      xk_dict = results['state'] # save next state
      xk_next = np.array(list(xk_dict.values())) 

   ## Add current step to file
   util.update_file(i*h,mpc.resultfile,mpc.resultdir,P,uk,xk)
   
        
print ('Elapsed time is %.2fs.' %(timer() - start))
if mpc_steps>=24: util.evaluate_mpc(hp_model,building_model,mpc.resultfile,mpc.resultdir,h=h) 
      
