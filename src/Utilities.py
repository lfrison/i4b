# -*- coding: utf-8 -*-
"""
Utility functions for plotting and data handling in the MPC/simulation workflow.
"""

import os
from importlib import import_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify material parameters

RHO_WATER = 997      # Density water [kg/m3]
RHO_AIR = 1.225      # Density air [kg/m3]
C_WATER_SPEC = 4181  # Spec. heat capacity of water [J/kg/K]
C_AIR_SPEC = 1.005   # Spec. heat capacity of air [J/kg/K]


def evaluate_results(x_arr, T_HP, T_amb, Q_HP, h=3600):   
    """Plot key simulation results.

    Parameters
    ----------
    x_arr : numpy.ndarray
        2D array of the results with room and return temperatures.
        Ensure the time dimension matches the other inputs (usually x_arr[:-1]).
    T_HP : numpy.ndarray
        supply flow temperature [°C]
    T_amb : numpy.ndarray
        ambient temperature [°C]
    Q_HP : numpy.ndarray
        thermal heat pump power [kW]
    h : int, optional
        time_steps in [sec]. The default is 3600.

    Returns
    -------
    None

    """
    num_steps = x_arr.shape[0]
    time =  np.linspace(0,(num_steps*h)-h,num_steps)
    
    fig, ax = plt.subplots(1, 1,figsize=(12,4))
    ax.plot(time,T_HP,label='T_HP', alpha=0.6,color='green')
    ax.plot(time,x_arr[:,-1],label='T_return', alpha=0.6,color='blue')
    ax.plot(time,x_arr[:,0],label='T_room', alpha=0.6,color='red')
    ax.plot(time,T_amb[:num_steps],label='T_amb', alpha=0.6,color='k')
    ax.set_ylabel('temperature [°C]')
    
    axtwin = ax.twinx()
    axtwin.plot(time,Q_HP,label='Qth_HP', alpha=0.6,color='darkred')
    axtwin.set_ylabel('kW')
    #ax.grid()
    ax.set_xlabel('time steps')
    ax.legend()   
    axtwin.legend() 
    plot(fig,ax,plt,tf=h*num_steps)
  
   
def plot_timeconstant(temp, temp_start, temp_end, tau, step_size):
   """Plot the time constant illustrating dynamic behavior."""
   num_steps = temp.shape[0]
   time = np.linspace(0,step_size*(num_steps-1)/3600, num_steps) # in hours
   temp_tau = (1-0.632)*temp_start
   
   fig, ax = plt.subplots(1, 1,figsize=(12,4))
   ax.plot(time,temp[:],label='T_room', alpha=0.6,color='blue')
   ax.plot(time,np.append(temp_end,temp_end[-1]),label='T_amb', alpha=0.6,color='green')
   ax.plot([0,tau/3600],[temp_tau,temp_tau], color='gray', linestyle='dashed') # tau - vertical line
   ax.plot([tau/3600,tau/3600],[temp_end[0], temp_tau], color='gray', linestyle='dashed' ) # tempstart - 63.2% - horizontal line
   ax.text(tau/3600, temp_tau + 0.5, r'$\tau$ = {:.2f} h'.format(tau/3600))
   ax.annotate("", xy = (0,temp_tau), xytext = (0,temp_start), arrowprops = dict(arrowstyle="<|-|>", shrinkA = 0, shrinkB = 0, linestyle = '--'))
   ax.text(-5, (temp_tau + temp_start) / 2, r'$T_{start} - 63.2 \%$',  rotation=90, ha='center', va='center')
   #ax.set_xticks([tau/3600, 3*tau/3600, 5*tau/3600],[r'\tau', r'3\tau', r'\5tau'])
   ax.set_ylabel('temperature [°C]')
   ax.grid()
   ax.set_xlabel('time [h]')
   ax.legend(loc="upper right")   
   
def plot_monthly_results(heatdemand):
   fig, ax = plt.subplots(1, figsize=(12,4))
   months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
   plt.plot(months, heatdemand.groupby([heatdemand.index.month]).sum(),label = 'Heating demand', c = 'red')
   
   # title and legend
   legend_label = ['Heating Demand', 'Transmission losses', 'Ventilation losses', 'Internal gains', 'Solar gains']
   plt.legend(legend_label, ncol = 5, bbox_to_anchor=([1, 1.05, 0, 0]), frameon = False)
   plt.title('Monthly sums\n', loc='left')
   plt.show()
      
def read_weather_data(input_file, h):
   """Read weather data (CSV) and resample to the given step size."""

   df_data = pd.read_csv('data\\%s'%(input_file+'.csv'), sep=',',header=0,index_col=0)

   # Optional: time filtering/resampling can be added here if needed
   df_data = df_data.fillna(0)
   df_data = df_data.iloc[::int(h/900)] # resampling
       
   return df_data['T_amb']


def plot(fig, ax, plt, tf):
   if tf > 86400:
      step=86400
      ax.set_xlabel('time [d]')
   else:
      step=3600
      ax.set_xlabel('time [h]')
   #ax.legend();
   ax.set_xlim([0,tf])
   n = int(tf/step)
   plt.xticks([scale*step for scale in range(n+1)], ['%i' % scale for scale in range(n+1)])
   plt.show()   
     

class LocalColor:
   def __init__(self, c='green', a=0.5):
       self.c = c
       self.a = max(0.0, min(1.0, a))

       if self.c=='green':
           self.c = (177./256., 200./256., 0./256., self.a)
           self.ci = (177, 200, 0)
           self.s = "#%02X%02X%02X" % self.ci
       elif self.c=='grey':
           self.c = (168./256., 175./256., 175./256., self.a)
           self.ci = (168, 175, 175)
           self.s = "#%02X%02X%02X" % self.ci
       elif self.c=='orange':
           self.c = (235./256., 106./256., 10./256., self.a)
           self.ci = (235, 106, 10)
           self.s = "#%02X%02X%02X" % self.ci
       elif self.c=='darkblue':
           self.c = (0./256., 110./256., 146./256., self.a)
           self.ci = (0, 110, 146)
           self.s = "#%02X%02X%02X" % self.ci
       elif self.c=='blue':
           self.c = (37./256., 186./256., 226./256., self.a)
           self.ci = (37, 186, 226)
           self.s = "#%02X%02X%02X" % self.ci
       else:  # FhG RGB
           self.c = (23./256., 156./256., 125./256., self.a)
           self.ci = (23, 156, 125)
           self.s = "#%02X%02X%02X" % self.ci


    
    
def get_all_buildings_as_dict():
    ''' Get a dictionary of all TABULA buildings in the data/buildings directory '''

    path = 'data/buildings/'
    construction_states = ['0_soc', '1_enev', '2_kfw']
    buildings = {}
    for file in os.listdir(path):
        if file.startswith('sfh'):
            bldg_module_name = f'{path}{file}'.replace('/','.')[:-3]
            building_module = import_module(bldg_module_name)
            for construction_state in construction_states:
                bldg_name = f'{file[:-3]}_{construction_state}'
                building = getattr(building_module, bldg_name)
                buildings[bldg_name] = building
    return buildings
