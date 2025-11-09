
# imports and rendering options 
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter

# Set the default style
# Use seaborn style if available, otherwise use matplotlib defaults with grid
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    # Fallback to matplotlib's whitegrid configuration
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['axes.facecolor'] = 'white'
# Set the default font size
plt.rcParams["font.size"] = 14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.titlesize"] = 14

# Set the default size and DPI for figures
plt.rcParams["figure.dpi"] = 300

# remove the top, right and left line in the plot
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.bottom"] = False

# remove the vertical grid - most of the time it is not necessary
plt.rcParams["axes.grid.axis"] = "y"

# Define the primary color cycle
C0 = (23/255, 156/255, 125/255)
C1 = (0/255, 91/255, 127/255)
C2 = (166/255, 187/255, 200/255)
C3 = (0/255, 133/255, 152/255)
C4 = (57/255, 193/255, 205/255)
C5 = (178/255, 210/255, 53/255)

# Accent color
C6 = (230/255, 136/255, 60/255)
mpl.rcParams['axes.prop_cycle'] = cycler('color', [C0, C1, C2, C3, C4, C5, C6])


#plt.rcParams["figure.figsize"] = (14,6)



def plot_table_of_results(results):
    ''' Prints the table of results for the given period.
    

    Parameters
    ----------
    results: dict
        - states (pandas.DataFrame)  : containing all states of the rule based simulation
        - costs (pandas.DataFrame)   : containing all costs of the rule based simulation
        - control (pandas.Series)    : timeseries of control variable
    
    Returns
    -------
    None.
    '''

    # calc seasonal results
    E_el_sum = results['costs']['E_el'].sum() / 1000 # kWh
    Q_th_sum = (results['costs']['E_el'] * results['costs']['COP']).sum() /1000 # kWh

    P_el_peak = results['costs']['P_el'].max() / 1000 # kW
    Qdot_th_peak = results['costs']['Qdot_th'].max() / 1000 # kW

    seasonal_results = pd.DataFrame({
                                     "Peak El. Power [kW]": [round(P_el_peak,2)],
                                     "Peak Th. Power [kW]": [round(Qdot_th_peak,2)],
                                     "Electricity Cons. [kWh]": [round(E_el_sum,2)],
                                     "Heat Demand [kWh]": [round(Q_th_sum,2)],
                                     "SCOP": [round((Q_th_sum / E_el_sum),2)],
                                     # add more results here
                                     })

    # use matplotlib to plot a nice table
    fig, ax = plt.subplots(figsize=(14, 1))
    ax.axis('off')
    ax.table(cellText=seasonal_results.values,
                    colLabels=seasonal_results.columns,
                    cellLoc='center', loc='center', edges='open')
    plt.show()



def plot_temperatures(results, disturbances):

    fig, ax = plt.subplots()
    ax = plot_bldg_temperatures(ax, results)
    
    date_form = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.plot(results['states']['T_hp_ret'], label = 'Return', color = 'C4')
    ax.plot(results['control'], label = 'Supply', color = 'C6')
    ax.plot(disturbances['T_amb'], label = 'Ambient', color = 'C1')
    ax.set_ylabel('Temperature in °C')
    fig.legend(bbox_to_anchor=[0.5, 0.95], loc = 'center', ncol=5, frameon = True)

def plot_bldg_temperatures(ax, results):
    ''' Plots the building states, controls, and thermal power of the hp for the given period.

    Parameters
    ----------
    results: dict
        - states (pandas.DataFrame)  : containing all states of the rule based simulation
        - costs (pandas.DataFrame)   : containing all costs of the rule based simulation
        - control (pandas.Series)    : timeseries of control variable

    Returns
    -------
    None.

    '''
    #fig, ax = plt.subplots()
    for state in results['states'].columns:
        if 'T_bldg' in state:
            ax.plot(results['states'][state], label = "Building", color = 'C0')
        if 'T_room' in state:
            ax.plot(results['states'][state], label = "Room", color = 'C2')
        if 'T_wall' in state:
            ax.plot(results['states'][state], label = "Wall", color = 'C3')
    ax.set_ylabel('Temperature in °C')
    return ax

def plot_disturbances(disturbances):
    ''' Plots the ambient temperature, solar gains and internal heat gains for the given period.

    Parameters
    ----------
    disturbances: pandas df
        - T_amb : ambient temperature [degC]
        - Qdot_tot : total heat gains (solar and internal gains) [W]
        - Qdot_sol : solar heat gains  [W]
        - Qdot_int : internal heat gains (occupancy and appliances) [W]

    Returns
    -------
    None.

    '''

    fig, ax = plt.subplots()
    #ax.set_title('Disturbances')

    ax.plot(disturbances['T_amb'], label = 'Ambient Temperature [°C]', color = 'C1')

    # Set the date format
    date_form = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(date_form)

    ax1 = ax.twinx()
    ax1.plot(disturbances['Qdot_sol'] / 1000, label = 'Solar Gains [kW]', color = 'C6')
    ax1.plot(disturbances['Qdot_int'] / 1000, label = 'Internal Gains [kW]', color = 'C5')
    ax.set_ylabel('Temperature in °C')
    ax1.set_ylabel('Heatflow in kW')
    fig.legend(ncol = 3,frameon = True, bbox_to_anchor=[0.5, 0.95], loc = 'center')



def plot_temperature_deviations(results):
    ''' Boxplot of the temperature deviations for the given period.

    Parameters
    ----------
    results: dict
        - states (pandas.DataFrame)  : containing all states of the rule based simulation
        - costs (pandas.DataFrame)   : containing all costs of the rule based simulation
        - control (pandas.Series)    : timeseries of control variable

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots()
    deviations = [ - results['costs']['dev_neg_max'], results['costs']['dev_pos_max'],  - results['costs']['dev_neg_sum'], results['costs']['dev_pos_sum']]
    ax.boxplot(deviations, labels=['dev_neg_max [K]','dev_pos_max [K]','dev_neg_sum [Kh]','dev_pos_sum [Kh]']);
    ax.set_ylabel('Temperature deviation');



def plot_system_performance(results):
    '''Box plot of electrical and thermal power. Bar graph of the electric and thermal energy. Boxplot for COP

    Parameters
    ----------
    results: dict
        - states (pandas.DataFrame)  : containing all states of the rule based simulation
        - costs (pandas.DataFrame)   : containing all costs of the rule based simulation
        - control (pandas.Series)    : timeseries of control variable

    Returns
    -------
    None.
    
    '''
    fig, axs = plt.subplots(1, 3, figsize=(12,4))

  
    # 1. SUBPLOT - LOADS
    # create a load profile for the electrical and thermal power
    # sort the values from high to low
    P_el = results['costs']['P_el'].sort_values(ascending = False).values / 1000
    Q_th = results['costs']['Qdot_th'].sort_values(ascending = False).values / 1000
    # create the x-axis
    x = range(len(P_el))
    # plot the electrical and thermal power
    axs[0].bar(x, Q_th, label = 'Thermal Power', color = 'C1')
    axs[0].bar(x, P_el, label = 'Electrical Power', color = 'C0')
    # set title for the subplot
    axs[0].set_title('Loads',loc = 'left')
    # set the y-axis label
    axs[0].set_ylabel('Power [kW]')


    # 2. SUBPLOT - ENERGY
    Q_th = (results['costs']['E_el'] * results['costs']['COP']).sum() / 1e3 # kWh
    E_el = results['costs']['E_el'].sum() / 1e3 # kWh
    axs[1].bar([0,1], [E_el, Q_th], tick_label = ['Electricity', 'Heat'], color = ['C0', 'C1']);
    # add the values to the bar plot
    for i in range(2):
        axs[1].text(i, [E_el, Q_th][i], round([E_el, Q_th][i],2), ha = 'center', va = 'bottom')

    axs[1].set_ylabel('Energy [kWh]');
    # remove the y-axis ticks
    axs[1].set_yticks([])
    # set the title for the subplot, align the title to the left
    axs[1].set_title('Demands', loc = 'left')

    # 3. SUBPLOT - COP
    axs[2].boxplot([results['costs']['COP']], labels=['COP'], widths=[0.25]);
    # add the values to the boxplot
    #axs[2].text(1, results['costs']['COP'].values, ha = 'center', va = 'bottom')
    axs[2].set_title('Coefficient of Performance', loc = 'left')
    # set the y-axis min to 0 leave the max open
    axs[2].set_ylim(0, None)


def plot_report(results, disturbances):
    '''Plots the report for the given period.

    Returns
    -------
    None.

    '''
    #table_of_results(results)
    plot_temperatures(results, disturbances)
    #plot_temperature_deviations(results)
    plot_system_performance(results)
    plot_disturbances(disturbances)
    plt.show()