# -*- coding: utf-8 -*-
"""Fast and simple environment for building and HVAC simulation."""

import numpy as np
import pandas as pd
import random

from scipy.integrate import solve_ivp

class Model_simulator:
    """Simulator for one-step integration and rule-based simulation using a heating curve.
    
    Attributes
    ----------
    hp_model : model_hvac.Heatpump_xx
        Model of a heat pump.

    bldg_model : model_buildings.Building
        Model of a building.

    timestep : int, optional
        Sampling interval [s]. The default is 3600 s.
    """
    def __init__(self,hp_model,bldg_model,timestep = 3600):
        self.hp_model = hp_model
        self.bldg_model = bldg_model
        self.timestep = timestep  # sampling interval in s
        
    def get_next_state(self, x_init, uk, pk):
        """Compute next building temperature states given the heat pump supply setpoint.
        
        Parameters
        ----------
        x_init : dict
            Initial state of the system (state keys depend on the selected method).
            Available keys are listed in building_model.state_keys after initialisation.
        
        uk : float
            Control variable:
            T_hp_sup (float)  : Supply flow temperature [°C]
            
        pk : dict
            Disturbances:

            - T_amb    (float)  : Ambient temperature [°C]
            - Qdot_gains (float) : Heat gains (sum of: occupancy, appliances, solar) [W]

        Returns
        -------
        dict
            - state (dict) : dictionary containing the next state of the system 
            - cost  (dict) : dictionary containing cost of heat pump and building        
        """
        
        # Get available state and required input keys from building_model
        state_keys =  self.bldg_model.state_keys
        input_keys = self.bldg_model.input_keys
        
        # Create initial state based on state_keys and x_init dict
        x_init_np = np.array([x_init[key] for key in state_keys])
             
        # Combine control variable uk and disturbance vector pk
        # to input vector as needed for state space equation
        input_dict = {'T_hp_sup' : uk, **pk}
        
        # Extract values from input vector based on
        # required keys in building_model
        input_values = [[input_dict[key] for key in input_keys]]
        
        # Compute next state
        ode_result_obj = solve_ivp(self.bldg_model.calc, #model.dynamics,
                                   [0, self.timestep],
                                   x_init_np,
                                   args = input_values,
                                   method = 'LSODA')
        """ Integration method to use:
        'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]. The error is controlled assuming accuracy of the fourth-order method, but steps are taken using the fifth-order accurate formula (local extrapolation is done). A quartic interpolation polynomial is used for the dense output [2]. Can be applied in the complex domain.
        'LSODA': Adams/BDF method with automatic stiffness detection and switching [7], [8]. This is a wrapper of the Fortran solver from ODEPACK.
        """
      
        if ode_result_obj.success is False:
            print(ode_result_obj.message)

        # The solve_ivp function returns several intermediate states. 
        ode_result = ode_result_obj.y # y = integration results with all inter. states

        # Save all intermediate states of the integration
        state_dict =  {key: ode_result[i]for i, key in enumerate(state_keys)}
        
        # Save final state of the integration
        next_state = {key: ode_result[i,-1]for i, key in enumerate(state_keys)} 
        
        # Compute costs
        cost = self.calc_cost(state_dict = state_dict, 
                              input_dict = input_dict)
       
        # Calculate operative room temperature
        if "T_op" in state_keys:
            if "T_surf_floor" in state_keys:
                f_floor = self.bldg_model.params['area_floor'] / self.bldg_model.params['A_surf']
                next_state["T_surf"] = f_floor * next_state['T_surf_floor'] + (1 - f_floor) * next_state['T_surf_wall']
            next_state["T_op"] = 0.7 * next_state["T_room"] + 0.3 * next_state["T_surf"]

        return {"state": next_state, "controls": uk, "parameters": pk, "cost" : cost}
    

    def simulate(self, x_init, p, T_room_set = 20, SCENARIO_MODE = False, ctrl_method = 'heatcurve', 
                 num_steps = 0,  **kwargs):
        """Building simulation with heating curve or PID controller.

        Make sure that the frequency of the datetime index is equal to the timestep of the Simulator.
     
        Parameters
        ----------
        x_init : dict
            Initial state of the system (state keys depend on calculation method).
            They can be found in Building.state_keys after initialisation.

        T_room_set : float
            Set point temperature for indoor comfort [degC]

        p : pandas.DataFrame
            Disturbances:
            - index      (datetime.index)  : date and time
            - T_amb      (timeseries)      : Ambient temperature [deg C]
            - Qdot_gains (timeseries)      : Heat gains (sum of: occupancy, appliances, solar) [W]
            - Qdot_int   (timeseries)      : Precomputed internal gains [W]
            - Qdot_sol   (timeseries)      : Precomputed solar gains [W]

        ctrl_method : str
            Control strategy, possible options: 'heatcurve' (default), 'pid'.

        num_steps : float, optional
            Number of simulation steps. If not specified, the length of the disturbance vector is used.

        **kwargs
            If ctrl == "pid": **kwargs = 
                - KP (float) : Proportional control gain
                - KI (float) : Integrative control gain
                - KD (float) : Differential control gain
            If ctrl == "heatcurve": 
                - shift  : shift of heating limit temperature [K]
                - offset : offset of supply and return flow temperature [K]

        Returns
        -------
        dict
            - states (pandas.DataFrame)  : containing all states of the rule based simulation
            - costs (pandas.DataFrame)   : containing all costs of the rule based simulation
            - control (pandas.Series)    : timeseries of control variable
        """

        if ctrl_method == 'heatcurve':
            from src.controller.heatcurve import heatcurve
            ctrl = heatcurve.Heatingcurve(T_room_set = T_room_set)
        elif ctrl_method == 'pid':
            from src.controller.pid import pid
            ctrl = pid.PID(timestep = self.timestep, KP = kwargs['KP'], KI = kwargs['KI'], KD = kwargs['KD'])
        
        # If maximum number of steps is not defined, use length of disturbance vector
        if num_steps: 
            p = p.iloc[:num_steps] 

        timestep = self.timestep # length of one simulation step [s]

        # INITIALIZE STATES_DF
        # Get available state and control keys from building_model
        state_keys =  self.bldg_model.state_keys
        
        # Create dataframes to store results
        states_df = pd.DataFrame(index = p.index,
                                 columns=([key for key in state_keys]))
        
        # We have to add an additional index at the end of the df
        timestep = p.index[1] - p.index[0] # Extract length of timestep from disturbance data
        last_index = p.index[-1] + timestep # Create index for last simulation step
        states_df.loc[last_index] = [np.nan for key in state_keys] # Append row at end of df
        
        # Insert initial values at initial index
        states_df.iloc[0] = x_init

        # INITIALIZE TIME SERIES FOR CONTROL VARIABLE
        u = pd.Series(np.zeros(len(p)), name = 'T_hp_sup', index = p.index)
       
        # INITIALIZE COST DF
        # Generate pseudo-cost dictionary to get all available keys 
        cost_bldg = self.bldg_model.calc_comfort_dev(T_room = [0,0], timestep = 3600)
        cost_hp = self.hp_model.calc(T_hp_ret = [0,0], T_hp_sup = 0, T_amb = 0, timestep = 3600)
        
        # Extract cost_keys from pseudo dict to initialize cost_df
        cost_keys = {**cost_bldg, **cost_hp}.keys()
        
        # Create dataframe for individual costs and add cost_keys as column names
        costs_df = pd.DataFrame(index = states_df.index,
                                columns=([key for key in cost_keys]))
        
        # Initialize first row of costs_df with zeros (will be filled after first step)
        costs_df.iloc[0] = 0.0
        
        bldg_model_params_H_ve = self.bldg_model.params['H_ve'] # store value
        bldg_model_params_H_rad_con = self.bldg_model.params['H_rad_con'] # store value
        bldg_model_params_H_tr_heavy = self.bldg_model.params['H_tr_heavy']
        bldg_model_params_H_tr_light = self.bldg_model.params['H_tr_light']
                
        for i in range(len(p)):
            pk = p.iloc[i].to_dict() # get disturbance values for current time step

            # Calculate control variable
            if ctrl_method == 'pid':
                uk = ctrl.calc(set_point = T_room_set, actual_value = states_df['T_room'].iloc[i]) # Supply temperature [degC]
            elif ctrl_method == 'heatcurve':
                shift, offset = 0, 0
                if 'shift' in kwargs:
                    shift = kwargs['shift']
                if 'offset' in kwargs:
                    offset = kwargs['offset']
                uk = ctrl.calc(pk['T_amb'], shift_T_lim = shift, offset_T_flow_nom = offset)[0]

            # Only if upper heating limit temperature is not exceeded by ambient temperature
            if pk['T_amb'] < self.bldg_model.params['T_amb_lim']:
                # Set control variable supply flow temperature according to heating curve
                uk = max(uk + self.bldg_model.params['T_offset'], states_df['T_hp_ret'].iat[i])
            else:
                # Set supply flow temperature to return flow temperature -> no heating
                uk = states_df['T_hp_ret'].iat[i]
            
            # Check if HP supply temperature is within the range
            uk = self.hp_model.check_hp(uk, states_df['T_hp_ret'].iat[i])
            
            # Scenario generation
            if SCENARIO_MODE:
                if i%8==0: random_var = random.choices([0, 1], weights=[99,1])[0]
                if random_var == 1:
                    self.bldg_model.params['H_ve']=10*bldg_model_params_H_ve
                    self.bldg_model.params['H_rad_con']=1*bldg_model_params_H_rad_con
                    self.bldg_model.params['H_tr_heavy']=5*bldg_model_params_H_tr_heavy
                    self.bldg_model.params['H_tr_light']=5*bldg_model_params_H_tr_light
                else:
                    self.bldg_model.params['H_ve']=bldg_model_params_H_ve
                    self.bldg_model.params['H_rad_con']=bldg_model_params_H_rad_con
                    self.bldg_model.params['H_tr_heavy']=bldg_model_params_H_tr_heavy
                    self.bldg_model.params['H_tr_light']=bldg_model_params_H_tr_light                
            
            # Do one-step integration and save as next state
            results = self.get_next_state(x_init = states_df.iloc[i].to_dict(), 
                                          uk = uk, pk = pk)

            u.iat[i] = uk
            states_df.iloc[i + 1] = results['state']
            costs_df.iloc[i + 1]  = results['cost']
            
              
        return {'states': states_df, 'costs' : costs_df, 'control' : u, 'parameters' : p}

    def calc_cost(self, state_dict, input_dict):
        ''' Calculate the costs for one time step of heat pump and building models.
        
        Parameters
        ----------
        state_dict : dict
            Intermediate states of last simulation step of the system
            (State keys depend on calculation method, they can be found
            in Building.state_keys after initalisation.) e.g. '2R2C':

            - T_room       (list or numpy.ndarray) : Intermediate states of indoor air temperature - depending on building model either T_bldg, T_zone, T_air, ...
            - T_hp_ret     (list or numpy.ndarray) : Intermediate states of return temperature as returned by solve_ivp()
        
        input_dict : dict

            - T_hp_sup  (float) : Supply temperature of the hvac system 
            - T_amb     (float) : Ambient temperature 

        Returns
        -------
        dict
            costs of heat pump model and building model
        '''
        # Compute performance of the heat pump
        cost_hp = self.hp_model.calc(T_hp_ret = state_dict['T_hp_ret'],
                                     T_hp_sup = input_dict['T_hp_sup'],
                                     T_amb = input_dict['T_amb'],
                                     timestep = self.timestep)


        # Calculate comfort deviation of the building
        cost_bldg = self.bldg_model.calc_comfort_dev(T_room = state_dict['T_room'],
                                                     timestep = self.timestep)
    
        return {**cost_hp, **cost_bldg}


def main():
    import sys
    from pathlib import Path
    
    # Ensure the root of your package is in the PYTHONPATH
    root_path = str(Path(__file__).resolve().parent)
    print(root_path)
    sys.path.insert(0, root_path)
    
    
    import src.models.model_buildings as model_buildings
    
    # load example building data
    from data.buildings.sfh_1919_1948 import sfh_1919_1948_0_soc as building
    
    # Initialize the building model
    bldg_model = model_buildings.Building(params    = building, # More example buildings can be found in data/buildings/.
                                          mdot_hp   = 0.25,          # Massflow of the heat supply system. [kg/s]
                                          method    = '7R5C',        # For available calculation methods see: model_buildings.py.
                                          verbose   = False)
    
    import src.models.model_hvac as model_hvac
    
    # Initalize the heat pump model, 
    # set verbose to True, to generate a performance graph of the HP
    hp_model = model_hvac.Heatpump_AW(mdot_HP = 0.25, verbose = False)
    
    # Set the control variable
    uk = 20  # T_hp_sup [°C]
    
    # Either manually create the disturbance dict or choose one timestep (rand_int) of the previously generated disturbance dataframe.
    pk = {'T_amb': 2.9,     # ambient temperature
     'Qdot_gains': 1046,    # total gains (sum of solar and internal gains)
     'Qdot_sol': 115,       # solar gains
     'Qdot_int': 931}       # internal gains
    
    # Get available state keys from building_model
    state_keys = bldg_model.state_keys
    #print(state_keys)
    
    # Setup initial state vector
    x_init = {key : 20 for key in state_keys} # here all inital states are set to 20 °C
    
    m = Model_simulator(bldg_model = bldg_model,
                        hp_model   = hp_model)
    
    print(m.get_next_state(x_init = x_init, uk = uk, pk = pk))
    
if __name__ == '__main__':
    main()