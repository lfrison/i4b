# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import src.models.model_hvac
import src.simulator
import casadi as cas

# generic constants
RHO_WATER = 997      # Density water [kg/m3]
RHO_AIR = 1.225      # Density air [kg/m3]
C_WATER_SPEC = 4181  # Spec. heat capacity of water [J/kg/K]
C_AIR_SPEC = 1.005   # Spec. heat capacity of air [J/kg/K]

# Specify building constants
C_INT_SPEC = 10000    # [J/m^2/K]
H_UFH_SPEC = 4.4      # [W/m^2/K]
H_UFH_SURF_SPEC = 10.8 # W/m^2/K
V_TS_SPEC  = 5        # [l/m^2]
V_UFH_SPEC = 1.5      # [l/m^2]
R_SI       = 0.13     # [m^2*K/W]
h_tr_int   = 9.1      # [W/(m^2K)] heattransfer coefficient between building mass and indoor surface ISO13790 12.2.2
h_air2surf = 3.45     # [W/(m^2K)] heattransfer coefficient between indoor surface and indoor air ISO13790 7.2.2.2
c_P_screed = 1000 # specific heat capacity of screed [J/(kgK)] source: https://www.schweizer-fn.de/stoff/wkapazitaet/wkapazitaet_baustoff_erde.php
d_screed = 50     # thickness of screed for UFH [mm] source: https://www.kesselheld.de/heizestrich/
rho_screed = 2000 # dry bulk density of screed (original German: Trockenrohdichte) [kg/m^3] source: https://www.profibaustoffe.com/wp-content/files/TD_2059_2056_ESTRICH-CT-C20-F4-E225-MIT-FASERN_041119.pdf


class Building:
    """ With the building class, a bulding model can be generated from pysical and geometrical parameters as specified in the data/buildings directory.
    Different calculation methods are available to model the heat flows of the building and an underfloor heating system in a dynamic way (2R2C, 4R3C, 5R4C).
    Ordinary differential equations (ODEs) are used to model the respective thermal networks, containing different thermal resistances (R) and capacities (C).
    The ODEs are transformed into state space equations in the form :math:`\dot{x} = A \cdot x + B \cdot u` and :math:`y = C \cdot x + D \cdot u` with:
        - A : System matrix
        - B : Control matrix
        - C : Output matrix
        - D : Feedthrough matrix
        - x : state vector
        - u : input vector (containing control and disturbance variables)
        - y : output vector

    For more details about the individual calculation methods check out the documentation of the individual calc_xRxC functions.
    
    The required input parameters for the chosen calculation method are computed while initializing the building model,
    using the input parameters of the individual building and the following general constants:
        - C_INT_SPEC = 10000 J/m^2/K    : Spec heat capacitiy of interior acc. to ISO52016 Tab. B17
        - H_UFH_SPEC = 4.4 W/m^2/K      : Spec heat transmission coefficient for underfloor heating acc. to `Daniel Rüdiser <https://www.htflux.com/en/dynamic-simulation-and-comparison-of-two-underfloor-heating-systems/>`_
        - H_UFH_SURF_SPEC = 10.8 W/m^2/K : Spec heat convection from floor to air with underfloor heating system acc to EN 1264-5
        - V_TS_SPEC = 5 l/m^2           : Spec volume of thermal heat storage per heated floor area acc. to DGS – German Society for Solar Energy (original German: Deutsche Gesellschaft für Sonnenenergie). Guide to Solar Thermal Systems – Expert Presentation on DVD, 9th Edition (original German: Leitfaden Solarthermische Anlagen – Experten-Vortrag auf DVD, 9. Auflage)
        - V_UFH_SPEC = 1.5 l/m^2        : Spec volume of water in underfloor heating system per heated floor area acc. to `BaCoGa Technik GmBH <https://www.bacoga.com/wp-content/uploads/2013/02/Volumenberechnung.pdf>`_
        - R_SI = 0.13 m^2*K/W           : Heat resistance of internal surfaces acc. to DIN EN ISO 6946 Tabelle 7. 
        - h_tr_int = 9.1 W/(m^2K)       : Heattransfer coefficient between building mass and indoor surface ISO13790 12.2.2
        - h_air2surf = 3.45 W/(m^2K)    : Heattransfer coefficient between indoor surface and indoor air ISO13790 7.2.2.2
        - c_P_screed = 1000 # specific heat capacity of screed [J/(kgK)] source: https://www.schweizer-fn.de/stoff/wkapazitaet/wkapazitaet_baustoff_erde.php
        - d_screed = 50     # thickness of screed for UFH [mm] source: https://www.kesselheld.de/heizestrich/
        - rho_screed = 2000 # dry bulk density of screed (original German: Trockenrohdichte) [kg/m^3] source: https://www.profibaustoffe.com/wp-content/files/TD_2059_2056_ESTRICH-CT-C20-F4-E225-MIT-FASERN_041119.pdf

    Attributes
    ----------
   
    params : dict
        Containing easily accissible building parameters as specified in buildings.py
        
    mdot_hp : float (optional)
        :math:`\dot{m}_{hp}`  : Mass flow rate of the hp system, default = 0.25 [kg/s]
        
    T_room_set_lower : float (optional)
        :math:`T_{room,set,lower}`  : Lower set point temperature, default = 18 [°C]
    
    T_room_set_upper : float (optional)
        :math:`T_{room,set,upper}`  : Upper set point temperature, default = 26 [°C]    
    
    method : string
        Choose calculation method ['2R2C' (default), '4R3C', '5R4C', '6R4C'].
        The electable calc method depends on the provided building parameters.
        The required parameters are described for each calculation method separately.
    
    usage : string
        Choose usage of the building ['Office', 'ResidentialDetached' (default), 'ResidentialFlat', 'SchoolClassroom'].

    verbose: bool
        flag to print and plot further building information

    Notes
    -----
        The following parameters are calculated and added to the params dictionary by initialization. 

            - :math:`H_{ve,tr} = H_{ve} + H_{tr}`   (H_ve_tr)      : Heat transfer coefficient for ventilation and transmission (indoors --> ambient) [W/K]
            - :math:`H_{ve}`                        (H_ve)         : Heat transfer coefficient for ventilation (indoors --> ambient) [W/K]
            - :math:`H_{tr}`                        (H_tr)         : Heat transfer coefficient for transmission (indoors --> ambient) [W/K]
            - :math:`H_{rad,con}`                   (H_rad_con)    : Heat transfer coefficient for radiation and convection (hvac --> indoors) [W/K]
            - :math:`H_{int}`                       (H_int)        : Heat transfer coefficient for convection and radiation (interior surfaces --> indoor air) [W/K]
            - :math:`H_{tr,heavy}`                  (H_tr_heavy)   : Heat transfer coefficient for the heavy building components (wall, floor, roof) (indoors --> ambient) [W/K]
            - :math:`C_{bldg} = C_{wall} + C_{zone}` (C_bldg)      : Heat capacity of the entire building (air and wall) [J/K]
            - :math:`C_{wall}`                      (C_wall)       : Heat capacity of the walls [J/K]
            - :math:`C_{zone} = C_{air} + C_{int}`  (C_zone)       : Heat capacity of the thermal zone [J/K]
            - :math:`C_{air}`                       (C_air)        : Heat capacity of the indoor air [J/K]
            - :math:`C_{int}`                       (C_int)        : Heat capacity of the interior [J/K]
            - :math:`C_{water}`                     (C_water)      : Heat capacity of the water in the hvac system [J/K]
            - :math:`C_{surf}`                      (C_surf)       : Heat capacity fo the inside envelope surface [J/K]
            - :math:`C_{surf_wall}                  (C_surf_wall)  : Heat capacitiy of the wall surface
            - :math:`C_{surf_floor}                 (C_surf_floor) : Heat capacitiy of the floor surface
            - :math:`C_{bldg,heavy}`                (C_bldg_heavy) : Heat capacity of the heavy building components, excluding the inside envelope surface [J/K]
            - :math:`A_{surf}`                      (A_surf)       : Internal surface area [m^2]
            - :math:`A_{mass}`                      (A_mass)       : Internal surface area of the heavy building components [m^2] 
            - :math:`V_{air}`                       (volume_air)   : Indoor air volume [m^3]

    """   
    def __init__(self, 
                 params      = None,
                 mdot_hp = 0.25,
                 T_room_set_lower = 20,
                 T_room_set_upper = 26,
                 method     = '2R2C',
                 usage = 'ResidentialDetached',
                 verbose = False
                ):
            
        self.params = params
        self.mdot_hp = mdot_hp
        self.T_room_set_lower = T_room_set_lower
        self.T_room_set_upper = T_room_set_upper
        self.method = method
        self.usage = usage
        self.__calc_bldg_parameters()
        self.__init_calc_method()

        if verbose == True:
            self.print_params()
            self.plot_step_response()
        
    def __calc_bldg_parameters(self):
        ''' Function to calculate additional building parameters.
        
        Parameters
        ----------
        params : dict
            Containing easily accessible building parameters as specified in docs/buildings/readme.md
        '''
        self.params['volume_air'] = self.params['area_floor'] * self.params['height_room'] # [m^3]
        self.params['H_ve_tr'] = self.params['H_ve'] + self.params['H_tr'] # [W/K]
        if self.params['H_tr_light']:
            self.params['H_tr_heavy'] = self.params['H_tr'] - self.params['H_tr_light']
        self.params['H_con_floor'] = H_UFH_SURF_SPEC * self.params['area_floor'] # [W/K]
        self.params['H_rad_con'] = H_UFH_SPEC * self.params['area_floor'] # [W/K]
        self.params['H_tr_floor'] = 1 / ((1 / self.params['H_rad_con']) - (1 / self.params['H_con_floor'])) # [W/K]
        volume_water = (V_TS_SPEC + V_UFH_SPEC) * self.params['area_floor'] / 1000 # [m^3]
        self.params['C_water'] = RHO_WATER * C_WATER_SPEC * volume_water # [J/K]
        self.params['C_air'] = RHO_AIR * C_AIR_SPEC * self.params['volume_air'] # [J/K]
        self.params['C_int'] = C_INT_SPEC * self.params['area_floor'] # [J/K]
        self.params['C_zone'] = self.params['C_air'] + self.params['C_int'] # [J/K]
        self.params['C_bldg'] = self.params['c_bldg'] * self.params['area_floor'] * 3600 # [J/K]
        self.params['C_wall'] = self.params['C_bldg'] - self.params['C_zone'] # [J/K]
        self.params['C_surf'] = (self.params['C_bldg'] - self.params['C_air']) * 1 / 8 # [J/K] share of surface mass according to ISO 52016 equally distributed building mass
        self.params['C_bldg_heavy'] = self.params['C_bldg'] - self.params['C_air'] - self.params['C_surf'] # [J/K]
        self.params['A_surf'] = self.params['area_floor'] * 4.5 # see ISO13760 7.2.2.2
        f_floor = self.params['area_floor'] / self.params['A_surf']
        self.params['C_surf_floor'] =  f_floor *  self.params['C_surf']  # heat capacity of floor surface [J/K]
        self.params['C_surf_wall'] = (1 - f_floor) * self.params['C_surf']
        self.params['H_int'] = 1 / R_SI *  self.params['A_surf']
        window_area = 0
        for orientation in self.params['windows']:
            window_area += orientation['area']
        self.params['A_mass'] = self.params['A_surf'] - window_area # Surface area of heavy bldg components

    def __init_calc_method(self):
        """ Set state and input keys according to selected building model building.
        """
        if self.method == '2R2C':
            self.state_keys = ("T_room", "T_hp_ret")
            self.input_keys = ("T_hp_sup", "T_amb", "Qdot_gains")
        elif self.method == '4R3C':
            self.state_keys = ("T_room", "T_wall", "T_hp_ret") 
            self.input_keys = ("T_hp_sup", "T_amb", "Qdot_gains") 
        elif self.method == '5R4C':
            self.state_keys = ("T_room", "T_int", "T_wall", "T_hp_ret") 
            self.input_keys = ("T_hp_sup", "T_amb", "Qdot_gains")
        elif self.method == '6R4C':
            self.state_keys = ("T_room", "T_surf", "T_op", "T_mass", "T_hp_ret") 
            self.input_keys = ("T_hp_sup", "T_amb", "Qdot_gains", "Qdot_int", "Qdot_sol") 
        elif self.method == '7R5C':
            self.state_keys = ("T_room", "T_surf_wall", "T_op", "T_mass", "T_surf_floor",  "T_hp_ret")
            self.input_keys = ("T_hp_sup", "T_amb", "Qdot_gains", "Qdot_int", "Qdot_sol")
        else:
            print("Calculation method does not exist")
    

    def calc(self, t, x, args):
        ''' Chooses and calls the building model class depending on selected method.
        
        Parameters
        ----------
        t : not used - but needed for scipy.integrate.solve_ivp() which calls this function

        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[i] : depending on selected bldg model [degC]
                - x[-1] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations according to selected building model
        '''

        if self.method == '2R2C':
            return self.calc_2r2c(t, x, args[0], args[1:])
        elif self.method == '4R3C':
            return self.calc_4r3c(t, x, args[0], args[1:])
        elif self.method == '5R4C':
            return self.calc_5r4c(t, x, args[0], args[1:])
        elif self.method == '6R4C':
            return self.calc_6r4c(t, x, args[0], args[1:])
        elif self.method == '7R5C':
            return self.calc_7r5c(t, x, args[0], args[1:])
        else:
            print("Calculation method does not exist")       


    def calc_2r2c(self, t, x, u, p):
        ''' Calculate state changes of building temperature and return flow temperature. 
        
        The following building attributes have to be set:
        H_ve_tr, H_rad_con, C_bldg, C_water

        The corresponding state vector has the following entries:
        [T_room, T_hp_ret]

        Parameters
        ----------
        t : not used - but needed for scipy.integrate.solve_ivp() which calls this function

        x : list
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_hp_ret/dt : return flow temperature [degC/s]
        
        Notes
        -----
        The equivalent thermal network is shown here:

        .. image:: ../graphics/2R2C_network.svg
            :width: 100 %
        
        The resulting state space equation is as follows: 

        .. image:: ../graphics/2R2C_equation.svg
           :width: 100 %
                
        The resulting ordinary differential equations are as follows:
            
        :math:`\dot{T}_{room} = 1/C_{bldg} \cdot ( \dot{Q}_{gain} + H_{rad,con} \cdot (T_{hp,ret} - T_{room}) - H_{ve,tr} \cdot (T_{room}- T_{amb}) )`
        :math:`\dot{T}_{hp,ret} = 1/C_{water} \cdot ( \dot{m}_{hp} \cdot c_{p,water} \cdot (T_{hp,sup} - T_{hp,ret}) - H_{rad,con} \cdot (T_{hp,ret} - T_{room}) )`
        '''
        T_room     = x[0] # [degC]
        T_hp_ret   = x[1] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = np.zeros(2)
        rhs[0] = 1/self.params['C_bldg'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) - self.params['H_ve_tr'] * (T_room - T_amb)) # T_room   [degC/s]
        rhs[1] = 1/self.params['C_water'] * (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) - self.params['H_rad_con'] * (T_hp_ret - T_room))    # T_hp_ret [degC/s]
        return rhs

    def calc_4r3c(self, t, x, u, p):
        ''' Calculate state changes of building temperature and return flow temperature.

        The following building attributes have to be set:
        H_tr, H_ve, H_rad_con, C_wall, C_zone, C_water

        The corresponding state vector has the following entries:
        [T_room, T_wall, T_hp,ret]
        
        Parameters
        ----------
        x : list
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Wall temperature [degC]
                - x[2] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector 
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_wall/dt   : wall temperature [degC/s]
            - rhs[2] of dT_hp_ret/dt : return flow temperature [degC/s]
        
        Notes
        -----
        The equivalent thermal network is shown here:

        .. image:: ../graphics/4R3C_network.svg
            :width: 100 %

        The resulting state space equation is as follows: 

        .. image:: ../graphics/4R3C_equation.svg
           :width: 100 %

        The ordinary differential equations for the 4R3C calculation method are:
            
        :math:`\dot{T}_{room} = 1/C_{zone} \cdot ( \dot{Q}_{gain} + H_{rad,con} \cdot (T_{hp,ret} - T_{room}) - 2 \cdot H_{tr} \cdot (T_{room} - T_{wall}) - H_{ve} \cdot (T_{room} - T_{amb}) )`
        :math:`\dot{T}_{wall} = 1/C_{wall} \cdot (2 \cdot H_{tr} \cdot (T_{room} - T_{wall}) - 2 \cdot H_{tr} \cdot (T_{wall} - T_{amb}) )`
        :math:`\dot{T}_{hp,ret} = 1/C_{water} \cdot ( \dot{m}_{hp} \cdot c_{p,water} \cdot (T_{hp,sup} - T_{hp,ret}) - H_{rad,con} \cdot (T_{hp,ret} - T_{room}) )`
        '''

        T_room     = x[0] # [degC]
        T_wall     = x[1] # [degC]
        T_hp_ret   = x[2] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = np.zeros(3)

        rhs[0] = 1 / self.params['C_zone'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) \
                - 2 * self.params['H_tr'] * (T_room - T_wall) - self.params['H_ve'] * (T_room - T_amb)) # T_room [degC/s]
        rhs[1] = 1 / self.params['C_wall'] * (2 * self.params['H_tr'] * (T_room - T_wall) \
                - 2 * self.params['H_tr'] * (T_wall - T_amb))     # T_wall [degC/s]
        rhs[2] = 1 / self.params['C_water'] * (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) \
                - self.params['H_rad_con'] * (T_hp_ret - T_room)) # T_hp_ret [degC/s]
        return rhs


    def calc_5r4c(self, t, x, u, p):
        ''' Calculate state changes of building temperature and return flow temperature
        
        The following building attributes have to be set:
        H_tr, H_ve, H_rad_con, H_int, C_wall, C_air, C_water, C_int

        The corresponding state vector has the following entries:
        [T_room, T_int, T_wall, T_hp,ret]

        Parameters
        ----------
        x : list
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Interior temperature [degC]
                - x[2] : Wall temperature [degC]
                - x[3] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 

        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_int/dt    : interior temperature [degC/s]
            - rhs[2] of dT_wall/dt   : wall temperature [degC/s]
            - rhs[3] of dT_hp_ret/dt : return flow temperature [degC/s]
        
        Notes
        -----
        The equivalent thermal network is shown here:

        .. image:: ../graphics/5R4C_network.svg
            :width: 100 %

        The resulting state space equation is as follows: 

        .. image:: ../graphics/5R4C_equation.svg
           :width: 100 %

        The ordinary differential equations for the 5R4C calculation method are:
            
        :math:`\dot{T}_{air} = 1/C_{air} \cdot ( \dot{Q}_{gain} + H_{rad,con} \cdot (T_{hp,ret} - T_{room}) - H_{int} \cdot (T_{room} - T_{int}) - 2 \cdot H_{tr} \cdot (T_{room} - T_{wall}) - H_{ve} \cdot (T_{room} - T_{amb}) )`
        :math:`\dot{T}_{int} = 1/C_{int} \cdot H_{int} \cdot (T_{room} - T_{int})`
        :math:`\dot{T}_{wall} = 1/C_{wall} \cdot (2 \cdot H_{tr} \cdot (T_{room} - T_{wall}) - 2 \cdot H_{tr} \cdot (T_{wall} - T_{amb}) )`
        :math:`\dot{T}_{hp,ret} = 1/C_{water} \cdot ( \dot{m}_{hp} \cdot c_{p,water} \cdot (T_{hp,sup} - T_{hp,ret}) - H_{rad,con} \cdot (T_{hp,ret} - T_{room}) )`
        '''

        T_room     = x[0] # [degC]
        T_int      = x[1] # [degC]
        T_wall     = x[2] # [degC]
        T_hp_ret   = x[3] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = np.zeros(4)
        rhs[0] = 1 / self.params['C_air'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) - self.params['H_int'] * (T_room - T_int) \
                - 2 * self.params['H_tr'] * (T_room - T_wall) - self.params['H_ve'] * (T_room - T_amb)) # T_room [degC/s]
        rhs[1] = 1 / self.params['C_int'] * self.params['H_int'] * (T_room - T_int) # T_int [degC/s]
        rhs[2] = 1 / self.params['C_wall'] * (2 * self.params['H_tr'] * (T_room - T_wall) \
                - 2 * self.params['H_tr'] * (T_wall - T_amb))     # T_wall [degC/s]
        rhs[3] = 1 / self.params['C_water'] * (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) \
                - self.params['H_rad_con'] * (T_hp_ret - T_room)) # T_hp_ret [degC/s]

        return rhs

    def calc_6r4c(self, t, x, u, p):
        ''' Calculate state changes of building temperature and return flow temperature
        
        The following building attributes have to be set:
        H_rad_con, H_ve, C_bldg,heavy, C_surf, C_air, C_water

        The corresponding state vector has the following entries:
        [T_room, T_surf, T_op, T_mass, T_hp,ret]

        Parameters
        ----------
        x : list
            state vector
                - x[0] : Interior air temperature [degC]
                - x[1] : Wall inside surface temperature [degC]
                - x[2] : Operative room temperature (not calculated here - see simulator.get_next_state)
                - x[3] : Building mass temperature [degC]
                - x[4] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 

        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed total gains [W]
                - p[2] : Precomputed internal gains [W]
                - p[3] : Precomputed solar gains [W]
        
        Returns
        -------
        List of right hand side equations
            - rhs[0] of dT_room/dt   : interior air temperature [degC/s]
            - rhs[1] of dT_surf/dt   : wall inside surface temperature [degC/s]
            - rhs[2] placeholder for operative temperature 
            - rhs[3] of dT_mass/dt   : building mass temperature [degC/s]
            - rhs[4] of dT_hp_ret/dt : return flow temperature [degC/s]
        
        Notes
        -----
        This building model is based on the ISO13790 which consists of a 5R1C model. One additional capacity to represent the thermal mass of the
        heat supply system and one resistance to calculate the heatflow from the heat supply system to the building are used.
        Furthermore capacities for the room air and the inside wall surface are added, resulting in a 6R4C thermal network as shown in the following figure:
        ! Please note that the surface temperature does not include the effects of the underfloor heating system.


        .. image:: ../graphics/6R4C_network.svg
           :width: 100 %
        
        The internal heatgains by occupants and appliances as well as the solar heat gains are distributed to the temperature nodes according to ISO13790.
        The capacties of the building envelope and the inside wall surfaces is calculated according the ISO52016 6.5.7.2 class D 
        for a wall with evenly distributed thermal mass.

        The ordinary differential equations for the 6R4C calculation method are:
        
        :math:`\dot{T}_{room} = 1 / C_{air} \cdot (\dot{Q}_{gain2air} + \dot{Q}_{hp2air} - \dot{Q}_{ve} - \dot{Q}_{air2surf})`
        :math:`\dot{T}_{surf} = 1 / C_{surf} \cdot (\dot{Q}_{gain2surf} + \dot{Q}_{air2surf} - \dot{Q}_{tr,light} - \dot{Q}_{tr,int})`
        :math:`\dot{T}_{mass} = 1 / C_{bldg,heavy} \cdot (\dot{Q}_{gain2mass} + \dot{Q}_{tr,int} - \dot{Q}_{tr,ext})`
        :math:`\dot{T}_{HP} = 1 / C_{water} \cdot (\dot{Q}_{hvac} - \dot{Q}_{hp2air}))`

        with

        :math:`\dot{Q}_{gain2air} = 0.5 \cdot \dot{Q}_{gain,int}`
        :math:`\dot{Q}_{gain2surf} = (1 - (A_{mass} / A_{surf}) - H_{tr,light} / (9.1 \cdot A_{surf})) \cdot (0.5 \cdot \dot{Q}_{gain,int} + \dot{Q}_{gain,sol})`
        :math:`\dot{Q}_{gain2mass} = (A_{mass} / A_{surf}) \cdot (0.5 \cdot \dot{Q}_{gain,int} + \dot{Q}_{gain,sol})`

        :math:`\dot{Q}_{hp2air} = \dot{m} \cdot c_{P,water} \cdot (T_{hp,sup} - T_{hp,ret})`
        :math:`\dot{Q}_{ve} = H_{ve} \cdot (T_{room} - T_{amb})`
        :math:`\dot{Q}_{air2surf} = H_{air2surf} \cdot (T_{room} - T_{surf})`
        :math:`\dot{Q}_{tr,light} = H_{tr,light} \cdot (T_{surf} - T_{amb})`
        :math:`\dot{Q}_{hp2air} = H_{rad,con} \cdot (T_{hp,ret} - T_{room})` 
        :math:`\dot{Q}_{tr,int} = H_{tr,int} \cdot (T_{surf} - T_{mass})`
        :math:`\dot{Q}_{tr,ext} = H_{tr,ext} \cdot (T_{mass} - T_{amb})`

        :math:`H_{tr,int}     = h_{tr,int} \cdot A_{mass}`
        :math:`1 / H_{tr,ext} = 1 / H_{tr,heavy} - 1 / H_{tr,int}`

        The operative room temperature for this model is calculated in the get_next_state function according to the following equation:

        :math:`T_{op} = 0.3 \cdot T_{room} + 0.7 \cdot T_{surf}`

        '''
        T_room      = x[0] # [degC]
        T_surf     = x[1] # [degC]
        # T_op     = x[2] # [degC] calculated after solving ODEs
        T_mass     = x[3] # [degC]
        T_hp_ret   = x[4] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]
        Qdot_int   = p[2] # [W]
        Qdot_sol   = p[3] # [W]


        Qdot_gain2air = 0.5 * Qdot_int # [W]
        Qdot_gain2mass = self.params['A_mass']/self.params['A_surf'] * (0.5 * Qdot_int + Qdot_sol) # [W]
        Qdot_gain2surf = (1 - (self.params['A_mass'] / self.params['A_surf']) - (self.params['H_tr_light']/(9.1 * self.params['A_surf']))) * (0.5 * Qdot_int + Qdot_sol) # [W]

        H_air2surf = h_air2surf * self.params['A_surf'] # [W/K]
        H_tr_int = h_tr_int * self.params['A_mass'] # [W/K] 
        H_tr_ext = 1 / ((1/self.params['H_tr_heavy']) - (1/H_tr_int))

        Qdot_tr_int = H_tr_int * (T_surf - T_mass) # [W]
        Qdot_tr_ext = H_tr_ext * (T_mass - T_amb) # [W]

        Qdot_hvac = self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
        Qdot_hp2air = self.params['H_rad_con'] * (T_hp_ret - T_room)

        rhs  = np.zeros(5)
        rhs[0] = 1 / self.params['C_air'] * (Qdot_gain2air + self.params['H_rad_con'] * (T_hp_ret - T_room) - self.params['H_ve'] * (T_room - T_amb) - H_air2surf * (T_room - T_surf))
        rhs[1] = 1 / self.params['C_surf'] * (Qdot_gain2surf + H_air2surf * (T_room - T_surf) - self.params['H_tr_light'] * (T_surf - T_amb) - H_tr_int * (T_surf - T_mass))
        # rhs[2] placeholder for T_op 
        rhs[3] = 1 / self.params['C_bldg_heavy'] * (Qdot_gain2mass + Qdot_tr_int - Qdot_tr_ext) # T_mass/dt [degC/s]
        rhs[4] = 1 / self.params['C_water'] * (Qdot_hvac - Qdot_hp2air) # T_hp_ret/dt [degC/s]

        return rhs   

    def calc_7r5c(self, t, x, u, p):
        ''' Calculate state changes of building temperature and return flow temperature
        
        The following building attributes have to be set:
        H_tr_floor, H_con_floor H_ve, C_bldg_heavy, C_surf_wall, C_air, C_water, C_surf_floor

        The corresponding state vector has the following entries:
        [T_room, T_surf_wall, T_op, T_mass, T_surf_floor, T_hp,ret]

        Parameters
        ----------
        x : list
            state vector
                - x[0] : Interior air temperature [degC]
                - x[1] : Wall inside surface temperature [degC]
                - x[2] : Operative room temperature (not calculated here - see simulator.get_next_state)
                - x[3] : Building mass temperature [degC]
                - x[4] : Floor surface temperature [degC]
                - x[5] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 

        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed total gains [W]
                - p[2] : Precomputed internal gains [W]
                - p[3] : Precomputed solar gains [W]
        
        Returns
        -------
        List of right hand side equations
            - rhs[0] of dT_room/dt        : interior air temperature [degC/s]
            - rhs[1] of dT_surf_wall/dt   : wall inside surface temperature [degC/s]
            - rhs[2] placeholder for operative temperature 
            - rhs[3] of dT_mass/dt        : building mass temperature [degC/s]
            - rhs[4] of dT_surf_floor/dt  : floor surface temperature [degC/s]
            - rhs[4] of dT_hp_ret/dt      : return flow temperature [degC/s]
        
        Notes
        -----
        This building model is based on the ISO13790 which consists of a 5R1C model. One additional capacity to represent the thermal mass of the
        heat supply system and one resistance to calculate the heatflow from the heat supply system to the building are used.
        Furthermore capacities for the room air, inside wall surface and floor surface are added, resulting in a 7R5C thermal network as shown in the following figure:
        
        .. image:: ../graphics/7R5C_network.svg
           :width: 100 %
        
        The internal heatgains by occupants and appliances as well as the solar heat gains are distributed to the temperature nodes according to ISO13790.
        The capacties of the building envelope and the inside wall surfaces is calculated according the ISO52016 6.5.7.2 class D 
        for a wall with evenly distributed thermal mass.

        The ordinary differential equations for the 7R5C calculation method are:
        
        :math:`\dot{T}_{room} = 1 / C_{air} \cdot (\dot{Q}_{gain2air} + \dot{Q}_{surf2air}^{floor} - \dot{Q}_{ve} - \dot{Q}_{air2surf}^{wall})`
        :math:`\dot{T}_{surf}^{wall} = 1 / C_{surf}^{wall} \cdot (\dot{Q}_{gain2surf}^{wall} + \dot{Q}_{air2surf}^{wall} - \dot{Q}_{tr,light} - \dot{Q}_{tr,int})`
        :math:`\dot{T}_{mass} = 1 / C_{bldg,heavy} \cdot (\dot{Q}_{gain2mass} + \dot{Q}_{tr,int} - \dot{Q}_{tr,ext})`
        :math:`\dot{T}_{surf}^{floor} = 1 / C_{surf}^{floor} \cdot (\dot{Q}_{gain2surf}^{floor} + \dot{Q}_{tr,floor} - \dot{Q}_{surf2air}^{floor})`
        :math:`\dot{T}_{HP} = 1 / (C_{water} - C_{surf}^{floor}) \cdot (\dot{Q}_{hp} - \dot{Q}_{tr,floor}))`

        with

        :math:`\dot{Q}_{gain2air} = 0.5 \cdot \dot{Q}_{gain,int}`
        :math:`\dot{Q}_{gain2surf} = (1 - (A_{mass} / A_{surf}) - H_{tr,light} / (9.1 \cdot A_{surf})) \cdot (0.5 \cdot \dot{Q}_{gain,int} + \dot{Q}_{gain,sol})`
        :math:`\dot{Q}_{gain2mass} = (A_{mass} / A_{surf}) \cdot (0.5 \cdot \dot{Q}_{gain,int} + \dot{Q}_{gain,sol})`
        :math:`\dot{Q}_{gain2surf}^{wall} = (1 - A_{floor} / A_{surf}) \cdot \dot{Q}_{gain2surf}`
        :math:`\dot{Q}_{gain2surf}^{floor} = A_{floor} / A_{surf} \cdot \dot{Q}_{gain2surf}`

        :math:`\dot{Q}_{hp} = \dot{m} \cdot c_{P,water} \cdot (T_{hp,sup} - T_{hp,ret})`
        :math:`\dot{Q}_{ve} = H_{ve} \cdot (T_{room} - T_{amb})`
        :math:`\dot{Q}_{air2surf}^{wall} = H_{air2surf}^{wall} \cdot (T_{room} - T_{surf}^{wall})`
        :math:`\dot{Q}_{surf2air}^{floor} = H_{con_floor} \cdot (T_{surf}^{floor} - T_{room})`
        :math:`\dot{Q}_{tr,light} = H_{tr,light} \cdot (T_{surf}^{wall} - T_{amb})`
        :math:`\dot{Q}_{tr,int} = H_{tr,int} \cdot (T_{surf}^{wall} - T_{mass})`
        :math:`\dot{Q}_{tr,ext} = H_{tr,ext} \cdot (T_{mass} - T_{amb})`
        :math:`\dot{Q}_{tr,floor} = H_{tr,floor} \cdot (T_{hp,ret} - T_{surf}^{floor})`

        :math:`H_{tr,int}     = h_{tr,int} \cdot A_{mass}`
        :math:`1 / H_{tr,ext} = 1 / H_{tr,heavy} - 1 / H_{tr,int}`

        The operative room temperature for this model is calculated in the get_next_state function according to the following equation:

        :math:`T_{surf} = A_{surf}^{floor} / A_{surf} \cdot T_{surf}^{floor} + (1 - A_{surf}^{floor} / A_{surf})  \cdot T_{surf}^{wall}`
        :math:`T_{op} = 0.3 \cdot T_{room} + 0.7 \cdot T_{surf}`

        '''
        T_room       = x[0] # [degC]
        T_surf_wall  = x[1] # [degC]
        # T_op       = x[2] # [degC] calculated after solving ODEs
        T_mass       = x[3] # [degC]
        T_surf_floor = x[4] # [degC]
        T_hp_ret     = x[5] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]
        Qdot_int   = p[2] # [W]
        Qdot_sol   = p[3] # [W]


        Qdot_gain2air = 0.5 * Qdot_int # [W]
        Qdot_gain2mass = self.params['A_mass']/self.params['A_surf'] * (0.5 * Qdot_int + Qdot_sol) # [W]
        Qdot_gain2surf = (1 - (self.params['A_mass'] / self.params['A_surf']) - (self.params['H_tr_light']/(9.1 * self.params['A_surf']))) * (0.5 * Qdot_int + Qdot_sol) # [W]

        f_floor = self.params['area_floor'] / self.params['A_surf'] # factor of floor area related to total surface area

        Qdot_gain2surf_floor = f_floor * Qdot_gain2surf
        Qdot_gain2surf_wall = (1-f_floor) * Qdot_gain2surf

        H_air2surf_wall = h_air2surf * self.params['A_surf'] * (1 - f_floor) # [W/K]
        H_tr_int = h_tr_int * self.params['A_mass'] # [W/K] 
        H_tr_ext = 1 / ((1/self.params['H_tr_heavy']) - (1/H_tr_int))

        Qdot_tr_int = H_tr_int * (T_surf_wall - T_mass) # [W]
        Qdot_tr_ext = H_tr_ext * (T_mass - T_amb) # [W]

        Qdot_hvac = self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
        Qdot_air2surf_wall = H_air2surf_wall * (T_room - T_surf_wall)
        Qdot_surf2air_floor = self.params['H_con_floor'] * (T_surf_floor - T_room)
        Qdot_tr_floor = self.params['H_tr_floor'] * (T_hp_ret - T_surf_floor)

        rhs  = np.zeros(6)
        rhs[0] = 1 / self.params['C_air'] * (Qdot_gain2air + Qdot_surf2air_floor - self.params['H_ve'] * (T_room - T_amb) - Qdot_air2surf_wall)
        rhs[1] = 1 / self.params['C_surf_wall'] * (Qdot_gain2surf_wall + Qdot_air2surf_wall - self.params['H_tr_light'] * (T_surf_wall - T_amb) - H_tr_int * (T_surf_wall - T_mass))
        # rhs[2] placeholder for T_op 
        rhs[3] = 1 / self.params['C_bldg_heavy'] * (Qdot_gain2mass + Qdot_tr_int - Qdot_tr_ext) # T_mass/dt [degC/s]
        rhs[4] = 1 / self.params['C_surf_floor'] * (Qdot_gain2surf_floor + Qdot_tr_floor - Qdot_surf2air_floor)
        rhs[5] = 1 / (self.params['C_water']) * (Qdot_hvac - Qdot_tr_floor) # T_hp_ret/dt [degC/s]

        return rhs   



    def calc_casadi(self, x, u, p):
        ''' Chooses and calls the casadi building model class depending on selected method.
        
        Parameters
        ----------
        t : not used - but needed for scipy.integrate.solve_ivp() which calls this function

        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations for (casadi.casadi.SX) according to selected building model
        '''

        if self.method == '2R2C':
            return self.calc_2r2c_casadi(x, u, p)
        elif self.method == '4R3C':
            return self.calc_4r3c_casadi(x, u, p)
        elif self.method == '5R4C':
            return self.calc_5r4c_casadi(x, u, p)
        else:
            print("Casadi calculation method does not exist")       

    def calc_2r2c_casadi(self, x, u, p):
        ''' This function contains the same building model as the fuction calc_2r2c.
        Here the casadi framework is used to use this model in a MPC controller.
        Check out calc_2r2c for details about the model.
        
        Parameters
        ----------
        t : not used - but needed for scipy.integrate.solve_ivp() which calls this function

        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations for (casadi.casadi.SX)
            - rhs[0] of dT_R/dt   : room temperature [degC/s]
            - rhs[1] of dT_RL/dt  : return flow temperature [degC/s]
        '''

        T_room     = x[0] # [degC]
        T_hp_ret   = x[1] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = cas.SX.sym("rhs",2)
        rhs[0] = 1/self.params['C_bldg'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) - self.params['H_ve_tr'] * (T_room - T_amb)) # T_room   [degC/s]
        rhs[1] = 1/self.params['C_water']* (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) - self.params['H_rad_con'] * (T_hp_ret - T_room))    # T_hp_ret [degC/s]
        return rhs

    def calc_4r3c_casadi(self, x, u, p):
        ''' This function contains the same building model as the fuction calc_4r3c.
        Here the casadi framework is used to use this model in a MPC controller.
        Check out calc_4r3c for details about the model.
        
        Parameters
        ----------
        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Wall temperature [degC]
                - x[2] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations for (casadi.casadi.SX)
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_wall/dt   : wall temperature [degC/s]
            - rhs[2] of dT_hp_ret/dt : return flow temperature [degC/s]
        '''

        T_room     = x[0] # [degC]
        T_wall     = x[1] # [degC]
        T_hp_ret   = x[2] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = cas.SX.sym("rhs",3)

        rhs[0] = 1 / self.params['C_zone'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) \
                - 2 * self.params['H_tr'] * (T_room - T_wall) - self.params['H_ve'] * (T_room - T_amb)) # T_room [degC/s]
        rhs[1] = 1 / self.params['C_wall'] * (2 * self.params['H_tr'] * (T_room - T_wall) \
                - 2 * self.params['H_tr'] * (T_wall - T_amb))     # T_wall [degC/s]
        rhs[2] = 1 / self.params['C_water'] * (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) \
                - self.params['H_rad_con'] * (T_hp_ret - T_room)) # T_hp_ret [degC/s]
        return rhs

    def calc_5r4c_casadi(self, x, u, p):
        ''' This function contains the same building model as the fuction calc_5r4c.
        Here the casadi framework is used to use this model in a MPC controller.
        Check out calc_5r4c for details about the model.
        
        Parameters
        ----------
        x : list
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Interior temperature [degC]
                - x[2] : Wall temperature [degC]
                - x[3] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 

        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations for (casadi.casadi.SX)
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_int/dt    : interior temperature [degC/s]
            - rhs[2] of dT_wall/dt   : wall temperature [degC/s]
            - rhs[3] of dT_hp_ret/dt : return flow temperature [degC/s]
        '''
        
        T_room     = x[0] # [degC]
        T_int      = x[1] # [degC]
        T_wall     = x[2] # [degC]
        T_hp_ret   = x[3] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = cas.SX.sym("rhs",4)
        rhs[0] = 1 / self.params['C_air'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) - self.params['H_int'] * (T_room - T_int) \
                - 2 * self.params['H_tr'] * (T_room - T_wall) - self.params['H_ve'] * (T_room - T_amb)) # T_room [degC/s]
        rhs[1] = 1 / self.params['C_int'] * self.params['H_int'] * (T_room - T_int) # T_int [degC/s]
        rhs[2] = 1 / self.params['C_wall'] * (2 * self.params['H_tr'] * (T_room - T_wall) \
                - 2 * self.params['H_tr'] * (T_wall - T_amb))     # T_wall [degC/s]
        rhs[3] = 1 / self.params['C_water'] * (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) \
                - self.params['H_rad_con'] * (T_hp_ret - T_room)) # T_hp_ret [degC/s]

        return rhs

    def calc_comfort_dev(self, T_room, timestep):
        """ Calculation of the comfort deviation of the building between
        set comfort levels and computed indoor temperature.
        
        Parameters
        ----------
        T_room : numpy.ndarray
            :math:`T_{room}` : Room air temperature  [°C]
            
        timestep : float
            Duration of the integration time step [s]

        Returns
        -------
        dict
            - dev_neg_sum  (float) : sum of negative deviations  [Kh]
            - dev_neg_max  (float) : maximum negative deviation  [K]
            - dev_pos_sum  (float) : sum of positive deviations  [Kh]
            - dev_pos_max  (float) : maximum positive deviation  [K]
        """

        isteps = len(T_room) # number of intermediate integration steps
        dev_neg = np.max(np.column_stack((self.T_room_set_lower * np.ones(isteps) - np.reshape((T_room), (isteps,)), np.zeros(isteps))), 1) 
        dev_pos = np.max(np.column_stack((np.reshape((T_room),(isteps,)) - self.T_room_set_upper * np.ones(isteps), np.zeros(isteps))), 1)    

        return {'dev_neg_sum'  : sum(dev_neg)/isteps * timestep/3600,
                'dev_neg_max'  : max(dev_neg),
                'dev_pos_sum'  : sum(dev_pos)/isteps* timestep/3600,
                'dev_pos_max'  : max(dev_pos)} 


    def print_params(self):
        """ This function prints information about the selected building.
        """
        print(f"Building name: {self.params['name']}")
        print(f"Floor area: {self.params['area_floor']}  m^2")
        print(f"Overall heat transfer coefficient: {self.params['H_ve_tr'] :.2f}  W/K")
        print(f"Heat capacity of the building: {self.params['C_bldg']/1000 :.2f}  kJ/K")
        print('Heat distribution system: Under Floor Heating')
        print(f"Radiative and convective heat transfer coefficient: {self.params['H_rad_con'] :.2f}  W/K")
        print(f"Heat capacity of the water: {self.params['C_water']/1000 :.2f}  kJ/K")


    def plot_step_response(self):
        """ This function plots the step response of the building envelope (excluding the thermal mass of the heating system) ,
        i. e. the thermal response of the room temperature to a step in ambient temperature from 0 to 20 degC.
        """
        import matplotlib.pyplot as plt

        # initialize heat pump and building model
        hp_model = model_hvac.Heatpump_AW(mdot_HP = 0.25)
        timestep = 3600 # sec
        sim = simulator.Model_simulator(bldg_model = self,
                                        hp_model   = hp_model,
                                        timestep   = timestep)

        # save original parameters
        C_water_original = self.params['C_water']
        H_rad_con_original = self.params['H_rad_con']
        H_con_floor_original = self.params['H_con_floor']
        H_tr_floor_original = self.params['H_tr_floor']

        # decouple building from heat supply system
        self.params['C_water'] = 0.00001 
        self.params['H_con_floor'] = 0.00001
        self.params['H_tr_floor'] = 0.00001
        self.params['H_rad_con'] = 0.00001

        timeconstant = self.params['C_bldg']/self.params['H_ve_tr'] # sec
        #print(timeconstant/3600)
        time = np.arange(0,10*timeconstant/timestep, 1)

        # initialize states dictionary
        state_keys = self.state_keys
        x_init = {key : 0 for key in state_keys}
        
        # initialize disturbance vector
        p = pd.DataFrame(index = time)
        p['T_amb'] = 20
        p['T_amb'][0] = 0
        p['Qdot_gains'], p['Qdot_int'], p['Qdot_sol'] = 0, 0, 0
        
        # simulate rule based
        results =  sim.simulate(x_init = x_init,p = p)
        T_room = results['states']['T_room']

        # plot step response
        fig, ax = plt.subplots()
        ax.plot(time, p['T_amb'], label = 'T_amb')
        ax.plot(time, T_room[0:len(time)-1], label = 'T_room')
        fig.legend(bbox_to_anchor=[0.5, 0.95], loc = 'center', ncol=2)
        ax.set_ylabel('Temperature [degC]')
        ax.set_xlabel('Time [h]')

        # reset parameters to original values
        self.params['C_water'] = C_water_original 
        self.params['H_rad_con'] = H_rad_con_original
        self.params['H_con_floor'] = H_con_floor_original
        self.params['H_tr_floor'] = H_tr_floor_original


if __name__ == "__main__":
    # load example building data
    from data.buildings.sfh_58_68_geg import sfh_58_68_geg

    # Initialize the building model
    building = sfh_58_68_geg
    bldg_model = Building(params    = building, # More example buildings can be found in data/buildings/.
                          mdot_hp   = 0.25,          # Massflow of the heat supply system. [kg/s]
                          method    = '5R4C',        # For available calculation methods see: model_buildings.py.
                          verbose   = True)
