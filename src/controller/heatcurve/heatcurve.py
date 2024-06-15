import numpy as np
import pandas as pd


class Heatingcurve:
    """ Model of a heating curve for different heat distribution systems in a building.

    Parameters
    ----------
    T_room_set : float
        set room temperature (Raumsolltemperatur) [degC]

    T_flow_nom : float
        nominal supply flow temperature (nominale Heizkreis Vorlauf-Temperatur) [degC]

    T_ret_nom : float
        nominal return flow temperature (nominale Heizkreis Rücklauf-Temperatur) [degC]

    T_amb_nom : float
        nominal ambient temperature (Normaußentemperatur) [degC] (location specific; -12.1 degC for Potsdam)

    T_amb_lim : float
        heating limit temperature (Heizgrenztemperatur) [degC]

    heatingexp : float
        heat distribution exponent (Heizkörperexponent), e. g.:
            - underfloor heating (Fußbodenheizung) n = 1,1 (default)
            - plate radiator (Plattenheizkörper) n = 1,20-1,30
            - tube radiator (Rohrheizkörper) n = 1,25
            - finned radiator (Rippenrohrheizkörper) n = 1,25
            - radiator (Heizkörper) n = 1,30
            - convector (Konvektoren) n = 1,25 – 1,45
    """
    def __init__(self,
                 T_room_set = 20,    # set room temperature [degC]
                 T_sup_nom = 35,     # nominal supply flow temperature for underfloor heating [degC]
                 T_ret_nom  = 20,    # nominal return temperature for underfloor heating [degC]
                 T_amb_nom  = -12.1, # design ambient temperature for Potsdam [degC]   
                 T_amb_lim  = 15,    # heating limit temperature [degC]
                 heatingexp = 1.1    # heat exponent for underfloor heating [-]
                ):
    
        self.T_room_set          = T_room_set
        self.T_sup_nom           = T_sup_nom
        self.T_ret_nom           = T_ret_nom
        self.T_amb_nom           = T_amb_nom
        self.T_amb_lim           = T_amb_lim
        self.heatingexp          = heatingexp

    def calc(self, T_amb, shift_T_lim = 0, offset_T_flow_nom = 0):
        """ Calculate supply and return flow temperature from heating curve.

        Parameters
        ----------
        T_amb : float or numpy array
            ambient temperature [degC]
        
        shift_T_lim: float
            shift of heating limit temperature [K], default = 0

        offset_T_flow_nom: float
            offset of supply and return flow temperature [K], default = 0
        
        Returns
        -------
        float or numpy array
            - supply flow temperature [degC]
            - return flow temperature [degC]

        Notes
        -----
        :math:`T_{sup,set} = T_{room,set} + ((T_{sup,nom}+T_{ret,nom})/2 - T_{room,set}) \cdot \dot{Q}_{rel}^{1/n} + (T_{sup,nom}-T_{ret,nom})/2 \cdot \dot{Q}_{rel}`
        :math:`T_{ret,set} = T_{sup,set} - \dot{Q}_{rel} \cdot (T_{sup,nom} - T_{ret,nom})`

        with

        :math:`\dot{Q}_{rel} = (T_{amb,lim}-T_{amb})/(T_{amb,lim}-T_{amb,nom})`

        n : heat exponent

        acc. to 
        `M. Laemmle <https://www.researchgate.net/publication/357211490_Performance_of_air_and_ground_source_heat_pumps_retrofitted_to_radiator_heating_systems_and_measures_to_reduce_space_heating_temperatures_in_existing_buildings>`_
        
        """

        T_amb_lim = self.T_amb_lim + shift_T_lim
        T_sup_nom = self.T_sup_nom + offset_T_flow_nom
        T_ret_nom = self.T_ret_nom + offset_T_flow_nom

        Qdot_rel = np.maximum(0,(T_amb_lim - T_amb) / (T_amb_lim - self.T_amb_nom))
        T_sup_set = self.T_room_set + ((T_sup_nom + T_ret_nom)/2 - self.T_room_set) * np.power(Qdot_rel, 1/self.heatingexp) + (T_sup_nom - T_ret_nom)/2 * Qdot_rel
        T_ret_set = T_sup_set - Qdot_rel * (T_sup_nom - T_ret_nom)

        return T_sup_set, T_ret_set


def tune_building_t_offset_and_mdot_hp(building, method='4R3C', t_offset_min=-15, t_offset_max=15, t_offset_step=1.0,
                                       mdot_hp_min=0.15, mdot_hp_max=0.35, mdot_hp_step=0.01, repo_filepath=''):
    '''
    Takes a building configuration struct and optimizes T_offset and mdot_hp by simply trying out different
    values in the given ranges with the given step widths. T_offset is optimized first, afterwards
    mdot_hp is finetuned with the best found value for T_offset.

    Parameters
    ----------

    building: string
        ISO3166-1 code of country to return random location from.
        Default is 'DE' (Germany).

    t_offset_min: str, optional
        Minimum T_offset for tuning loop.
        Default is -15.

    t_offset_max: str, optional
        Maximum T_offset for tuning loop.
        Default is 15.

    t_offset_step: str, optional
        Step width with which values ranging from t_offset_min to t_offset_max are tried.
        Default is 1.

    mdot_hp_min: str, optional
        Minimum mdot_hp for tuning loop.
        Default is 0.15.

    mdot_hp_max: str, optional
        Maximum mdot_hp for tuning loop.
        Default is 0.35.

    mdot_hp_step: str, optional
        Step width with which values ranging from mdot_hp_min to mdot_hp_max are tried.
        Default is 0.01.

    repo_filepath: str, optional
        Base filepath of the i4c repository in case the function
        is called from outside the repository.

    Returns
    -------

    Tuple (best_t_offset, best_mdot)
    '''
    from disturbances import load_weather, get_int_gains, get_solar_gains
    import model_buildings
    import model_hvac
    import simulator

    INT_GAIN_PROFILE = f'{repo_filepath}data/profiles/InternalGains/ResidentialDetached.csv'
    INITIAL_TEMP = 20 # °C
    SIMULATION_STEP = 3600 # 1h

    best_comfort_deviation = 100
    best_t_offset = 0
    best_mdot_hp = 0.25

    print(f'\n----- Tune T_offset & mdot_hp for building {building["name"]}')

    print('Load weather data and take first month')
    df_weather = load_weather(building['position']['lat'], building['position']['long'],
                              tz=building['position']['timezone'])
    df_weather = df_weather[:(31*24)]

    print('Create disturbances')
    df_int_gains = get_int_gains(time=df_weather.index, profile_path=INT_GAIN_PROFILE,
                                 bldg_area=building['area_floor'])
    df_solar_gains = get_solar_gains(weather=df_weather, bldg_params=building)
    df_qdot_gains = pd.DataFrame((df_solar_gains + df_int_gains['Qdot_tot']),
                                 columns=['Qdot_gains'])
    df_disturbances = pd.concat([df_weather['T_amb'], df_qdot_gains], axis=1)

    # Inner convenience function for simulation and calculation of
    # mean comfort deviation
    def _calc_comfort_deviation(bldg, mdot, meth, dist):
        model_bldg = model_buildings.Building(params=bldg, mdot_hp=mdot, method=meth)
        model_hp = model_hvac.Heatpump_AW(mdot_HP=mdot)
        sim = simulator.Model_simulator(bldg_model=model_bldg, hp_model=model_hp,
                                        timestep=SIMULATION_STEP)
        state = {key : INITIAL_TEMP for key in model_bldg.state_keys}
        res = sim.simulate(x_init=state, p=dist)
        return abs(res['states']['T_room'].mean() - INITIAL_TEMP)

    print(f'Find best T_offset (for mdot_hp = {best_mdot_hp})')
    t_offset_max += t_offset_step
    for t_offset in np.arange(t_offset_min, t_offset_max, t_offset_step):
        building['T_offset'] = t_offset
        comfort_deviation = _calc_comfort_deviation(building, best_mdot_hp,
                                                    method, df_disturbances)
        if comfort_deviation < best_comfort_deviation:
            best_comfort_deviation = comfort_deviation
            best_t_offset = t_offset
    print(f'Best T_offset = {best_t_offset} (with mdot_hp = {best_mdot_hp})')

    print(f'Find best mdot_hp (for T_offset = {best_t_offset})')
    mdot_hp_max += mdot_hp_step
    for mdot_hp in np.arange(mdot_hp_min, mdot_hp_max, mdot_hp_step):
        building['T_offset'] = best_t_offset
        comfort_deviation = _calc_comfort_deviation(building, mdot_hp,
                                                    method, df_disturbances)
        if comfort_deviation < best_comfort_deviation:
            best_comfort_deviation = comfort_deviation
            best_mdot_hp = mdot_hp
    print(f'Best mdot_hp = {best_mdot_hp} (with T_offset = {best_t_offset})')

    return best_t_offset, best_mdot_hp

