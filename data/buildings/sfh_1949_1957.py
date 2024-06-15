''' 
Single family house
Construction period 1949 - 1957
'''


# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.04.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1949_1957_0_soc = {'H_ve': 57,            # [W/K]
                       'H_tr': 465,           # [W/K]
                       'H_tr_light' : 51.5 + 6, # [W/K]
                       'c_bldg': 72,          # [Wh/(m^2K)] zweischaliges Mauerwerk - schwer nach DIN EN ISO 13790 - Tabelle 12
                       'area_floor': 111.1,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1949_1957_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_1949_1957_0_soc['T_offset'] = 11    # [K]
sfh_1949_1957_0_soc['T_amb_lim'] = 20   # [°C]
sfh_1949_1957_0_soc['mdot_hp'] = 0.27

# Window properties
w_east = {'area': 3.2, 'tilt': 90, 'azimuth': 90, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 8.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.2, 'tilt': 90, 'azimuth': 270, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.3, 'tilt': 90, 'azimuth': 0, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1949_1957_0_soc['windows'] = windows

# Geolocation
sfh_1949_1957_0_soc['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.SFH.04.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1949_1957_1_enev = {'H_ve': 57,            # [W/K]
                        'H_tr': 171,           # [W/K]
                        'H_tr_light' : 23.9 + 2.6, # [W/K]
                        'c_bldg': 72,          # [Wh/(m^2K)] zweischaliges Mauerwerk - schwer nach DIN EN ISO 13790 - Tabelle 12
                        'area_floor': 111.1,   # [m^2]
                        'height_room': 2.5,    # [m]
                        'name': 'sfh_1949_1957_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_1949_1957_1_enev['T_offset'] = -4   # [K]
sfh_1949_1957_1_enev['T_amb_lim'] = 20  # [°C]
sfh_1949_1957_1_enev['mdot_hp'] = 0.24

# Window properties
w_east = {'area': 3.2, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 8.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.2, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.3, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1949_1957_1_enev['windows'] = windows

# Geolocation
sfh_1949_1957_1_enev['position'] = {'lat': 48.0252,
                                    'long': 7.7184,
                                    'altitude': 207,
                                    'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.04.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1949_1957_2_kfw = {'H_ve': 47,            # [W/K]
                       'H_tr': 75,            # [W/K]
                       'H_tr_light' : 14.7 + 1.6, # [W/K]
                       'c_bldg': 72,          # [Wh/(m^2K)] zweischaliges Mauerwerk - schwer nach DIN EN ISO 13790 - Tabelle 12
                       'area_floor': 111.1,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1949_1957_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_1949_1957_2_kfw['T_offset'] = -9    # [K]
sfh_1949_1957_2_kfw['T_amb_lim'] = 20   # [°C]
sfh_1949_1957_2_kfw['mdot_hp'] = 0.23

# Window properties
w_east = {'area': 3.2, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 8.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.2, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.3, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1949_1957_2_kfw['windows'] = windows

# Geolocation
sfh_1949_1957_2_kfw['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}
