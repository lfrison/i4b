''' 
Single family house
Construction period 1979 - 1983
'''



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.07.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and material properties
sfh_1979_1983_0_soc = {'H_ve': 110,           # [W/K]
                       'H_tr': 364,           # [W/K]
                       'H_tr_light' : 116.1 + 6, # [W/K]
                       'c_bldg': 30,          # [Wh/(m^2K)] Light hollow brick - light (Leicht-Hohlziegel - leicht) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 215.6,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1979_1983_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_1979_1983_0_soc['T_offset'] = 0     # [K]
sfh_1979_1983_0_soc['T_amb_lim'] = 20   # [°C]
sfh_1979_1983_0_soc['mdot_hp'] = 0.27

# Window properties
w_east = {'area': 8.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 6.0, 'tilt': 90, 'azimuth': 180, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 8.7, 'tilt': 90, 'azimuth': 270, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.6, 'tilt': 90, 'azimuth': 0, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1979_1983_0_soc['windows'] = windows

# Geolocation
sfh_1979_1983_0_soc['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.SFH.07.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and material properties
sfh_1979_1983_1_enev = {'H_ve': 110,           # [W/K]
                        'H_tr': 161,           # [W/K]
                        'H_tr_light' : 35.1 + 2.6, # [W/K]
                        'c_bldg': 30,          # [Wh/(m^2K)] Light hollow brick - light (Leicht-Hohlziegel - leicht) according to DIN EN ISO 13790 - Table 12
                        'area_floor': 215.6,   # [m^2]
                        'height_room': 2.5,    # [m]
                        'name': 'sfh_1979_1983_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_1979_1983_1_enev['T_offset'] = -7   # [K]
sfh_1979_1983_1_enev['T_amb_lim'] = 20  # [°C]
sfh_1979_1983_1_enev['mdot_hp'] = 0.28

# Window properties
w_east = {'area': 8.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 6.0, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 8.7, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.6, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1979_1983_1_enev['windows'] = windows

# Geolocation
sfh_1979_1983_1_enev['position'] = {'lat': 48.0252,
                                    'long': 7.7184,
                                    'altitude': 207,
                                    'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.07.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and material properties
sfh_1979_1983_2_kfw = {'H_ve': 92,            # [W/K]
                       'H_tr': 84,            # [W/K]
                       'H_tr_light' : 21.6 + 1.6, # [W/K]
                       'c_bldg': 30,          # [Wh/(m^2K)] Light hollow brick - light (Leicht-Hohlziegel - leicht) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 215.6,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1979_1983_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_1979_1983_2_kfw['T_offset'] = -10   # [K]
sfh_1979_1983_2_kfw['T_amb_lim'] = 20   # [°C]
sfh_1979_1983_2_kfw['mdot_hp'] = 0.29

# Window properties
w_east = {'area': 8.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 6.0, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 8.7, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.6, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1979_1983_2_kfw['windows'] = windows

# Geolocation
sfh_1979_1983_2_kfw['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}
