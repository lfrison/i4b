''' 
Single family house
Construction period 1958 - 1968
'''



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.05.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1958_1968_0_soc = {'H_ve': 62,            # [W/K]
                       'H_tr': 496,           # [W/K]
                       'H_tr_light' : 75.9 + 6.3, # [W/K]
                       'c_bldg': 30,          # [Wh/(m^2K)] Hollow brick - light (Hohlziegel - leicht) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 121.2,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1958_1968_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_1958_1968_0_soc['T_offset'] = 12    # [K]
sfh_1958_1968_0_soc['T_amb_lim'] = 20   # [°C]
sfh_1958_1968_0_soc['mdot_hp'] = 0.24

# Window properties
w_east = {'area': 5.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 6.3, 'tilt': 90, 'azimuth': 180, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 8.9, 'tilt': 90, 'azimuth': 270, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 4.1, 'tilt': 90, 'azimuth': 0, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1958_1968_0_soc['windows'] = windows

# Geolocation
sfh_1958_1968_0_soc['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.05.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1958_1968_1_enev = {'H_ve': 62,            # [W/K]
                        'H_tr': 205,           # [W/K]
                        'H_tr_light' : 35.2 + 2.7, # [W/K]
                        'c_bldg': 30,          # [Wh/(m^2K)] Hollow brick - light (Hohlziegel - leicht) according to DIN EN ISO 13790 - Table 12
                        'area_floor': 121.2,   # [m^2]
                        'height_room': 2.5,    # [m]
                        'name': 'sfh_1958_1968_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_1958_1968_1_enev['T_offset'] = -3   # [K]
sfh_1958_1968_1_enev['T_amb_lim'] = 20  # [°C]
sfh_1958_1968_1_enev['mdot_hp'] = 0.25

# Window properties
w_east = {'area': 5.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 6.3, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 8.9, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 4.1, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1958_1968_1_enev['windows'] = windows

# Geolocation
sfh_1958_1968_1_enev['position'] = {'lat': 48.0252,
                                    'long': 7.7184,
                                    'altitude': 207,
                                    'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.05.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1958_1968_2_kfw = {'H_ve': 52,            # [W/K]
                       'H_tr': 102,           # [W/K]
                       'H_tr_light' : 21.7 + 1.7, # [W/K]
                       'c_bldg': 30,          # [Wh/(m^2K)] Hollow brick - light (Hohlziegel - leicht) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 121.2,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1958_1968_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_1958_1968_2_kfw['T_offset'] = -8    # [K]
sfh_1958_1968_2_kfw['T_amb_lim'] = 20   # [°C]
sfh_1958_1968_2_kfw['mdot_hp'] = 0.23

# Window properties
w_east = {'area': 5.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 6.3, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 8.9, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 4.1, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1958_1968_2_kfw['windows'] = windows

# Geolocation
sfh_1958_1968_2_kfw['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}
