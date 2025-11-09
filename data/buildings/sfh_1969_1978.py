''' 
Single family house
Construction period 1969 - 1978
'''



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.06.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1969_1978_0_soc = {'H_ve': 88,            # [W/K]
                       'H_tr': 493,           # [W/K]
                       'H_tr_light' : 95.8 + 6, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] Masonry - medium (Mauerwerk - mittel) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 173.2,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1969_1978_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_1969_1978_0_soc['T_offset'] = 6     # [K]
sfh_1969_1978_0_soc['T_amb_lim'] = 20   # [°C]
sfh_1969_1978_0_soc['mdot_hp'] = 0.25

# Window properties
w_east = {'area': 5.0, 'tilt': 90, 'azimuth': 90, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 16.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 5.0, 'tilt': 90, 'azimuth': 270, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 7.6, 'tilt': 90, 'azimuth': 0, 'g_value': 0.75, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1969_1978_0_soc['windows'] = windows

# Geolocation
sfh_1969_1978_0_soc['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.SFH.06.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1969_1978_1_enev = {'H_ve': 88,            # [W/K]
                        'H_tr': 198,           # [W/K]
                        'H_tr_light' : 44.5 + 2.6, # [W/K]
                        'c_bldg': 45,          # [Wh/(m^2K)] Masonry - medium (Mauerwerk - mittel) according to DIN EN ISO 13790 - Table 12
                        'area_floor': 173.2,   # [m^2]
                        'height_room': 2.5,    # [m]
                        'name': 'sfh_1969_1978_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_1969_1978_1_enev['T_offset'] = -6   # [K]
sfh_1969_1978_1_enev['T_amb_lim'] = 20  # [°C]
sfh_1969_1978_1_enev['mdot_hp'] = 0.29

# Window properties
w_east = {'area': 5.0, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 16.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 5.0, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 7.6, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1969_1978_1_enev['windows'] = windows

# Geolocation
sfh_1969_1978_1_enev['position'] = {'lat': 48.0252,
                                    'long': 7.7184,
                                    'altitude': 207,
                                    'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.06.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1969_1978_2_kfw = {'H_ve': 74,            # [W/K]
                       'H_tr': 113,           # [W/K]
                       'H_tr_light' : 27.4 + 1.6, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] Masonry - medium (Mauerwerk - mittel) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 173.2,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1969_1978_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_1969_1978_2_kfw['T_offset'] = -9    # [K]
sfh_1969_1978_2_kfw['T_amb_lim'] = 20   # [°C]
sfh_1969_1978_2_kfw['mdot_hp'] = 0.23

# Window properties
w_east = {'area': 5.0, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 16.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 5.0, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 7.6, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1969_1978_2_kfw['windows'] = windows

# Geolocation
sfh_1969_1978_2_kfw['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}
