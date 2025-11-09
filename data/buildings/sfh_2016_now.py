''' 
Single family house
Construction period 2016 - now
'''



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.12.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_2016_now_0_soc = {'H_ve': 95,             # [W/K]
                      'H_tr': 142,            # [W/K]
                      'H_tr_light' : 46.2 + 3.4, # [W/K]
                      'c_bldg': 45,          # [Wh/(m^2K)] Masonry with ETICS - medium (Mauerwerk mit Wärmedämmverbundsystem - mittel) according to DIN EN ISO 13790 - Table 12
                      'area_floor': 186.8,    # [m^2]
                      'height_room': 2.5,     # [m]
                      'name': 'sfh_2016_now_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_2016_now_0_soc['T_offset'] = -8     # [K]
sfh_2016_now_0_soc['T_amb_lim'] = 20    # [°C]
sfh_2016_now_0_soc['mdot_hp'] = 0.22

# Window properties
w_east = {'area': 2.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 22.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 13.0, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.7, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_2016_now_0_soc['windows'] = windows

# Geolocation
sfh_2016_now_0_soc['position'] = {'lat': 48.0252,
                                  'long': 7.7184,
                                  'altitude': 207,
                                  'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.SFH.12.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_2016_now_1_enev = {'H_ve': 79,             # [W/K]
                       'H_tr': 136,            # [W/K]
                       'H_tr_light' : 46.2 + 3.4, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] Masonry with ETICS - medium (Mauerwerk mit Wärmedämmverbundsystem - mittel) according to DIN EN ISO 13790 - Table 12
                       'area_floor': 186.8,    # [m^2]
                       'height_room': 2.5,     # [m]
                       'name': 'sfh_2016_now_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_2016_now_1_enev['T_offset'] = -9    # [K]
sfh_2016_now_1_enev['T_amb_lim'] = 20   # [°C]
sfh_2016_now_1_enev['mdot_hp'] = 0.23

# Window properties
w_east = {'area': 2.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 22.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 13.0, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.7, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_2016_now_1_enev['windows'] = windows

# Geolocation
sfh_2016_now_1_enev['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.12.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_2016_now_2_kfw = {'H_ve': 71,              # [W/K]
                      'H_tr': 89,             # [W/K]
                      'H_tr_light' : 29.4 + 2.1, # [W/K]
                      'c_bldg': 45,          # [Wh/(m^2K)] Masonry with ETICS - medium (Mauerwerk mit Wärmedämmverbundsystem - mittel) according to DIN EN ISO 13790 - Table 12
                      'area_floor': 186.8,    # [m^2]
                      'height_room': 2.5,     # [m]
                      'name': 'sfh_2016_now_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_2016_now_2_kfw['T_offset'] = -12    # [K]
sfh_2016_now_2_kfw['T_amb_lim'] = 20    # [°C]
sfh_2016_now_2_kfw['mdot_hp'] = 0.30

# Window properties
w_east = {'area': 2.7, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 22.6, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 13.0, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.7, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_2016_now_2_kfw['windows'] = windows

# Geolocation
sfh_2016_now_2_kfw['position'] = {'lat': 48.0252,
                                  'long': 7.7184,
                                  'altitude': 207,
                                  'timezone': 'Europe/Berlin'}
