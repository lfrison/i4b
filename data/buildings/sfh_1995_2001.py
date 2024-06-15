''' 
Single family house
Construction period 1995 - 2001
'''



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.09.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1995_2001_0_soc = {'H_ve': 62,            # [W/K]
                       'H_tr': 197,           # [W/K]
                       'H_tr_light' : 61.7 + 4, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] zweischaliges Mauerwerk - mittel nach DIN EN ISO 13790 - Tabelle 12
                       'area_floor': 121.9,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1995_2001_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_1995_2001_0_soc['T_offset'] = -3    # [K]
sfh_1995_2001_0_soc['T_amb_lim'] = 20   # [°C]
sfh_1995_2001_0_soc['mdot_hp'] = 0.24

# Window properties
w_east = {'area': 3.6, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 5.0, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.6, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 20.3, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1995_2001_0_soc['windows'] = windows

# Geolocation
sfh_1995_2001_0_soc['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.SFH.09.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1995_2001_1_enev = {'H_ve': 62,            # [W/K]
                        'H_tr': 156,           # [W/K]
                        'H_tr_light' : 42.2 + 2.6, # [W/K]
                        'c_bldg': 45,          # [Wh/(m^2K)] zweischaliges Mauerwerk - mittel nach DIN EN ISO 13790 - Tabelle 12
                        'area_floor': 121.9,   # [m^2]
                        'height_room': 2.5,    # [m]
                        'name': 'sfh_1995_2001_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_1995_2001_1_enev['T_offset'] = -5   # [K]
sfh_1995_2001_1_enev['T_amb_lim'] = 20  # [°C]
sfh_1995_2001_1_enev['mdot_hp'] = 0.23

# Window properties
w_east = {'area': 3.6, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 5.0, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.6, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 20.3, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1995_2001_1_enev['windows'] = windows

# Geolocation
sfh_1995_2001_1_enev['position'] = {'lat': 48.0252,
                                    'long': 7.7184,
                                    'altitude': 207,
                                    'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.09.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_1995_2001_2_kfw = {'H_ve': 52,             # [W/K]
                       'H_tr': 87,            # [W/K]
                       'H_tr_light' : 26 + 1.6, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] zweischaliges Mauerwerk - mittel nach DIN EN ISO 13790 - Tabelle 12
                       'area_floor': 121.9,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_1995_2001_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_1995_2001_2_kfw['T_offset'] = -9    # [K]
sfh_1995_2001_2_kfw['T_amb_lim'] = 20   # [°C]
sfh_1995_2001_2_kfw['mdot_hp'] = 0.29

# Window properties
w_east = {'area': 3.6, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 5.0, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.6, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 20.3, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_1995_2001_2_kfw['windows'] = windows

# Geolocation
sfh_1995_2001_2_kfw['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}
