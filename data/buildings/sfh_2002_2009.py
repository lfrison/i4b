''' 
Single family house
Construction period 2002 - 2009
'''



# ------------------------------------------------------------
# State of construction
# TABULA source DE.N.SFH.10.Gen.ReEx.001.001
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_2002_2009_0_soc = {'H_ve': 62,            # [W/K]
                       'H_tr': 152,           # [W/K]
                       'H_tr_light' : 39.6 + 4, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] zweischaliges Mauerwerk - mittel nach DIN EN ISO 13790 - Tabelle 12
                       'area_floor': 146.5,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_2002_2009_0_soc'}

# Heating system properties (for target room temperature of 20°C)
sfh_2002_2009_0_soc['T_offset'] = -8    # [K]
sfh_2002_2009_0_soc['T_amb_lim'] = 20   # [°C]
sfh_2002_2009_0_soc['mdot_hp'] = 0.30

# Window properties
w_east = {'area': 3.9, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 17.3, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.9, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.1, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_2002_2009_0_soc['windows'] = windows

# Geolocation
sfh_2002_2009_0_soc['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# Standard refurbishment (EnEV 2009/2014/2016)
# TABULA source DE.N.SFH.10.Gen.ReEx.001.002
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_2002_2009_1_enev = {'H_ve': 62,            # [W/K]
                        'H_tr': 128,           # [W/K]
                        'H_tr_light' : 36.8 + 2.6, # [W/K]
                        'c_bldg': 45,          # [Wh/(m^2K)] zweischaliges Mauerwerk - mittel nach DIN EN ISO 13790 - Tabelle 12
                        'area_floor': 146.5,   # [m^2]
                        'height_room': 2.5,    # [m]
                        'name': 'sfh_2002_2009_1_enev'}

# Heating system properties (for target room temperature of 20°C)
sfh_2002_2009_1_enev['T_offset'] = -9   # [K]
sfh_2002_2009_1_enev['T_amb_lim'] = 20  # [°C]
sfh_2002_2009_1_enev['mdot_hp'] = 0.28

# Window properties
w_east = {'area': 3.9, 'tilt': 90, 'azimuth': 90, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 17.3, 'tilt': 90, 'azimuth': 180, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.9, 'tilt': 90, 'azimuth': 270, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.1, 'tilt': 90, 'azimuth': 0, 'g_value': 0.6, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_2002_2009_1_enev['windows'] = windows

# Geolocation
sfh_2002_2009_1_enev['position'] = {'lat': 48.0252,
                                    'long': 7.7184,
                                    'altitude': 207,
                                    'timezone': 'Europe/Berlin'}



# ------------------------------------------------------------
# KfW refurbishment
# TABULA source DE.N.SFH.10.Gen.ReEx.001.003
# ------------------------------------------------------------

# Geometry and meterial properties
sfh_2002_2009_2_kfw = {'H_ve': 62,            # [W/K]
                       'H_tr': 89,            # [W/K]
                       'H_tr_light' : 22.6 + 1.6, # [W/K]
                       'c_bldg': 45,          # [Wh/(m^2K)] zweischaliges Mauerwerk - mittel nach DIN EN ISO 13790 - Tabelle 12
                       'area_floor': 146.5,   # [m^2]
                       'height_room': 2.5,    # [m]
                       'name': 'sfh_2002_2009_2_kfw'}

# Heating system properties (for target room temperature of 20°C)
sfh_2002_2009_2_kfw['T_offset'] = -10   # [K]
sfh_2002_2009_2_kfw['T_amb_lim'] = 20   # [°C]
sfh_2002_2009_2_kfw['mdot_hp'] = 0.25

# Window properties
w_east = {'area': 3.9, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_south = {'area': 17.3, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_west = {'area': 3.9, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
w_north = {'area': 3.1, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.6}
windows = [w_east, w_south, w_west, w_north]
sfh_2002_2009_2_kfw['windows'] = windows

# Geolocation
sfh_2002_2009_2_kfw['position'] = {'lat': 48.0252,
                                   'long': 7.7184,
                                   'altitude': 207,
                                   'timezone': 'Europe/Berlin'}
