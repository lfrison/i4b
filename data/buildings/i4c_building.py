#%% i4c building - Two family house, kfw 40+
from src.models.model_buildings import C_AIR_SPEC, RHO_AIR,C_INT_SPEC

# Geometry and material properties
i4c = {"H_ve": 88.93,
        "H_tr": 192.88, # including thermal bridges
        "area_floor": 393.32,
        "height_room": 2.37, 
        "name" : 'i4c'
        }

# Calculate specific heat capacities and derived totals
volume_air = i4c['height_room'] * i4c['area_floor']
C_wall = 24845 * 3600 # J/K
C_air = C_AIR_SPEC * RHO_AIR * volume_air # J/K
C_int = C_INT_SPEC * i4c['area_floor'] # J/K
C_zone = C_air + C_int # J/K
C_bldg = C_wall + C_zone # J/K

i4c['c_bldg'] = C_bldg / i4c['area_floor'] / 3600 # Wh/m^2/K

# heating system properties
i4c['T_offset']  = 0#- 5.576
i4c['T_amb_lim'] = 20 

# windows
skylight = {'area': 1.89, 'tilt': 15, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.9}
w_north = {'area': 15.38, 'tilt': 90, 'azimuth': 0, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.9}
w_east = {'area': 29.00, 'tilt': 90, 'azimuth': 90, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.9}
w_south = {'area': 39.63, 'tilt': 90, 'azimuth': 180, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.9}
w_west = {'area': 14.23, 'tilt': 90, 'azimuth': 270, 'g_value': 0.5, 'c_frame': 0.3, 'c_shade': 0.9}
windows = [skylight, w_north, w_east, w_south, w_west]
i4c['windows'] = windows

# Geolocation
i4c['position'] = {'lat': 48.0252,
                   'long': 7.7184,
                   'altitude': 207,
                   'timezone' : 'Europe/Berlin'}
#%% Calculation of heat transmission through light building elements

''' TRANSMISSION HEAT LOSSES
According to ISO 13790, transmission heat losses through light (doors, windows)
and heavy building elements are handled individually. We first calculate the
specific losses for light elements and subtract them from the total specific
losses to obtain the losses through heavy elements. '''

# Geometry - parameters from EnEV certificate

a_surf = 836.23 # Total surface area of the building [m^2]
a_cond = 393.32 # Conditioned floor area [m^2]
v_cond = 934.13 # Heated air volume [m^3]

# Areas of all 'light' building elements [m^2]
a_win = 100.11 # Total window area (Page 6 EnEv) 
a_win_roof = 1.89 # Area of the skylight (Pos. 2)
a_win_h = a_win - a_win_roof # Area of all horizontal windows
a_door = 2.62 + 4.52 # Area of both doors (Pos. 19 & 39)

# Material
# U-values of all light building elements [W/(m^2K)]
u_win_roof = 1.0 # u-value of the skylight
u_win_h = 0.76 # u-value of all horizontal windows
u_door = 1 # u-value of both doors

# Heat losses due to thermal bridges
H_tb_light = 6.88 # heat losses - thermal bridges for light building elements

# Area of all light building elements
a_light = a_win_roof + a_win_h + a_door

# Overall u-value of light building elements
u_light = ((u_win_roof * a_win_roof + u_win_h * a_win_h + u_door * a_door) / a_light)

# Specific heat transmission of light building elements [W/K]
H_tr_w = u_light * a_light + H_tb_light

i4c['H_tr_light'] = H_tr_w