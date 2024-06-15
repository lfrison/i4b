
bldg (dict) : 
    containing easily accissible building parameters as specified in buildings.py:
    - H_ve   : :math:`H_{ve}`      : Heat transfer coefficient for ventilation (indoors --> ambient) [W/K]
    - H_tr   : :math:`H_{tr}`      : Heat transfer coefficient for transmission (indoors --> ambient) [W/K]
    - H_tr_light :  :math:`H_{tr,light}` : Heat transfer coefficient for transmission of light building components (windows and doors) (indoors --> ambient) [W/K] only needed for 5r2c calc method (ISO13790) 
    - H_int  : :math:`H_{int}`     : Heat transfer coefficient for convection and radiation (interior surfaces --> indoor air) [W/K]
    - c_bldg : :math:`c_{bldg}`    : Specific heat capacity of the bldg [Wh/m^2/K] *
    - area_floor : :math:`A_{floor}` : Conditioned floor area  [m²]
    - height_room : :math:`h_{room}` : Average height of the heated zone [m³]
    - T_offset   : :math:`T_{offset}` : optional parameter, only used for heating curve control. Constant offset of the heating curve [K]
    - T_amb_lim  : :math:`T_{amb,lim}` : optional parameter, only used for heating curve control [°C], For T_amb > T_amb_lim, the heating is switched off.
    - windows (list of dicts) :
        each dict contains:
        - area:     absolute window area [m²]
        - tilt:     tilt angle [degree]
        - azimuth:  azimuth angle, 0 = North, 180 = South [degree]
        - g-value:  total solar heat gain factor [-]
        - c_frame:  fraction of window that is opaque due to the frame [-]
        - c_shade:  shading factor due to external influences, e.g. trees [-]
    - position (dict) :
        - lat:      latitutde [degree]
        - long:     longitude [degree]
        - altitude: altitude above sealevel [m]
        - timezone: Timezone as defined in: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

* values taken from DIN EN ISO 13790 - Table 12: Specific heat capacity of the bldg [Wh/m^2/K]
very light: 22
light     : 30
medium    : 45
heavy     : 72
very heavy: 102