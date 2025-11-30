# -*- coding: utf-8 -*-
"""
Shared physical constants for the project.
"""

RHO_WATER = 997      # Density water [kg/m3]
RHO_AIR = 1.225      # Density air [kg/m3]
C_WATER_SPEC = 4181  # Spec. heat capacity of water [J/kg/K]
C_AIR_SPEC = 1.005   # Spec. heat capacity of air [J/kg/K]

# Building constants
C_INT_SPEC = 10000    # [J/m^2/K]
H_UFH_SPEC = 4.4      # [W/m^2/K]
H_UFH_SURF_SPEC = 10.8 # W/m^2/K
V_TS_SPEC  = 5        # [l/m^2]
V_UFH_SPEC = 1.5      # [l/m^2]
R_SI       = 0.13     # [m^2*K/W]
H_TR_INT   = 9.1      # [W/(m^2K)] heattransfer coefficient between building mass and indoor surface ISO13790 12.2.2
H_AIR2SURF = 3.45     # [W/(m^2K)] heattransfer coefficient between indoor surface and indoor air ISO13790 7.2.2.2
C_P_SCREED = 1000     # specific heat capacity of screed [J/(kgK)] source: https://www.schweizer-fn.de/stoff/wkapazitaet/wkapazitaet_baustoff_erde.php
D_SCREED = 50         # thickness of screed for UFH [mm] source: https://www.kesselheld.de/heizestrich/
RHO_SCREED = 2000     # dry bulk density of screed [kg/m^3] source: https://www.profibaustoffe.com/wp-content/files/TD_2059_2056_ESTRICH-CT-C20-F4-E225-MIT-FASERN_041119.pdf
