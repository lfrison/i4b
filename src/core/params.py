from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

from src.constants import (
    RHO_WATER, RHO_AIR, C_WATER_SPEC, C_AIR_SPEC,
    C_INT_SPEC, H_UFH_SPEC, H_UFH_SURF_SPEC, V_TS_SPEC, V_UFH_SPEC,
    R_SI, H_TR_INT, H_AIR2SURF,
)


@dataclass
class BuildingParams:
    raw: Dict[str, Any]
    derived: Dict[str, float]


def compute_derived_params(params: Dict[str, Any]) -> Dict[str, float]:
    """Compute derived thermal parameters from a building's raw parameter dict.

    Adds the following derived quantities (in-place on a copy):
      - volume_air, H_ve_tr                        (ventilation / total transmission)
      - H_con_floor, H_rad_con, H_tr_floor         (underfloor heating conductances)
      - C_water, C_air, C_int, C_zone              (thermal capacitances)
      - C_bldg, C_wall, C_surf, C_bldg_heavy       (building mass capacitances)
      - C_surf_floor, C_surf_wall                  (surface split by floor fraction)
      - A_surf, A_mass                             (effective surface areas)
      - H_int, H_air2surf, H_tr_int, H_tr_ext      (internal / external conductances)
    """
    out = dict(params)
    out['volume_air'] = out['area_floor'] * out['height_room']
    out['H_ve_tr'] = out['H_ve'] + out['H_tr']
    if out.get('H_tr_light'):
        out['H_tr_heavy'] = out['H_tr'] - out['H_tr_light']
    out['H_con_floor'] = H_UFH_SURF_SPEC * out['area_floor']
    out['H_rad_con'] = H_UFH_SPEC * out['area_floor']
    out['H_tr_floor'] = 1 / ((1 / out['H_rad_con']) - (1 / out['H_con_floor']))
    volume_water = (V_TS_SPEC + V_UFH_SPEC) * out['area_floor'] / 1000
    out['C_water'] = RHO_WATER * C_WATER_SPEC * volume_water
    out['C_air'] = RHO_AIR * C_AIR_SPEC * out['volume_air']
    out['C_int'] = C_INT_SPEC * out['area_floor']
    out['C_zone'] = out['C_air'] + out['C_int']
    out['C_bldg'] = out['c_bldg'] * out['area_floor'] * 3600
    out['C_wall'] = out['C_bldg'] - out['C_zone']
    out['C_surf'] = (out['C_bldg'] - out['C_air']) * 1 / 8
    out['C_bldg_heavy'] = out['C_bldg'] - out['C_air'] - out['C_surf']
    out['A_surf'] = out['area_floor'] * 4.5
    f_floor = out['area_floor'] / out['A_surf']
    out['C_surf_floor'] = f_floor * out['C_surf']
    out['C_surf_wall'] = (1 - f_floor) * out['C_surf']
    out['H_int'] = 1 / R_SI * out['A_surf']
    window_area = 0.0
    for orientation in out['windows']:
        window_area += orientation['area']
    out['A_mass'] = out['A_surf'] - window_area
    out['H_air2surf'] = H_AIR2SURF * out['A_surf']
    out['H_tr_int'] = H_TR_INT * out['A_mass']
    out['H_tr_ext'] = 1 / ((1 / out['H_tr_heavy']) - (1 / out['H_tr_int']))
    return out


def stack_params(param_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Stack a list of scalar param dicts into numpy arrays (per-env)."""
    if not param_list:
        return {}
    keys = param_list[0].keys()
    stacked = {}
    for k in keys:
        vals = [p[k] for p in param_list]
        if all(isinstance(v, (int, float, np.floating, np.integer)) for v in vals):
            stacked[k] = np.array(vals, dtype=np.float32)
    return stacked
