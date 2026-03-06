from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from src.constants import C_WATER_SPEC


def t_ground_from_amb(t_ambient):
    return 6.645 * jnp.tanh(0.188 * (t_ambient - 9.177)) + 7.872


def cop_vitocal(T_sink, T_amb):
    T_source = t_ground_from_amb(T_amb)
    z0 = 10.893436
    a = -0.228602
    b = 0.266006
    c = 0.001461
    d = 0.000501
    f = -0.003546
    g = 0.0
    return z0 + (a * T_sink) + (b * T_source) + (c * T_sink**2) + (d * T_source**2) + (f * T_sink * T_source) + (g * T_sink**2 * T_source**2)


def cop_aw(T_sink, T_amb):
    a0, a1, a2, a3, a4, a5 = (8.2553, -0.17068, 0.16176, 0.00108, 0.00022, -0.00186)
    return a0 + a1 * T_sink + a2 * T_amb + a3 * T_sink**2 + a4 * T_amb**2 + a5 * T_sink * T_amb


def calc_hp(T_hp_ret, T_hp_sup, T_amb, timestep, params):
    cop = params['cop_fn'](T_hp_sup, T_amb) * params.get('cop_scale', 1.0)
    mdot = params['mdot_hp']
    if not hasattr(mdot, "shape"):
        mdot = jnp.array(mdot)
    Qdot_th = jnp.mean(mdot[..., None] * C_WATER_SPEC * (T_hp_sup[..., None] - T_hp_ret), axis=-1)
    Qdot_th = jnp.maximum(Qdot_th, 0.0)
    P_el = Qdot_th / cop
    E_el = P_el * timestep / 3600.0
    return {'P_el': P_el, 'E_el': E_el, 'COP': cop, 'Qdot_th': Qdot_th}


def check_hp(T_HP, T_RL, params):
    Q_HP = params['mdot_hp'] * C_WATER_SPEC * (T_HP - T_RL)
    Q_HP_min = params.get('Q_HP_min', 2000.0)
    Q_HP_max = params.get('Q_HP_max', 60000.0)
    Q_HP = jnp.where(Q_HP < Q_HP_min, 0.0, Q_HP)
    Q_HP = jnp.where(Q_HP > Q_HP_max, Q_HP_max, Q_HP)
    T_HP_new = Q_HP / (params['mdot_hp'] * C_WATER_SPEC) + T_RL
    return T_HP_new


def hp_params(hp_model_name: str, mdot_hp: float) -> Dict:
    if hp_model_name == 'Heatpump_AW':
        cop_fn = cop_aw
    elif hp_model_name == 'Heatpump_Vitocal':
        cop_fn = cop_vitocal
    else:
        raise ValueError(f"Unknown hp_model: {hp_model_name}")
    return {'mdot_hp': mdot_hp, 'cop_fn': cop_fn}
