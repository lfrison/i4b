"""Right-hand-side (RHS) functions for RC thermal building models.

All RHS functions share the same calling convention::

    dx/dt = rhs(x, u, p, params)

where:
  x      – state vector (temperatures in °C)
  u      – scalar control input: heat-pump supply temperature T_hp_sup (°C)
  p      – disturbance vector: [T_amb, Qdot_gains, ...] (°C / W)
  params – dict of building parameters (thermal resistances/capacitances, etc.)

Model variants and their state layouts:
  2R2C : x = [T_room, T_hp_ret]
  4R3C : x = [T_room, T_wall, T_hp_ret]
  5R4C : x = [T_room, T_int, T_wall, T_hp_ret]
  6R4C : x = [T_room, T_surf, T_op*, T_mass, T_hp_ret]   (*placeholder, set post-step)
  7R5C : x = [T_room, T_surf_wall, T_op*, T_mass, T_surf_floor, T_hp_ret]

The linear models (all of the above) support exact zero-order-hold discretization
via ``src.core.integrators.linear_discretize_batch``.
"""
from __future__ import annotations
from typing import Dict
import jax.numpy as jnp
from src.constants import C_WATER_SPEC, H_AIR2SURF, H_TR_INT


def rhs_2r2c(x, u, p, params):
    """2R2C RHS. x=[T_room, T_hp_ret], u=T_hp_sup, p=[T_amb, Qdot_gains]."""
    T_room = x[..., 0]
    T_hp_ret = x[..., 1]
    T_hp_sup = u
    T_amb = p[..., 0]
    Qdot_gains = p[..., 1]
    rhs0 = 1 / params['C_bldg'] * (Qdot_gains + params['H_rad_con'] * (T_hp_ret - T_room)
                                   - params['H_ve_tr'] * (T_room - T_amb))
    rhs1 = 1 / params['C_water'] * (params['mdot_hp'] * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
                                   - params['H_rad_con'] * (T_hp_ret - T_room))
    return jnp.stack([rhs0, rhs1], axis=-1)


def rhs_4r3c(x, u, p, params):
    """4R3C RHS. x=[T_room, T_wall, T_hp_ret], u=T_hp_sup, p=[T_amb, Qdot_gains]."""
    T_room = x[..., 0]
    T_wall = x[..., 1]
    T_hp_ret = x[..., 2]
    T_hp_sup = u
    T_amb = p[..., 0]
    Qdot_gains = p[..., 1]
    rhs0 = 1 / params['C_zone'] * (Qdot_gains + params['H_rad_con'] * (T_hp_ret - T_room)
                                   - 2 * params['H_tr'] * (T_room - T_wall)
                                   - params['H_ve'] * (T_room - T_amb))
    rhs1 = 1 / params['C_wall'] * (2 * params['H_tr'] * (T_room - T_wall)
                                   - 2 * params['H_tr'] * (T_wall - T_amb))
    rhs2 = 1 / params['C_water'] * (params['mdot_hp'] * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
                                   - params['H_rad_con'] * (T_hp_ret - T_room))
    return jnp.stack([rhs0, rhs1, rhs2], axis=-1)


def rhs_5r4c(x, u, p, params):
    """5R4C RHS. x=[T_room, T_int, T_wall, T_hp_ret], u=T_hp_sup, p=[T_amb, Qdot_gains]."""
    T_room = x[..., 0]
    T_int = x[..., 1]
    T_wall = x[..., 2]
    T_hp_ret = x[..., 3]
    T_hp_sup = u
    T_amb = p[..., 0]
    Qdot_gains = p[..., 1]
    rhs0 = 1 / params['C_air'] * (Qdot_gains + params['H_rad_con'] * (T_hp_ret - T_room)
                                  - params['H_int'] * (T_room - T_int)
                                  - 2 * params['H_tr'] * (T_room - T_wall)
                                  - params['H_ve'] * (T_room - T_amb))
    rhs1 = 1 / params['C_int'] * params['H_int'] * (T_room - T_int)
    rhs2 = 1 / params['C_wall'] * (2 * params['H_tr'] * (T_room - T_wall)
                                   - 2 * params['H_tr'] * (T_wall - T_amb))
    rhs3 = 1 / params['C_water'] * (params['mdot_hp'] * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
                                   - params['H_rad_con'] * (T_hp_ret - T_room))
    return jnp.stack([rhs0, rhs1, rhs2, rhs3], axis=-1)


def rhs_6r4c(x, u, p, params):
    """6R4C RHS. x=[T_room, T_surf, T_op*, T_mass, T_hp_ret], p=[T_amb, Qdot_gains, Qdot_int, Qdot_sol].

    T_op (x[2]) is a placeholder computed post-step as 0.7*T_room + 0.3*T_surf.
    """
    T_room = x[..., 0]
    T_surf = x[..., 1]
    # x[..., 2] is T_op (placeholder)
    T_mass = x[..., 3]
    T_hp_ret = x[..., 4]
    T_hp_sup = u
    T_amb = p[..., 0]
    Qdot_int = p[..., 2]
    Qdot_sol = p[..., 3]

    Qdot_gain2air = 0.5 * Qdot_int
    Qdot_gain2mass = params['A_mass'] / params['A_surf'] * (0.5 * Qdot_int + Qdot_sol)
    Qdot_gain2surf = (1 - (params['A_mass'] / params['A_surf']) - (params['H_tr_light'] / (9.1 * params['A_surf']))) * (
        0.5 * Qdot_int + Qdot_sol
    )

    H_air2surf = H_AIR2SURF * params['A_surf']
    H_tr_int = H_TR_INT * params['A_mass']
    H_tr_ext = 1 / ((1 / params['H_tr_heavy']) - (1 / H_tr_int))

    Qdot_tr_int = H_tr_int * (T_surf - T_mass)
    Qdot_tr_ext = H_tr_ext * (T_mass - T_amb)

    Qdot_hvac = params['mdot_hp'] * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
    Qdot_hp2air = params['H_rad_con'] * (T_hp_ret - T_room)

    rhs0 = 1 / params['C_air'] * (Qdot_gain2air + params['H_rad_con'] * (T_hp_ret - T_room)
                                  - params['H_ve'] * (T_room - T_amb) - H_air2surf * (T_room - T_surf))
    rhs1 = 1 / params['C_surf'] * (Qdot_gain2surf + H_air2surf * (T_room - T_surf)
                                   - params['H_tr_light'] * (T_surf - T_amb) - H_tr_int * (T_surf - T_mass))
    rhs2 = jnp.zeros_like(rhs0)
    rhs3 = 1 / params['C_bldg_heavy'] * (Qdot_gain2mass + Qdot_tr_int - Qdot_tr_ext)
    rhs4 = 1 / params['C_water'] * (Qdot_hvac - Qdot_hp2air)
    return jnp.stack([rhs0, rhs1, rhs2, rhs3, rhs4], axis=-1)


def rhs_7r5c(x, u, p, params):
    """7R5C RHS. x=[T_room, T_surf_wall, T_op*, T_mass, T_surf_floor, T_hp_ret], p=[T_amb, Qdot_gains, Qdot_int, Qdot_sol].

    T_op (x[2]) is a placeholder computed post-step as a floor-area-weighted surface average.
    """
    T_room = x[..., 0]
    T_surf_wall = x[..., 1]
    # x[..., 2] is T_op (placeholder)
    T_mass = x[..., 3]
    T_surf_floor = x[..., 4]
    T_hp_ret = x[..., 5]
    T_hp_sup = u
    T_amb = p[..., 0]
    Qdot_int = p[..., 2]
    Qdot_sol = p[..., 3]

    Qdot_gain2air = 0.5 * Qdot_int
    Qdot_gain2mass = params['A_mass'] / params['A_surf'] * (0.5 * Qdot_int + Qdot_sol)
    Qdot_gain2surf = (1 - (params['A_mass'] / params['A_surf']) - (params['H_tr_light'] / (9.1 * params['A_surf']))) * (
        0.5 * Qdot_int + Qdot_sol
    )

    f_floor = params['area_floor'] / params['A_surf']
    Qdot_gain2surf_floor = f_floor * Qdot_gain2surf
    Qdot_gain2surf_wall = (1 - f_floor) * Qdot_gain2surf

    H_air2surf_wall = H_AIR2SURF * params['A_surf'] * (1 - f_floor)
    H_tr_int = H_TR_INT * params['A_mass']
    H_tr_ext = 1 / ((1 / params['H_tr_heavy']) - (1 / H_tr_int))

    Qdot_tr_int = H_tr_int * (T_surf_wall - T_mass)
    Qdot_tr_ext = H_tr_ext * (T_mass - T_amb)

    Qdot_hvac = params['mdot_hp'] * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
    Qdot_air2surf_wall = H_air2surf_wall * (T_room - T_surf_wall)
    Qdot_surf2air_floor = params['H_con_floor'] * (T_surf_floor - T_room)
    Qdot_tr_floor = params['H_tr_floor'] * (T_hp_ret - T_surf_floor)

    rhs0 = 1 / params['C_air'] * (Qdot_gain2air + Qdot_surf2air_floor
                                  - params['H_ve'] * (T_room - T_amb) - Qdot_air2surf_wall)
    rhs1 = 1 / params['C_surf_wall'] * (Qdot_gain2surf_wall + Qdot_air2surf_wall
                                        - params['H_tr_light'] * (T_surf_wall - T_amb)
                                        - H_tr_int * (T_surf_wall - T_mass))
    rhs2 = jnp.zeros_like(rhs0)
    rhs3 = 1 / params['C_bldg_heavy'] * (Qdot_gain2mass + Qdot_tr_int - Qdot_tr_ext)
    rhs4 = 1 / params['C_surf_floor'] * (Qdot_gain2surf_floor + Qdot_tr_floor - Qdot_surf2air_floor)
    rhs5 = 1 / params['C_water'] * (Qdot_hvac - Qdot_tr_floor)
    return jnp.stack([rhs0, rhs1, rhs2, rhs3, rhs4, rhs5], axis=-1)


def rhs_dispatch(method: str):
    """Return the RHS function for the given building model method string."""
    if method == '2R2C':
        return rhs_2r2c
    if method == '4R3C':
        return rhs_4r3c
    if method == '5R4C':
        return rhs_5r4c
    if method == '6R4C':
        return rhs_6r4c
    if method == '7R5C':
        return rhs_7r5c
    raise ValueError(f"Unknown method: {method}")
