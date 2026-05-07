from __future__ import annotations

import numpy as np
import casadi as cas

from src.constants import C_WATER_SPEC


class BESS:
    """Minimal one-state battery model for the electrical MPC extension."""

    def __init__(
        self,
        capacity_kwh,
        soc_init=0.5,
        p_ch_max_kw=0.0,
        p_dis_max_kw=0.0,
        eta_ch=1.0,
        eta_dis=1.0,
    ):
        self.capacity_kwh = max(0.0, float(capacity_kwh))
        self.soc_init = float(np.clip(soc_init, 0.0, 1.0)) if self.capacity_kwh > 0 else 0.0
        self.p_ch_max_kw = max(0.0, float(p_ch_max_kw)) if self.capacity_kwh > 0 else 0.0
        self.p_dis_max_kw = max(0.0, float(p_dis_max_kw)) if self.capacity_kwh > 0 else 0.0
        self.eta_ch = float(eta_ch)
        self.eta_dis = float(eta_dis)

        if self.eta_ch <= 0.0 or self.eta_dis <= 0.0:
            raise ValueError("BESS efficiencies must be positive.")

    @property
    def initial_energy_kwh(self):
        return self.capacity_kwh * self.soc_init

    @property
    def is_active(self):
        return self.capacity_kwh > 0.0


class BuildingBESS:
    """Wrap a thermal building model with a minimal electrical storage model."""

    control_keys = ("T_hp_sup", "P_pv_hp_kw", "P_pv_bess_kw", "P_bess_hp_kw")
    parameter_keys = ("T_amb", "Qdot_gains", "pv_kw", "T_room_set_lower", "grid")

    def __init__(self, base_building, bess: BESS):
        self.base_building = base_building
        self.bess = bess

        self.params = base_building.params
        self.method = base_building.method
        self.usage = getattr(base_building, "usage", None)
        self.mdot_hp = getattr(base_building, "mdot_hp", None)
        self.T_room_set_lower = getattr(base_building, "T_room_set_lower", None)
        self.T_room_set_upper = getattr(base_building, "T_room_set_upper", None)

        self.state_keys = tuple(base_building.state_keys) + ("E_bess_kWh",)
        self.input_keys = self.control_keys + self.parameter_keys

        self._base_nx = len(base_building.state_keys)
        self._u_idx = {name: idx for idx, name in enumerate(self.control_keys)}
        self._p_idx = {name: idx for idx, name in enumerate(self.parameter_keys)}

    def calc_casadi(self, x, u, p):
        """Return the augmented building and BESS dynamics."""
        rhs_building = self.base_building.calc_casadi(x[: self._base_nx], u[self._u_idx["T_hp_sup"]], p)
        p_pv_bess_kw = u[self._u_idx["P_pv_bess_kw"]]
        p_bess_hp_kw = u[self._u_idx["P_bess_hp_kw"]]
        dE_bess_dt = (self.bess.eta_ch * p_pv_bess_kw - p_bess_hp_kw / self.bess.eta_dis) / 3600.0
        return cas.vertcat(rhs_building, dE_bess_dt)

    def return_temperature_expr(self, x):
        return self.base_building.return_temperature_expr(x[: self._base_nx])

    def state_bounds(self):
        x_min, x_max, x_init = self.base_building.state_bounds()
        return (
            list(x_min) + [0.0],
            list(x_max) + [self.bess.capacity_kwh],
            list(x_init) + [self.bess.initial_energy_kwh],
        )

    def control_bounds(self):
        power_bound_kw = max(30.0, self.bess.p_ch_max_kw, self.bess.p_dis_max_kw)
        p_bess_ch_max = self.bess.p_ch_max_kw if self.bess.is_active else 0.0
        p_bess_dis_max = self.bess.p_dis_max_kw if self.bess.is_active else 0.0
        return (
            [0.0, 0.0, 0.0, 0.0],
            [65.0, power_bound_kw, p_bess_ch_max, p_bess_dis_max],
            [35.0, 0.0, 0.0, 0.0],
        )

    def stage_cost_expr(self, x, u, p, hp_model):
        grid_import_kw = self.hp_electric_power_expr(x, u, p, hp_model) - u[self._u_idx["P_pv_hp_kw"]] - u[self._u_idx["P_bess_hp_kw"]]
        return grid_import_kw / 100.0 * p[self._p_idx["grid"]]

    def constraint_exprs(self, x, u, p, s, hp_model, chance_epsilon=None, soft_lower_constraint=False):
        t_return = self.return_temperature_expr(x)
        thermal_power_kw = C_WATER_SPEC * hp_model.mdot_HP / 1000.0 * (u[self._u_idx["T_hp_sup"]] - t_return)
        hp_el_power_kw = self.hp_electric_power_expr(x, u, p, hp_model)
        comfort = self._lower_comfort_constraint(
            x=x,
            p=p,
            s=s,
            chance_epsilon=chance_epsilon,
            soft_lower_constraint=soft_lower_constraint,
        )
        pv_available = p[self._p_idx["pv_kw"]] - u[self._u_idx["P_pv_hp_kw"]] - u[self._u_idx["P_pv_bess_kw"]]
        hp_supply_balance = hp_el_power_kw - u[self._u_idx["P_pv_hp_kw"]] - u[self._u_idx["P_bess_hp_kw"]]
        return [
            comfort,
            thermal_power_kw,
            -(thermal_power_kw - 26.0),
            pv_available,
            hp_supply_balance,
        ]

    def hp_electric_power_expr(self, x, u, p, hp_model):
        t_return = self.return_temperature_expr(x)
        t_supply = u[self._u_idx["T_hp_sup"]]
        t_amb = p[self._p_idx["T_amb"]]
        cop = hp_model.COP(t_supply, t_amb)
        qth_kw = hp_model.mdot_HP * hp_model.c_water * (t_supply - t_return) / 1000.0
        return qth_kw / cop

    def _lower_comfort_constraint(self, x, p, s, chance_epsilon=None, soft_lower_constraint=False):
        slack_vec = s if soft_lower_constraint else 0 * s
        if hasattr(self.base_building, "lower_comfort_constraint_expr"):
            return self.base_building.lower_comfort_constraint_expr(
                x[: self._base_nx],
                p,
                slack_vec,
                epsilon=chance_epsilon,
            )
        if chance_epsilon is not None:
            raise AttributeError("Chance constraints require building_model.lower_comfort_constraint_expr(...).")
        t_room = x[0]
        t_set_low = p[self._p_idx["T_room_set_lower"]]
        return t_room - t_set_low + slack_vec[0]
