# -*- coding: utf-8 -*-
"""Intrusive PCE building models.

This module wraps the existing linear RC building models with an intrusive
polynomial-chaos expansion (PCE) in the uncertain parameters. The resulting
model evolves the coefficients of the state expansion directly, which makes it
usable inside direct chance-constrained optimal-control and MPC formulations.

Scope
-----
The implementation supports the linear models that already have CasADi support
in ``model_buildings.py``:

- ``2R2C``
- ``4R3C``
- ``5R4C``

The uncertainty is modeled as multiplicative Gaussian scaling on selected
building parameters. The default matches the current uncertain MPC notebook and
scales ``H_ve`` and ``H_tr`` around the nominal building.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import casadi as cas
import numpy as np
from statistics import NormalDist

from numpy.polynomial.hermite_e import hermegauss

from src.constants import C_WATER_SPEC
from src.controller.pc_mpc.core import PolynomialChaosBasis
from src.models.model_buildings import Building


def _expected_gaussian_weights(dim: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return tensor-product Gauss-Hermite-E nodes and standard-normal weights."""
    quad_order = max(4, 3 * order + 3)
    nodes_1d, weights_1d = hermegauss(quad_order)
    weight_scale = (2.0 * math.pi) ** (-0.5)

    nodes = []
    weights = []
    for multi_idx in itertools.product(range(quad_order), repeat=dim):
        node = [nodes_1d[i] for i in multi_idx]
        weight = weight_scale ** dim
        for i in multi_idx:
            weight *= weights_1d[i]
        nodes.append(node)
        weights.append(weight)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def _triple_products_hermite(basis: PolynomialChaosBasis) -> np.ndarray:
    """Compute E[phi_a phi_b phi_c] for the orthonormal Hermite basis."""
    nodes, weights = _expected_gaussian_weights(dim=basis.dim, order=basis.order)
    phi = basis.eval(nodes)
    n_basis = basis.size
    triple = np.zeros((n_basis, n_basis, n_basis), dtype=float)

    for a in range(n_basis):
        for b in range(n_basis):
            vals = weights * phi[:, a] * phi[:, b]
            for c in range(n_basis):
                triple[a, b, c] = np.sum(vals * phi[:, c])
    return triple


def _to_numpy_state(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


@dataclass
class _LinearTerm:
    """One affine term in the parameter-separated dynamics."""

    A: np.ndarray
    b_u: np.ndarray
    b_p: np.ndarray


class BuildingPCEIntrusive(Building):
    """Intrusive PCE wrapper around the existing linear building models.

    Parameters
    ----------
    pce_order : int
        Total PCE order.
    uncertain_param_keys : sequence of str
        Parameters modeled as multiplicative Gaussian uncertainty. The default
        ``("H_ve", "H_tr")`` matches the existing uncertain MPC workflow.
    theta_bar : ndarray, optional
        Mean of the multiplicative factors. Defaults to ones.
    L : ndarray, optional
        Linear map from standard-normal ``xi`` to parameter factors. Defaults
        to a diagonal matrix with 10 % std on ``H_ve`` and 5 % on ``H_tr``.
    """

    STD_SMOOTHING = 1e-4

    def __init__(
        self,
        params=None,
        mdot_hp: float = 0.25,
        T_room_set_lower: float = 20,
        T_room_set_upper: float = 26,
        method: str = "2R2C",
        usage: str = "ResidentialDetached",
        verbose: bool = False,
        pce_order: int = 2,
        uncertain_param_keys: Sequence[str] = ("H_ve", "H_tr"),
        theta_bar: Optional[np.ndarray] = None,
        L: Optional[np.ndarray] = None,
    ):
        super().__init__(
            params=params,
            mdot_hp=mdot_hp,
            T_room_set_lower=T_room_set_lower,
            T_room_set_upper=T_room_set_upper,
            method=method,
            usage=usage,
            verbose=verbose,
        )

        if method not in {"2R2C", "4R3C", "5R4C"}:
            raise NotImplementedError(
                "Intrusive PCE currently supports 2R2C, 4R3C and 5R4C."
            )

        self.base_state_keys = tuple(self.state_keys)
        self.base_disturbance_keys = tuple(key for key in self.input_keys if key != "T_hp_sup")
        self.base_nx = len(self.base_state_keys)

        self.uncertain_param_keys = tuple(uncertain_param_keys)
        self.uncertain_dim = len(self.uncertain_param_keys)
        self.theta_bar = (
            np.ones(self.uncertain_dim, dtype=float)
            if theta_bar is None
            else np.asarray(theta_bar, dtype=float).reshape(-1)
        )
        self.L = (
            np.diag([0.1, 0.05][: self.uncertain_dim]).astype(float)
            if L is None
            else np.asarray(L, dtype=float)
        )
        if self.theta_bar.shape[0] != self.uncertain_dim:
            raise ValueError("theta_bar length must match uncertain_param_keys.")
        if self.L.shape != (self.uncertain_dim, self.uncertain_dim):
            raise ValueError("L must be square with one row/column per uncertain parameter.")

        self.basis = PolynomialChaosBasis(
            dim=self.uncertain_dim,
            order=pce_order,
            kind="hermite",
        )
        self.n_basis = self.basis.size
        self.triple_products = _triple_products_hermite(self.basis)
        self.theta_coeffs = self._build_theta_coeffs()

        self.deterministic_term, self.param_terms = self._build_decomposition()

        # Coefficient-major state layout keeps the mean block first.
        self.state_keys = tuple(
            f"{base_key}__pc{basis_idx}"
            for basis_idx in range(self.n_basis)
            for base_key in self.base_state_keys
        )

    # ------------------------------------------------------------------ #
    #  Index helpers
    # ------------------------------------------------------------------ #

    def coeff_slice(self, basis_idx: int) -> slice:
        start = basis_idx * self.base_nx
        return slice(start, start + self.base_nx)

    def coeff_state_index(self, state_name: str, basis_idx: int = 0) -> int:
        state_offset = self.base_state_keys.index(state_name)
        return basis_idx * self.base_nx + state_offset

    def state_mean_expr(self, x, state_name: str):
        """Return the mean coefficient for a named physical state."""
        return x[self.coeff_state_index(state_name, basis_idx=0)]

    def state_variance_expr(self, x, state_name: str):
        """Return the symbolic variance from all non-constant coefficients."""
        idx0 = self.base_state_keys.index(state_name)
        coeffs = [x[self.coeff_state_index(self.base_state_keys[idx0], basis_idx=b)] for b in range(1, self.n_basis)]
        if not coeffs:
            return 0.0
        if isinstance(x, (cas.SX, cas.MX)):
            var = 0
            for coeff in coeffs:
                var += coeff * coeff
            return var
        coeff_arr = np.asarray(coeffs, dtype=float)
        return float(np.sum(coeff_arr ** 2))

    def state_std_expr(self, x, state_name: str):
        """Return the symbolic standard deviation for a named physical state."""
        var = self.state_variance_expr(x, state_name)
        eps = self.STD_SMOOTHING
        if isinstance(x, (cas.SX, cas.MX)):
            # Smooth sigma at zero variance so the chance-constraint Jacobian stays finite.
            return cas.sqrt(var + eps * eps) - eps
        return float(np.sqrt(max(var, 0.0) + eps * eps) - eps)

    def return_temperature_expr(self, x):
        """Return the mean return-temperature expression used by the controller."""
        return self.state_mean_expr(x, "T_hp_ret")

    def lower_comfort_constraint_expr(self, x, p, s, epsilon=None):
        """Return the lower comfort constraint for intrusive PCE states."""
        T_room_mean = self.state_mean_expr(x, "T_room")
        T_set_low = p[-2]
        if epsilon is None:
            return T_room_mean - T_set_low + s[0]

        z_value = NormalDist().inv_cdf(1.0 - epsilon)
        return T_room_mean - T_set_low - z_value * self.state_std_expr(x, "T_room") + s[0]

    def state_bounds(self):
        """Return bounds/initial guess for intrusive coefficient states."""
        x_min = []
        x_max = []
        x_init = []
        for basis_idx in range(self.n_basis):
            if basis_idx == 0:
                x_min.extend([0.0] * self.base_nx)
                x_max.extend([100.0] * self.base_nx)
                x_init.extend([35.0] * self.base_nx)
            else:
                x_min.extend([-100.0] * self.base_nx)
                x_max.extend([100.0] * self.base_nx)
                x_init.extend([0.0] * self.base_nx)
        return x_min, x_max, x_init

    def lift_deterministic_state(self, x0: Iterable[float] | Dict[str, float]) -> np.ndarray:
        """Embed a deterministic state in the coefficient state vector."""
        if isinstance(x0, dict):
            base = np.asarray([x0[key] for key in self.base_state_keys], dtype=float)
        else:
            base = _to_numpy_state(x0)
        if base.shape[0] == self.n_basis * self.base_nx:
            return base.copy()
        if base.shape[0] != self.base_nx:
            raise ValueError(f"x0 has length {base.shape[0]}, expected {self.base_nx}.")

        lifted = np.zeros(self.n_basis * self.base_nx, dtype=float)
        lifted[self.coeff_slice(0)] = base
        return lifted

    def mean_state(self, x_pc: Iterable[float]) -> np.ndarray:
        """Return the mean state block."""
        arr = _to_numpy_state(x_pc)
        return arr[self.coeff_slice(0)]

    def variance_state(self, x_pc: Iterable[float]) -> np.ndarray:
        """Return the per-state variance from higher-order coefficients."""
        arr = _to_numpy_state(x_pc).reshape(self.n_basis, self.base_nx)
        return np.sum(arr[1:] ** 2, axis=0)

    # ------------------------------------------------------------------ #
    #  Dynamics construction
    # ------------------------------------------------------------------ #

    def _build_theta_coeffs(self) -> np.ndarray:
        """Build basis coefficients for the uncertain parameter factors."""
        coeffs = np.zeros((self.uncertain_dim, self.n_basis), dtype=float)
        constant_idx = self.basis.multi_indices.index((0,) * self.uncertain_dim)
        coeffs[:, constant_idx] = self.theta_bar

        if self.basis.order >= 1:
            for dim_idx in range(self.uncertain_dim):
                alpha = [0] * self.uncertain_dim
                alpha[dim_idx] = 1
                basis_idx = self.basis.multi_indices.index(tuple(alpha))
                coeffs[:, basis_idx] = self.L[:, dim_idx]
        return coeffs

    def _empty_term(self) -> _LinearTerm:
        return _LinearTerm(
            A=np.zeros((self.base_nx, self.base_nx), dtype=float),
            b_u=np.zeros(self.base_nx, dtype=float),
            b_p=np.zeros((self.base_nx, len(self.base_disturbance_keys)), dtype=float),
        )

    def _build_decomposition(self) -> Tuple[_LinearTerm, List[_LinearTerm]]:
        """Split the dynamics into deterministic and parameter-dependent parts."""
        mdot_c = self.mdot_hp * C_WATER_SPEC

        if self.method == "2R2C":
            det = self._empty_term()
            det.A[0, 0] = -self.params["H_rad_con"] / self.params["C_bldg"]
            det.A[0, 1] = self.params["H_rad_con"] / self.params["C_bldg"]
            det.A[1, 0] = self.params["H_rad_con"] / self.params["C_water"]
            det.A[1, 1] = -(mdot_c + self.params["H_rad_con"]) / self.params["C_water"]
            det.b_u[1] = mdot_c / self.params["C_water"]
            det.b_p[0, 1] = 1.0 / self.params["C_bldg"]

            terms = []
            for key in self.uncertain_param_keys:
                coeff = self.params[key] / self.params["C_bldg"]
                term = self._empty_term()
                term.A[0, 0] = -coeff
                term.b_p[0, 0] = coeff
                terms.append(term)
            return det, terms

        if self.method == "4R3C":
            det = self._empty_term()
            det.A[0, 0] = -self.params["H_rad_con"] / self.params["C_zone"]
            det.A[0, 2] = self.params["H_rad_con"] / self.params["C_zone"]
            det.A[2, 0] = self.params["H_rad_con"] / self.params["C_water"]
            det.A[2, 2] = -(mdot_c + self.params["H_rad_con"]) / self.params["C_water"]
            det.b_u[2] = mdot_c / self.params["C_water"]
            det.b_p[0, 1] = 1.0 / self.params["C_zone"]

            terms = []
            for key in self.uncertain_param_keys:
                term = self._empty_term()
                if key == "H_ve":
                    coeff = self.params["H_ve"] / self.params["C_zone"]
                    term.A[0, 0] = -coeff
                    term.b_p[0, 0] = coeff
                elif key == "H_tr":
                    room_coeff = 2.0 * self.params["H_tr"] / self.params["C_zone"]
                    wall_coeff = 2.0 * self.params["H_tr"] / self.params["C_wall"]
                    term.A[0, 0] = -room_coeff
                    term.A[0, 1] = room_coeff
                    term.A[1, 0] = wall_coeff
                    term.A[1, 1] = -2.0 * wall_coeff
                    term.b_p[1, 0] = wall_coeff
                else:
                    raise KeyError(f"Unsupported uncertain parameter '{key}' for {self.method}.")
                terms.append(term)
            return det, terms

        if self.method == "5R4C":
            det = self._empty_term()
            det.A[0, 0] = -(self.params["H_rad_con"] + self.params["H_int"]) / self.params["C_air"]
            det.A[0, 1] = self.params["H_int"] / self.params["C_air"]
            det.A[0, 3] = self.params["H_rad_con"] / self.params["C_air"]
            det.A[1, 0] = self.params["H_int"] / self.params["C_int"]
            det.A[1, 1] = -self.params["H_int"] / self.params["C_int"]
            det.A[3, 0] = self.params["H_rad_con"] / self.params["C_water"]
            det.A[3, 3] = -(mdot_c + self.params["H_rad_con"]) / self.params["C_water"]
            det.b_u[3] = mdot_c / self.params["C_water"]
            det.b_p[0, 1] = 1.0 / self.params["C_air"]

            terms = []
            for key in self.uncertain_param_keys:
                term = self._empty_term()
                if key == "H_ve":
                    coeff = self.params["H_ve"] / self.params["C_air"]
                    term.A[0, 0] = -coeff
                    term.b_p[0, 0] = coeff
                elif key == "H_tr":
                    room_coeff = 2.0 * self.params["H_tr"] / self.params["C_air"]
                    wall_coeff = 2.0 * self.params["H_tr"] / self.params["C_wall"]
                    term.A[0, 0] = -room_coeff
                    term.A[0, 2] = room_coeff
                    term.A[2, 0] = wall_coeff
                    term.A[2, 2] = -2.0 * wall_coeff
                    term.b_p[2, 0] = wall_coeff
                else:
                    raise KeyError(f"Unsupported uncertain parameter '{key}' for {self.method}.")
                terms.append(term)
            return det, terms

        raise NotImplementedError(f"Unsupported method '{self.method}'.")

    # ------------------------------------------------------------------ #
    #  Intrusive PCE rhs
    # ------------------------------------------------------------------ #

    def _matvec(self, A: np.ndarray, x, backend):
        if backend is cas:
            return cas.mtimes(cas.DM(A), x)
        return A @ x

    def _forcing(self, term: _LinearTerm, u, p, backend):
        p_used = p[: len(self.base_disturbance_keys)]
        if backend is cas:
            return cas.DM(term.b_u) * u + cas.mtimes(cas.DM(term.b_p), p_used)
        return term.b_u * float(u) + term.b_p @ np.asarray(p_used, dtype=float)

    def _rhs_intrusive(self, x, u, p, backend):
        coeff_blocks = [x[self.coeff_slice(k)] for k in range(self.n_basis)]
        rhs_blocks = []

        for k in range(self.n_basis):
            rhs_k = self._matvec(self.deterministic_term.A, coeff_blocks[k], backend)
            if k == 0:
                rhs_k = rhs_k + self._forcing(self.deterministic_term, u, p, backend)

            for param_idx, term in enumerate(self.param_terms):
                for a in range(self.n_basis):
                    theta_a = self.theta_coeffs[param_idx, a]
                    if abs(theta_a) < 1e-14:
                        continue
                    for j in range(self.n_basis):
                        triple = self.triple_products[a, j, k]
                        if abs(triple) < 1e-14:
                            continue
                        rhs_k = rhs_k + (theta_a * triple) * self._matvec(term.A, coeff_blocks[j], backend)

                theta_k = self.theta_coeffs[param_idx, k]
                if abs(theta_k) > 1e-14:
                    rhs_k = rhs_k + theta_k * self._forcing(term, u, p, backend)

            rhs_blocks.append(rhs_k)

        if backend is cas:
            return cas.vertcat(*rhs_blocks)
        return np.concatenate(rhs_blocks)

    # ------------------------------------------------------------------ #
    #  Public model API
    # ------------------------------------------------------------------ #

    def calc(self, t, x, args):
        """Evaluate the intrusive PCE rhs numerically."""
        args_list = list(args)
        if not args_list:
            raise ValueError("Expected [u, *p] in args.")
        u = float(args_list[0])
        p = np.asarray(args_list[1:], dtype=float)
        return self._rhs_intrusive(_to_numpy_state(x), u, p, np)

    def calc_casadi(self, x, u, p):
        """Evaluate the intrusive PCE rhs in CasADi."""
        return self._rhs_intrusive(x, u, p, cas)
