"""Polynomial chaos MPC scaffold.

This module provides the core pieces needed to run nominal MPC against an
uncertain plant model:

- parametric uncertainty handling and sampling,
- polynomial chaos basis evaluation,
- closed-loop MPC simulation under random parameters,
- small helpers for disturbance ordering and PCE moments.

The functions are intentionally lightweight: they reuse the existing
``MPC_solver`` and ``Model_simulator`` classes so experiments can be wired
up without rewriting the deterministic MPC stack.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

from src.controller.mpc.casadi_framework import MPC_solver
from src.simulator import Model_simulator
import src.models.model_buildings as model_buildings


# --------------------------------------------------------------------------- #
#  Polynomial chaos basis
# --------------------------------------------------------------------------- #

def _hermite_eval(degree: int, x: np.ndarray) -> np.ndarray:
    """Evaluate orthonormal probabilists' Hermite polynomial.

    For ξ ~ N(0, 1), the probabilists' Hermite polynomials He_n satisfy
    E[He_n(ξ) He_m(ξ)] = n! δ_{nm}.

    We normalise for n > 0 by √(n!) so that the 1D basis functions
    are orthonormal w.r.t. the Gaussian measure, and keep H_0 ≡ 1.
    """
    x = np.asarray(x, dtype=float)
    if degree == 0:
        # Constant basis function, already normalised for N(0,1)
        return np.ones_like(x)

    from numpy.polynomial.hermite_e import hermeval

    coeffs = [0.0] * degree + [1.0]
    he_n = hermeval(x, coeffs)
    return he_n / math.sqrt(math.factorial(degree))


def _legendre_eval(degree: int, x: np.ndarray) -> np.ndarray:
    """Evaluate orthonormal Legendre polynomial on [-1, 1].

    For X ~ Uniform(-1, 1), the standard Legendre polynomials P_n satisfy
    E[P_n(X) P_m(X)] = δ_{nm} / (2n + 1).

    We scale for n > 0 by √(2n+1) so that the 1D basis functions are
    orthonormal w.r.t. the uniform probability measure, and keep P_0 ≡ 1.
    """
    x = np.asarray(x, dtype=float)
    if degree == 0:
        # Constant basis function, already normalised for U(-1,1)
        return np.ones_like(x)

    from numpy.polynomial.legendre import legval

    coeffs = [0.0] * degree + [1.0]
    p_n = legval(x, coeffs)
    return p_n * math.sqrt(2 * degree + 1.0)



@dataclass
class PolynomialChaosBasis:
    """Generate and evaluate multivariate PCE bases up to total order ``p``.

    Parameters
    ----------
    dim : int
        Number of random variables.
    order : int
        Maximum total polynomial degree.
    kind : str
        Either ``"hermite"`` (Gaussian variables) or ``"legendre"`` (uniform).
    """

    dim: int
    order: int
    kind: str = "hermite"

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in {"hermite", "legendre"}:
            raise ValueError(f"Unsupported basis kind '{self.kind}'.")
        self.multi_indices = self._generate_multi_indices()

    def _generate_multi_indices(self) -> List[Tuple[int, ...]]:
        """Return all multi-indices with |alpha| <= order."""
        indices: List[Tuple[int, ...]] = []
        for total_deg in range(self.order + 1):
            for alpha in itertools.product(range(self.order + 1), repeat=self.dim):
                if sum(alpha) == total_deg:
                    indices.append(alpha)
        return indices

    @property
    def size(self) -> int:
        """Number of basis elements."""
        return len(self.multi_indices)

    def eval(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate all basis polynomials at ``xi``.

        Parameters
        ----------
        xi : ndarray
            Shape (dim,) or (n_samples, dim).

        Returns
        -------
        ndarray
            Shape (size,) for a single sample or (n_samples, size) otherwise.
        """
        xi_arr = np.atleast_2d(np.asarray(xi, dtype=float))
        if xi_arr.shape[1] != self.dim:
            raise ValueError(f"xi has dim {xi_arr.shape[1]}, expected {self.dim}.")

        if self.kind == "hermite":
            univar_eval: Callable[[int, np.ndarray], np.ndarray] = _hermite_eval
        else:
            univar_eval = _legendre_eval

        basis_vals = np.zeros((xi_arr.shape[0], self.size))
        for k, alpha in enumerate(self.multi_indices):
            val = np.ones(xi_arr.shape[0])
            for d, degree in enumerate(alpha):
                if degree == 0:
                    continue
                val *= univar_eval(degree, xi_arr[:, d])
            basis_vals[:, k] = val
        return basis_vals if xi_arr.ndim > 1 or xi_arr.shape[0] > 1 else basis_vals[0]


# --------------------------------------------------------------------------- #
#  Uncertainty handling
# --------------------------------------------------------------------------- #


@dataclass
class UncertaintyModel:
    """Linear Gaussian uncertainty model ``theta = theta_bar + L @ xi``."""

    theta_bar: np.ndarray
    L: np.ndarray
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.theta_bar = np.asarray(self.theta_bar, dtype=float).reshape(-1)
        self.L = np.asarray(self.L, dtype=float)
        if self.L.shape[0] != self.theta_bar.shape[0]:
            raise ValueError("L rows must match theta_bar dimension.")
        self.rng = self.rng or np.random.default_rng()

    @property
    def dim(self) -> int:
        return self.L.shape[1]

    def theta_from_xi(self, xi: np.ndarray) -> np.ndarray:
        xi_vec = np.asarray(xi, dtype=float).reshape(self.dim)
        return self.theta_bar + self.L @ xi_vec

    def sample_xi(self) -> np.ndarray:
        return self.rng.standard_normal(self.dim)


@dataclass
class MultiplicativePlantBuilder:
    """Simple plant builder that scales selected building parameters.

    Each entry in ``param_keys`` is multiplied by the corresponding value in
    ``theta``. All other settings (method, mdot, comfort bounds, usage) are
    copied from the base building instance.
    """

    base_building: Any
    param_keys: Sequence[str]
    extra_kwargs: Optional[Dict[str, Any]] = None

    def __call__(self, theta: np.ndarray) -> Any:
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if theta_vec.shape[0] != len(self.param_keys):
            raise ValueError(f"theta has length {theta_vec.shape[0]}, expected {len(self.param_keys)}.")
        params = copy.deepcopy(self.base_building.params)
        for key, scale in zip(self.param_keys, theta_vec):
            if key not in params:
                raise KeyError(f"Parameter '{key}' not in building params.")
            params[key] = params[key] * float(scale)
        kwargs = dict(
            mdot_hp=self.base_building.mdot_hp,
            T_room_set_lower=self.base_building.T_room_set_lower,
            T_room_set_upper=self.base_building.T_room_set_upper,
            method=self.base_building.method,
            usage=getattr(self.base_building, "usage", "ResidentialDetached"),
        )
        if self.extra_kwargs:
            kwargs.update(self.extra_kwargs)
        return model_buildings.Building(params=params, **kwargs)


def ensure_disturbance_order(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return disturbances DataFrame with required column order."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing disturbance columns: {missing}")
    return df.loc[:, list(columns)]


def pc_moments(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance from PCE coefficients (orthonormal basis).

    Parameters
    ----------
    beta : ndarray
        Shape (n_basis, n_state).

    Returns
    -------
    mean : ndarray
    var : ndarray
    """
    beta = np.asarray(beta, dtype=float)
    mean = beta[0]
    # Under orthonormal basis the variance is the sum of squared higher coeffs
    var = np.sum(beta[1:] ** 2, axis=0)
    return mean, var


# --------------------------------------------------------------------------- #
#  Closed-loop MPC simulation
# --------------------------------------------------------------------------- #


def _to_state_dict(x: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    return {key: float(val) for key, val in zip(keys, x)}


def _to_state_vec(x: Dict[str, float], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([x[key] for key in keys], dtype=float)


@dataclass
class PcMpcRunner:
    """Closed-loop MPC runner with uncertainty-aware plant.

    Parameters
    ----------
    hp_model : model_hvac.Heatpump_xx
        Heat pump model used in both controller and plant.
    building_model_nominal : model_buildings.Building
        Nominal model used to build the MPC (controller).
    disturbances : pandas.DataFrame
        Disturbance trajectories; first ``npar`` columns are passed to MPC.
    theta_from_xi : callable
        Maps ``xi`` samples to concrete parameter vector ``theta``.
    plant_builder : callable
        ``plant_builder(theta) -> Building`` returning a plant model with those parameters.
    mpc_factory : callable, optional
        If provided, used to build a fresh ``MPC_solver``; otherwise a default
        ``MPC_solver`` is created with the supplied dimensions.
    mpc_solver : MPC_solver, optional
        Pre-built controller to reuse (e.g. nominal deterministic MPC).
    xi_sampler : callable, optional
        Custom sampler for xi; if omitted the runner tries to use
        ``theta_from_xi.__self__.sample_xi`` (e.g. an ``UncertaintyModel``),
        otherwise ``xi_dim`` must be provided for standard-normal sampling.
    """

    hp_model: Any  # heat pump model reused by controller and plant
    building_model_nominal: Any  # nominal building used to build the MPC
    disturbances: pd.DataFrame  # disturbance trajectories (first npar columns go to MPC)
    theta_from_xi: Callable[[np.ndarray], np.ndarray]  # maps xi -> theta
    plant_builder: Callable[[np.ndarray], Any]  # builds plant model from theta
    mpc_factory: Optional[Callable[[Any], MPC_solver]] = None  # optional factory for MPC_solver
    mpc_solver: Optional[MPC_solver] = None  # reuse an existing MPC_solver if provided
    xi_sampler: Optional[Callable[[], np.ndarray]] = None  # custom xi sampler
    xi_dim: Optional[int] = None  # dimension of xi (used if no sampler is given)
    nk: int = 96  # prediction horizon in steps
    h: int = 900  # sampling time [s] (15 min default)
    step_length: int = 1  # MPC receding-horizon shift (steps advanced per solve)
    ws: float = 0.1  # weight for temperature slack in objective
    npar: Optional[int] = None  # number of disturbance parameters passed to MPC
    nc: int = 3  # number of constraints (consistent with optimization_problem)
    ns: int = 1  # number of slack variables
    nu: int = 1  # number of inputs
    resultdir: str = "results_pc_mpc"  # output directory for MPC logs
    resultfile_prefix: str = "pc_mpc"  # result filename prefix
    grid_column: str = "grid"  # column name for grid price/signal in disturbances
    lower_set_column: str = "T_room_set_lower"  # column name for comfort lower bound
    comfort_margin: float | Callable[[pd.DataFrame], float] = 0.0  # optional tightening of comfort bound to hedge uncertainty

    def __post_init__(self) -> None:
        self.state_keys: Sequence[str] = tuple(self.building_model_nominal.state_keys)
        self.nx = len(self.state_keys)
        self.npar = self.npar or self.disturbances.shape[1]
        if self.mpc_solver is None and self.mpc_factory is None:
            self.mpc_factory = self._default_mpc_factory
        bound_uncertainty = getattr(self.theta_from_xi, "__self__", None)
        if self.xi_dim is None and bound_uncertainty is not None and hasattr(bound_uncertainty, "dim"):
            self.xi_dim = getattr(bound_uncertainty, "dim")

    # Controller and simulator construction helpers --------------------- #
    def _default_mpc_factory(self, building_model: Any) -> MPC_solver:
        resultfile = f"{self.resultfile_prefix}_N{self.nk}_h{self.h}"
        return MPC_solver(
            self.resultdir,
            resultfile,
            self.hp_model,
            building_model,
            nx=self.nx,
            nu=self.nu,
            npar=self.npar,
            ns=self.ns,
            nc=self.nc,
            h=self.h,
            nk=self.nk,
            ws=self.ws,
        )

    def _get_mpc(self) -> MPC_solver:
        return self.mpc_solver or self.mpc_factory(self.building_model_nominal)

    def _get_mpc_steps(self, mpc_steps: Optional[int]) -> int:
        if mpc_steps is not None:
            return mpc_steps
        # maximum number of steps such that a full horizon is available
        available = len(self.disturbances) - (self.nk + 1)
        return max(available // self.step_length + 1, 0)

    def _sample_xi(self) -> np.ndarray:
        if self.xi_sampler is not None:
            return np.asarray(self.xi_sampler(), dtype=float).reshape(-1)
        bound_uncertainty = getattr(self.theta_from_xi, "__self__", None)
        if bound_uncertainty is not None and hasattr(bound_uncertainty, "sample_xi"):
            return np.asarray(bound_uncertainty.sample_xi(), dtype=float).reshape(-1)
        if self.xi_dim is None:
            raise ValueError("xi_dim is not set; provide xi_dim or xi_sampler for Monte Carlo sampling.")
        return np.random.standard_normal(self.xi_dim)

    # Core simulation --------------------------------------------------- #
    def run_sample(self, x0: Iterable[float] | Dict[str, float], xi_sample: np.ndarray, mpc_steps: Optional[int] = None, capture_plan: bool = False) -> Dict[str, Any]:
        """Simulate closed-loop MPC for one xi sample.

        If ``capture_plan`` is True, the open-loop control plan from the first
        MPC solve is returned under the ``plan`` key (useful for plotting).
        """
        x_vec = _to_state_vec(x0, self.state_keys) if isinstance(x0, dict) else np.asarray(x0, dtype=float).reshape(self.nx)
        xi_vec = np.asarray(xi_sample, dtype=float).reshape(-1)
        if self.xi_dim is not None and xi_vec.shape[0] != self.xi_dim:
            raise ValueError(f"xi_sample has dim {xi_vec.shape[0]}, expected {self.xi_dim}.")
        theta = self.theta_from_xi(xi_vec)
        plant_model = self.plant_builder(theta)
        simulator = Model_simulator(self.hp_model, plant_model, self.h)
        mpc = self._get_mpc()

        rows: List[Dict[str, Any]] = []
        total_steps = self._get_mpc_steps(mpc_steps)
        open_loop_plan: Optional[np.ndarray] = None
        for step_idx in range(total_steps):
            start = step_idx * self.step_length
            window = self.disturbances.iloc[start : start + self.nk + 1]
            # Apply optional comfort tightening (e.g., to account for uncertainty)
            if self.comfort_margin:
                margin = self.comfort_margin(window) if callable(self.comfort_margin) else self.comfort_margin
                window = window.copy()
                if self.lower_set_column in window.columns:
                    window[self.lower_set_column] = window[self.lower_set_column] - float(margin)
            if len(window) < self.nk + 1:
                break

            P = window.iloc[:, : self.npar].values
            mpc.update_NLP(x_vec)
            if capture_plan and step_idx == 0:
                try:
                    uk_opt, xk_next_guess, res = mpc.solve_NLP(P, return_res=True)
                    open_loop_plan = self._extract_control_sequence(res["x"])
                except TypeError:
                    # Older MPC_solver without return_res support; fall back gracefully
                    uk_opt, xk_next_guess = mpc.solve_NLP(P)
                    open_loop_plan = None
            else:
                uk_opt, xk_next_guess = mpc.solve_NLP(P)
            uk_scalar = float(np.asarray(uk_opt).squeeze())

            pk = window.iloc[0].to_dict()
            state_dict = _to_state_dict(x_vec, self.state_keys)
            sim_out = simulator.get_next_state(x_init=state_dict, uk=uk_scalar, pk=pk)
            next_state = sim_out["state"]

            rows.append(
                {
                    "time": step_idx * self.step_length * self.h,
                    "theta": theta,
                    **next_state,
                    "T_hp_sup": uk_scalar,
                    **pk,
                }
            )
            x_vec = _to_state_vec(next_state, self.state_keys)

        traj = pd.DataFrame(rows)
        stats = self._evaluate_stats(traj)
        result = {"trajectory": traj, "theta": theta, "stats": stats}
        if open_loop_plan is not None:
            result["plan"] = {"u_sequence": open_loop_plan}
        return result

    def _extract_control_sequence(self, res_x: np.ndarray) -> np.ndarray:
        """Slice the optimal control sequence from solver decision vector."""
        dim = getattr(self.mpc_solver, "dim", None) or getattr(self.mpc_factory(self.building_model_nominal), "dim")
        nx, ns, nu, d = dim["nx"], dim["ns"], dim["nu"], dim["d"]
        res_arr = np.asarray(res_x).reshape(-1)
        controls = []
        offset = 0
        for _ in range(self.nk):
            offset += (d + 1) * nx
            offset += (d + 1) * ns
            controls.append(res_arr[offset : offset + nu])
            offset += nu
        return np.vstack(controls)

    def _evaluate_stats(self, traj: pd.DataFrame) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if traj.empty:
            return stats

        # Comfort violation probability and depth
        if "T_room" in traj.columns and self.lower_set_column in traj.columns:
            deviation = np.maximum(traj[self.lower_set_column] - traj["T_room"], 0.0)
            stats["violation_probability"] = float(np.mean(deviation > 0))
            stats["max_violation"] = float(np.max(deviation)) if len(deviation) else 0.0
            stats["avg_violation"] = float(np.mean(deviation))

        # Grid cost / electric energy
        ret_key = "T_return" if "T_return" in traj.columns else "T_hp_ret" if "T_hp_ret" in traj.columns else None
        if ret_key and "T_hp_sup" in traj.columns and "T_amb" in traj.columns:
            cop = np.maximum(self.hp_model.COP(traj["T_hp_sup"], traj["T_amb"]), 1e-6)
            pth = self.hp_model.mdot_HP * self.hp_model.c_water * (traj["T_hp_sup"] - traj[ret_key]) / 1000.0
            pel = pth / cop
            stats["avg_pel_kw"] = float(np.mean(pel))
            stats["total_energy_kwh"] = float(np.sum(pel) * self.h / 3600.0)
            if self.grid_column in traj.columns:
                stats["grid_cost"] = float(np.sum(pel * traj[self.grid_column]) * self.h / 3600.0)

        return stats


def simulate_closed_loop_mpc(
    x0: Iterable[float] | Dict[str, float],
    xi_sample: np.ndarray,
    controller_config: PcMpcRunner,
    mpc_steps: Optional[int] = None,
    capture_plan: bool = False,
) -> Dict[str, Any]:
    """Functional wrapper around ``PcMpcRunner.run_sample``."""
    return controller_config.run_sample(
        x0=x0, xi_sample=xi_sample, mpc_steps=mpc_steps, capture_plan=capture_plan
    )
