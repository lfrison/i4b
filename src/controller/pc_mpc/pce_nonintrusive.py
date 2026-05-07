"""
pce_uncertain_mpc
==================

This module provides a set of utilities for constructing non‐intrusive
polynomial chaos expansions (PCE) for uncertainty propagation in model
predictive control (MPC) problems.  The goal is to reuse the existing
deterministic MPC stack (``MPC_solver`` and friends) while augmenting it
with tools to approximate how the controlled system responds to random
parameter variations.

The functions here allow you to:

* Define a polynomial chaos basis (Hermite or Legendre) for a given
  number of random variables and expansion order.
* Sample collocation nodes in the random space and map them to
  concrete parameter values via a user supplied ``theta_from_xi`` mapping.
* Simulate the underlying building dynamics for each collocation node
  using the nominal plant model and control inputs from an existing
  ``MPC_solver`` (open‐loop prediction).
* Reconstruct the polynomial chaos coefficients at each prediction
  step by solving a least squares problem.  This yields an
  approximation of the random state vector ``x(t, ξ)`` as
  ``x(t, ξ) ≈ Σ_k β_k(t) φ_k(ξ)``, where ``β_k(t)`` is a vector of
  coefficients for each basis function and state component.
* Evaluate mean and variance of states from the PCE coefficients.

This approach follows the ``non‐intrusive polynomial chaos`` strategy
described in the paper by Frison et al.  Rather than deriving a
Galerkin‐projected set of differential equations for the PCE
coefficients, we rely on direct simulation of the system at a set of
collocation nodes and regression to determine the coefficients.  This
keeps the implementation simple and leverages the existing simulator.

Example usage
-------------

The typical workflow involves the following steps:

1. Define the uncertainty in your parameters via a function
   ``theta_from_xi(xi)`` and choose a PCE basis.
2. Solve a deterministic MPC problem using ``MPC_solver`` to obtain
   an open loop control sequence ``u_seq`` and disturbance sequence
   ``p_seq`` (the latter is passed from the disturbance DataFrame used
   in ``PcMpcRunner``).
3. Call :func:`compute_pce_coeffs` with the initial state, control
   sequence, disturbance sequence, basis and mapping ``theta_from_xi``.
   This returns a 3‑D array of PCE coefficients ``beta`` with shape
   (n_steps+1, n_basis, n_state).
4. Use :func:`pc_moments` to extract mean and variance of the state at
   any time step.

Limitations
-----------

* This implementation assumes that the plant dynamics are deterministic
  once the random parameters have been instantiated.  Stochastic
  disturbances (process noise) are not handled.
* The computational cost scales with the number of collocation nodes.
  The number of nodes must be at least equal to the number of basis
  functions to ensure an invertible regression.  For a basis of order
  ``p`` in ``d`` dimensions the number of basis functions is
  ``(d+p)!/(d!p!)``.  For example, with ``d=2`` and ``p=3`` one needs
  at least 10 collocation nodes.
* The control sequence used for propagation is assumed fixed.  In a
  receding horizon setting you would typically recompute controls at
  each step; this module approximates the behaviour of the MPC by
  freezing the control sequence obtained from the first solve.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import math
import sys

# Ensure legacy \"np\" module alias resolves to numpy if referenced elsewhere
sys.modules.setdefault("np", np)

from .core import PolynomialChaosBasis, pc_moments


def _eval_rhs_numeric(plant, x, u, p):
    """Evaluate the plant dynamics dx/dt using available numeric routines."""
    u_scalar = float(np.asarray(u).squeeze())
    p_vec = np.asarray(p, dtype=float).reshape(-1)
    x_vec = np.asarray(x, dtype=float)

    try:
        # Preferred: use the plain python dynamics used by the simulator
        dx = plant.calc(0.0, x_vec, [u_scalar, *p_vec.tolist()])
        return np.asarray(dx, dtype=float).reshape(-1)
    except Exception:
        pass

    rhs_func = getattr(plant, "calc_rhs", None)
    if rhs_func is not None:
        try:
            dx = rhs_func(x_vec, u_scalar, p_vec)
            return np.asarray(dx, dtype=float).reshape(-1)
        except Exception:
            pass

    rhs_casadi = getattr(plant, "calc_casadi", None)
    if rhs_casadi is not None:
        dx = rhs_casadi(x_vec, u_scalar, p_vec)
        return np.asarray(dx, dtype=float).reshape(-1)

    raise AttributeError("Plant model must provide calc, calc_rhs or calc_casadi.")


def build_collocation_nodes(
    basis: PolynomialChaosBasis,
    n_nodes: Optional[int] = None,
    oversampling: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a set of collocation nodes for non-intrusive PCE.

    We draw i.i.d. samples in the random space and fit the PCE via
    least squares. This is simple, non-intrusive and works well if we
    oversample relative to the basis size.

    Parameters
    ----------
    basis : PolynomialChaosBasis
        The polynomial chaos basis used for the expansion.
    n_nodes : int, optional
        Total number of collocation nodes. If ``None``, use
        ``n_nodes = max(basis.size, ceil(oversampling * basis.size))``.
    oversampling : float, optional
        Oversampling factor relative to ``basis.size`` when ``n_nodes``
        is not given. Default is 2.0.
    rng : numpy.random.Generator, optional
        Random number generator. If ``None``, a default_rng with a fixed
        seed is used for reproducibility.

    Returns
    -------
    nodes : ndarray
        Array of shape (n_nodes, dim) containing collocation nodes in
        the random space. For Hermite (Gaussian) chaos these are
        standard normal samples; for Legendre chaos they are drawn
        uniformly from ``[-1, 1]^dim``.
    """
    if n_nodes is None:
        n_nodes = max(basis.size, int(math.ceil(oversampling * basis.size)))

    dim = basis.dim
    if rng is None:
        rng = np.random.default_rng(12345)

    if basis.kind == "hermite":
        nodes = rng.standard_normal(size=(n_nodes, dim))
    else:
        # Legendre chaos: uniform in [-1, 1]^dim
        nodes = rng.uniform(low=-1.0, high=1.0, size=(n_nodes, dim))

    return nodes



@dataclass
class PcePropagationResult:
    """Container for the PCE propagation result."""

    beta: np.ndarray  # shape (n_steps+1, n_basis, n_state)
    basis: PolynomialChaosBasis
    states_nominal: np.ndarray  # shape (n_steps+1, n_state)
    collocation_nodes: np.ndarray  # shape (n_nodes, dim)
    u_seq: np.ndarray  # shape (n_steps, n_u)
    p_seq: np.ndarray  # shape (n_steps+1, n_p)

    def mean(self, step: int) -> np.ndarray:
        """Return the mean state at prediction step ``step``."""
        return pc_moments(self.beta[step])[0]

    def var(self, step: int) -> np.ndarray:
        """Return the variance of the state at prediction step ``step``."""
        return pc_moments(self.beta[step])[1]



def compute_pce_coeffs(
    x0: np.ndarray,
    u_seq: np.ndarray,
    p_seq: np.ndarray,
    basis: PolynomialChaosBasis,
    theta_from_xi: Callable[[np.ndarray], np.ndarray],
    building_model_nominal: any,
    plant_builder: Callable[[np.ndarray], any],
    h: float,
    n_substeps: int = 1,
    n_nodes: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> PcePropagationResult:
    """Compute PCE coefficients for the state trajectory under uncertain parameters.

    This helper propagates the building model from the given initial
    state ``x0`` under the provided control and disturbance sequences.
    For each collocation node ``xi_i`` drawn from the PCE basis a
    corresponding parameter vector ``theta_i = theta_from_xi(xi_i)``
    is constructed and a deterministic simulation of the plant
    is run using Euler integration.

    The resulting array of trajectories is then regressed onto the
    polynomial chaos basis to obtain the PCE coefficients ``beta``.

    Parameters
    ----------
    x0 : ndarray
        Initial state vector, shape (n_state,).
    u_seq : ndarray
        Control input sequence for the prediction horizon, shape
        (n_steps, n_u).
    p_seq : ndarray
        Disturbance sequence (parameters passed to
        ``optimization_problem``), shape (n_steps+1, n_p).
    basis : PolynomialChaosBasis
        Polynomial chaos basis describing the expansion.
    theta_from_xi : callable
        Maps a sample ``xi`` to the corresponding parameter vector
        ``theta``. This should be compatible with the plant builder.
    building_model_nominal : model_buildings.Building
        Nominal building model; used for its state dimension and to
        extract state keys.
    plant_builder : callable
        Function ``plant_builder(theta) -> Building`` that returns a
        plant model for the given parameters.
    h : float
        Sampling time [s] of the MPC.
    n_substeps : int, optional
        Number of Euler substeps per sampling interval. If > 1, each
        interval of length ``h`` is divided into ``n_substeps`` steps
        of size ``h / n_substeps`` with piecewise constant ``u`` and ``p``.
    n_nodes : int, optional
        If given, use exactly this many collocation nodes. If None,
        the number of nodes is chosen by :func:`build_collocation_nodes`
        (using an oversampling factor relative to ``basis.size``).
    rng : numpy.random.Generator, optional
        Random number generator used to draw collocation nodes.

    Returns
    -------
    PcePropagationResult
        Object containing PCE coefficients, nominal trajectory and
        metadata (collocation nodes, inputs, disturbances).
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    u_seq = np.asarray(u_seq, dtype=float)
    p_seq = np.asarray(p_seq, dtype=float)

    n_steps, n_u = u_seq.shape
    if p_seq.shape[0] != n_steps + 1:
        raise ValueError(f"p_seq must have {n_steps + 1} rows (got {p_seq.shape[0]}).")

    n_state = x0.shape[0]
    n_basis = basis.size

    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1.")
    dt = h / float(n_substeps)

    # Build collocation nodes and map to theta values
    xi_nodes = build_collocation_nodes(basis, n_nodes=n_nodes, rng=rng)
    n_nodes = xi_nodes.shape[0]
    theta_nodes = np.array([theta_from_xi(xi) for xi in xi_nodes])

    # Preallocate trajectory array: shape (n_nodes, n_steps+1, n_state)
    trajectories = np.zeros((n_nodes, n_steps + 1, n_state), dtype=float)
    trajectories[:, 0, :] = x0

    # Simulate each node deterministically
    for i_node, theta in enumerate(theta_nodes):
        plant = plant_builder(theta)
        x = x0.copy()
        trajectories[i_node, 0, :] = x
        for k in range(n_steps):
            u = u_seq[k]
            p = p_seq[k]
            for _ in range(n_substeps):
                dx = _eval_rhs_numeric(plant, x, u, p)
                x = x + dt * dx
            trajectories[i_node, k + 1, :] = x

    # Build Phi matrix (size n_nodes x n_basis) once
    Phi = basis.eval(xi_nodes)  # shape (n_nodes, n_basis)

    # Compute beta coefficients by least squares for each time and state
    beta = np.zeros((n_steps + 1, n_basis, n_state), dtype=float)
    # Precompute pseudo inverse in case Phi is not square
    pinv = np.linalg.pinv(Phi)
    for t in range(n_steps + 1):
        Xt = trajectories[:, t, :]  # shape (n_nodes, n_state)
        # Solve for beta at time t: beta = pinv @ Xt  (n_basis x n_state)
        beta[t] = pinv @ Xt

    # Also propagate the nominal (theta corresponding to xi=0) system for reference
    states_nominal = np.zeros((n_steps + 1, n_state), dtype=float)
    x_nom = x0.copy()
    states_nominal[0, :] = x_nom
    plant_nom = plant_builder(theta_from_xi(np.zeros(basis.dim)))
    for k in range(n_steps):
        u = u_seq[k]
        p = p_seq[k]
        for _ in range(n_substeps):
            dx_nom = _eval_rhs_numeric(plant_nom, x_nom, u, p)
            x_nom = x_nom + dt * dx_nom
        states_nominal[k + 1, :] = x_nom

    return PcePropagationResult(
        beta=beta,
        basis=basis,
        states_nominal=states_nominal,
        collocation_nodes=xi_nodes,
        u_seq=u_seq,
        p_seq=p_seq,
    )
__all__ = [
    "PolynomialChaosBasis",
    "PcePropagationResult",
    "compute_pce_coeffs",
    "build_collocation_nodes",
    "pc_moments",
]
