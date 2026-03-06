from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np
from scipy.linalg import expm


def _tree_rms_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    total = jnp.array(0.0)
    count = 0
    for leaf in leaves:
        total = total + jnp.sum(jnp.square(leaf))
        count += leaf.size
    return jnp.sqrt(total / count)


def _tree_max_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    max_vals = [jnp.max(jnp.abs(leaf)) for leaf in leaves]
    return jnp.max(jnp.stack(max_vals))


def _select_norm(norm_mode: str):
    if norm_mode == "rms":
        return _tree_rms_norm
    if norm_mode == "max":
        return _tree_max_norm
    raise ValueError(f"Unknown norm_mode: {norm_mode}")


def _diffrax_integrate_single(
    rhs_fn: Callable,
    x0,
    u,
    p,
    params,
    dt,
    steps: int,
    saveat_mode: str = "ts",
    norm_mode: str = "rms",
    step_mode: str = "adaptive",
):
    """Integrate a single environment with Diffrax (used internally by vmap_integrate)."""
    term = dfx.ODETerm(lambda t, y, args: rhs_fn(y, u, p, params))
    solver = dfx.Tsit5()
    if saveat_mode == "t1":
        saveat = dfx.SaveAt(t1=True)
    elif saveat_mode == "ts":
        ts = jnp.linspace(0.0, dt, steps, dtype=x0.dtype)
        saveat = dfx.SaveAt(ts=ts)
    else:
        raise ValueError(f"Unknown saveat_mode: {saveat_mode}")
    if step_mode == "fixed":
        stepsize_controller = dfx.ConstantStepSize()
    elif step_mode == "adaptive":
        stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-7, norm=_select_norm(norm_mode))
    else:
        raise ValueError(f"Unknown step_mode: {step_mode}")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=float(dt),
        dt0=float(dt) / steps,
        y0=x0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
    )
    return sol.ys


def diffrax_integrate_batch(
    rhs_fn: Callable,
    x0,
    u,
    p,
    params,
    dt,
    steps: int,
    saveat_mode: str = "ts",
    norm_mode: str = "rms",
    step_mode: str = "adaptive",
):
    """Integrate a batch of envs as a single stacked ODE."""
    term = dfx.ODETerm(lambda t, y, args: rhs_fn(y, args[0], args[1], args[2]))
    solver = dfx.Tsit5()
    if saveat_mode == "t1":
        saveat = dfx.SaveAt(t1=True)
    elif saveat_mode == "ts":
        ts = jnp.linspace(0.0, dt, steps, dtype=x0.dtype)
        saveat = dfx.SaveAt(ts=ts)
    else:
        raise ValueError(f"Unknown saveat_mode: {saveat_mode}")
    if step_mode == "fixed":
        stepsize_controller = dfx.ConstantStepSize()
    elif step_mode == "adaptive":
        stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-7, norm=_select_norm(norm_mode))
    else:
        raise ValueError(f"Unknown step_mode: {step_mode}")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=float(dt),
        dt0=float(dt) / steps,
        y0=x0,
        args=(u, p, params),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
    )
    return sol.ys


def vmap_integrate(
    rhs_fn: Callable,
    x0,
    u,
    p,
    params,
    dt,
    steps: int,
    saveat_mode: str = "ts",
    norm_mode: str = "rms",
    step_mode: str = "adaptive",
):
    """Vectorized integrate for batch of envs."""
    def _one(x0_i, u_i, p_i, params_i):
        return _diffrax_integrate_single(
            rhs_fn,
            x0_i,
            u_i,
            p_i,
            params_i,
            dt,
            steps,
            saveat_mode=saveat_mode,
            norm_mode=norm_mode,
            step_mode=step_mode,
        )

    return jax.vmap(_one)(x0, u, p, params)


def jax_rk4_integrate(
    rhs_fn: Callable,
    x0,
    u,
    p,
    params,
    dt,
    steps: int,
    saveat_mode: str = "ts",
):
    """Fixed-step RK4 integrator implemented in JAX."""
    h = jnp.asarray(dt, dtype=x0.dtype) / steps

    if saveat_mode == "t1":
        def _body(_, x):
            k1 = rhs_fn(x, u, p, params)
            k2 = rhs_fn(x + 0.5 * h * k1, u, p, params)
            k3 = rhs_fn(x + 0.5 * h * k2, u, p, params)
            k4 = rhs_fn(x + h * k3, u, p, params)
            return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return jax.lax.fori_loop(0, steps, _body, x0)
    if saveat_mode == "ts":
        def _scan_fn(x, _):
            k1 = rhs_fn(x, u, p, params)
            k2 = rhs_fn(x + 0.5 * h * k1, u, p, params)
            k3 = rhs_fn(x + 0.5 * h * k2, u, p, params)
            k4 = rhs_fn(x + h * k3, u, p, params)
            x_next = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return x_next, x_next

        _, ys = jax.lax.scan(_scan_fn, x0, None, length=steps)
        return jnp.swapaxes(ys, 0, 1)
    raise ValueError(f"Unknown saveat_mode: {saveat_mode}")


def _linearize_rhs_batch(
    rhs_fn: Callable,
    params,
    state_dim: int,
    p_dim: int,
    dtype,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract continuous-time A, B, C matrices by probing the RHS with unit inputs.

    For each env, evaluates:
      A[i] = d(rhs)/dx  at x=0, u=0, p=0  (state Jacobian, shape [s, s])
      B[i] = rhs(0, 1, 0)                  (control sensitivity, shape [s])
      C[i] = d(rhs)/dp  at x=0, u=0, p=0  (disturbance Jacobian, shape [s, p])

    This is exact for linear RC models and approximate otherwise.
    Uses jax.vmap over the env batch dimension.
    """
    x_basis = jnp.eye(state_dim, dtype=dtype)
    zeros_x = jnp.zeros((state_dim,), dtype=dtype)
    zeros_p = jnp.zeros((p_dim,), dtype=dtype) if p_dim > 0 else jnp.zeros((0,), dtype=dtype)
    if p_dim > 0:
        p_basis = jnp.eye(p_dim, dtype=dtype)
    else:
        p_basis = None

    def _env_mats(params_i):
        def _rhs_x(xi):
            return rhs_fn(xi, jnp.asarray(0.0, dtype=dtype), zeros_p, params_i)

        A_cols = jax.vmap(_rhs_x)(x_basis)
        A = jnp.swapaxes(A_cols, 0, 1)

        B = rhs_fn(zeros_x, jnp.asarray(1.0, dtype=dtype), zeros_p, params_i)

        if p_dim > 0:
            def _rhs_p(pi):
                return rhs_fn(zeros_x, jnp.asarray(0.0, dtype=dtype), pi, params_i)

            C_cols = jax.vmap(_rhs_p)(p_basis)
            C = jnp.swapaxes(C_cols, 0, 1)
        else:
            C = jnp.zeros((state_dim, 0), dtype=dtype)

        return A, B, C

    A, B, C = jax.vmap(_env_mats)(params)
    return A, B, C


def linear_discretize_batch(
    rhs_fn: Callable,
    x0,
    p0,
    params,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact discrete-time matrices for batched linear systems.

    Returns (Ad, Bd, Cd) for x_next = Ad x + Bd u + Cd p.
    Uses jax.vmap(jax.scipy.linalg.expm) for GPU-parallel matrix exponential.
    Falls back to sequential scipy CPU loop if JAX expm is unavailable.
    """
    state_dim = int(x0.shape[1])
    p_dim = int(p0.shape[1])
    dtype = x0.dtype
    m_dim = 1 + p_dim

    cpu_devices = ()
    try:
        cpu_devices = jax.devices("cpu")
    except Exception:
        cpu_devices = ()

    # Prefer CPU for one-time linearization work when available, but do not
    # require it (e.g. when JAX_PLATFORMS=cuda restricts visible backends).
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            A, B, C = _linearize_rhs_batch(rhs_fn, params, state_dim, p_dim, dtype)
    else:
        A, B, C = _linearize_rhs_batch(rhs_fn, params, state_dim, p_dim, dtype)

    n_envs = int(A.shape[0])

    # GPU-parallel matrix exponential via jax.vmap.
    # Build augmented matrix M = [[A, G], [0, 0]] where G = [B | C],
    # then expm(M * dt) yields Ad in the top-left block and F = [Bd | Cd] top-right.
    try:
        G = jnp.concatenate([B[:, :, None], C], axis=-1)           # (n, s, m)
        M_top = jnp.concatenate([A, G], axis=-1)                    # (n, s, s+m)
        M_bot = jnp.zeros((n_envs, m_dim, state_dim + m_dim), dtype=dtype)
        M = jnp.concatenate([M_top, M_bot], axis=1) * float(dt)    # (n, s+m, s+m)
        expM = jax.vmap(jax.scipy.linalg.expm)(M)                  # (n, s+m, s+m)
        Ad = np.array(expM[:, :state_dim, :state_dim].astype(jnp.float32))
        F = np.array(expM[:, :state_dim, state_dim:].astype(jnp.float32))
    except Exception:
        # Fallback: sequential scipy CPU loop (original path).
        A_np = np.array(A)
        B_np = np.array(B)
        C_np = np.array(C)
        Ad = np.zeros((n_envs, state_dim, state_dim), dtype=np.float32)
        F = np.zeros((n_envs, state_dim, m_dim), dtype=np.float32)
        for i in range(n_envs):
            G_i = np.concatenate([B_np[i][:, None], C_np[i]], axis=1)
            M_i = np.zeros((state_dim + m_dim, state_dim + m_dim), dtype=np.float64)
            M_i[:state_dim, :state_dim] = A_np[i]
            M_i[:state_dim, state_dim:] = G_i
            expM_i = expm(M_i * float(dt))
            Ad[i] = expM_i[:state_dim, :state_dim].astype(np.float32)
            F[i] = expM_i[:state_dim, state_dim:].astype(np.float32)

    Bd = F[:, :, 0]
    Cd = F[:, :, 1:]
    return Ad, Bd, Cd
