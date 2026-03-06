from __future__ import annotations

from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

from src.core.dynamics import rhs_dispatch
from src.core.integrators import vmap_integrate, diffrax_integrate_batch, jax_rk4_integrate, linear_discretize_batch
from src.core.hvac import calc_hp, check_hp


LINEAR_METHODS = {"2R2C", "4R3C", "5R4C", "6R4C", "7R5C"}


def _compress_param_batches(params):
    """Reduce duplicate per-env parameter rows to unique batches.

    When many envs share identical building parameters (common in homogeneous
    training setups), the matrix exponential in ``prepare_linear`` only needs
    to be computed once per unique configuration instead of once per env.
    Returns ``(unique_params, inverse_index)`` where ``inverse_index`` maps
    each original env index back to the corresponding unique row.
    """
    if not params:
        return params, None
    keys = sorted(params.keys())
    values = [np.asarray(params[k]) for k in keys]
    n_envs = int(values[0].shape[0])
    if n_envs <= 1:
        return params, None
    stacked = np.stack(values, axis=1)
    unique_rows, inverse = np.unique(stacked, axis=0, return_inverse=True)
    if unique_rows.shape[0] == stacked.shape[0]:
        return params, None
    unique_params = {
        k: jnp.asarray(unique_rows[:, idx], dtype=params[k].dtype)
        for idx, k in enumerate(keys)
    }
    return unique_params, inverse


class JaxSimulator:
    def __init__(
        self,
        method: str,
        timestep: int,
        integration_steps: int = 10,
        batch_mode: str = "vmap",
        saveat_mode: str = "ts",
        norm_mode: str = "rms",
        step_mode: str = "adaptive",
        integrator: str = "diffrax",
    ):
        self.method = method
        self.timestep = timestep
        self.integration_steps = integration_steps
        self.batch_mode = batch_mode
        self.saveat_mode = saveat_mode
        self.norm_mode = norm_mode
        self.step_mode = step_mode
        self.integrator = integrator
        self.rhs = rhs_dispatch(method)
        self._compiled_step = None
        self._linear_mats = None
        self._linear_shapes = None

    def reset_cache(self):
        """Clear compiled step function and linear matrices.

        Forces recompilation on the next ``step()`` or ``prepare_linear()`` call.
        Required when the model structure changes (e.g. different number of envs
        or state dimensions). Not needed for value-only updates (params, matrices).
        """
        self._compiled_step = None
        self._linear_mats = None
        self._linear_shapes = None

    def prepare_linear(self, params, state_dim: int, p_dim: int, dtype=jnp.float32):
        """Precompute exact discrete-time matrices (Ad, Bd, Cd) for the linear integrator.

        Linearizes the RC dynamics symbolically, then computes the zero-order-hold
        discretization via GPU-parallel matrix exponential (``jax.vmap`` over
        ``jax.scipy.linalg.expm``). Deduplicates identical parameter rows before
        the expm to avoid redundant computation.

        Clears the compiled step kernel only when array shapes change; value-only
        updates (e.g. after domain randomization) reuse the existing XLA kernel.

        Args:
            params:    dict of (n_envs,) JAX arrays with building parameters.
            state_dim: number of thermal state variables.
            p_dim:     number of disturbance inputs.
            dtype:     floating-point dtype for the matrices (default float32).
        """
        if self.integrator != "linear":
            return
        if self.method not in LINEAR_METHODS:
            raise ValueError(f"Linear integrator not supported for method {self.method}.")
        if not params:
            raise ValueError("Linear integrator requires non-empty params.")
        n_envs = int(next(iter(params.values())).shape[0])
        params_compact, inverse = _compress_param_batches(params)
        n_unique = int(next(iter(params_compact.values())).shape[0])
        x0 = jnp.zeros((n_unique, state_dim), dtype=dtype)
        p0 = jnp.zeros((n_unique, p_dim), dtype=dtype)
        Ad, Bd, Cd = linear_discretize_batch(self.rhs, x0, p0, params_compact, self.timestep)
        if inverse is not None:
            Ad = Ad[inverse]
            Bd = Bd[inverse]
            Cd = Cd[inverse]
        new_shapes = ((n_envs, state_dim), (n_envs, p_dim))
        # Only clear the compiled step when shapes change (different JIT trace needed).
        # When only matrix VALUES change (same shapes), the compiled kernel is reused.
        if new_shapes != self._linear_shapes:
            self._compiled_step = None
        self._linear_mats = (
            jnp.array(Ad, dtype=dtype),
            jnp.array(Bd, dtype=dtype),
            jnp.array(Cd, dtype=dtype),
        )
        self._linear_shapes = new_shapes

    def step(self, x, u, p, params, linear_mats=None):
        """One-step integration for batch of envs.

        For the linear integrator, matrices are passed as explicit JIT args via
        ``linear_mats=(Ad, Bd, Cd)`` so that updating them does not require
        recompiling the kernel (same shapes → same XLA computation).
        """
        if self._compiled_step is None:
            if self.integrator == "jax_rk4":
                self._compiled_step = jax.jit(
                    lambda _x, _u, _p, _params: jax_rk4_integrate(
                        self.rhs,
                        _x,
                        _u,
                        _p,
                        _params,
                        self.timestep,
                        self.integration_steps,
                        saveat_mode=self.saveat_mode,
                    ),
                    donate_argnums=(0,),
                )
            elif self.integrator == "diffrax":
                if self.batch_mode == "vmap":
                    self._compiled_step = jax.jit(
                        lambda _x, _u, _p, _params: vmap_integrate(
                            self.rhs,
                            _x,
                            _u,
                            _p,
                            _params,
                            self.timestep,
                            self.integration_steps,
                            saveat_mode=self.saveat_mode,
                            norm_mode=self.norm_mode,
                            step_mode=self.step_mode,
                        ),
                        donate_argnums=(0,),
                    )
                elif self.batch_mode == "stacked":
                    self._compiled_step = jax.jit(
                        lambda _x, _u, _p, _params: diffrax_integrate_batch(
                            self.rhs,
                            _x,
                            _u,
                            _p,
                            _params,
                            self.timestep,
                            self.integration_steps,
                            saveat_mode=self.saveat_mode,
                            norm_mode=self.norm_mode,
                            step_mode=self.step_mode,
                        ),
                        donate_argnums=(0,),
                    )
                else:
                    raise ValueError(f"Unknown batch_mode: {self.batch_mode}")
            elif self.integrator == "linear":
                if self.saveat_mode != "t1":
                    raise ValueError("Linear integrator only supports saveat_mode='t1'.")
                if self._linear_mats is None:
                    if hasattr(x, "aval"):
                        raise RuntimeError(
                            "Linear integrator matrices must be prepared before JIT. "
                            "Call JaxSimulator.prepare_linear(...) prior to building the step function."
                        )
                    Ad_tmp, Bd_tmp, Cd_tmp = linear_discretize_batch(self.rhs, x, p, params, self.timestep)
                    self._linear_mats = (
                        jnp.array(Ad_tmp, dtype=x.dtype),
                        jnp.array(Bd_tmp, dtype=x.dtype),
                        jnp.array(Cd_tmp, dtype=x.dtype),
                    )
                    self._linear_shapes = (x.shape, p.shape)

                # Matrices passed as explicit args so changing values needs no recompile.
                def _linear_step(_x, _u, _p, _Ad, _Bd, _Cd):
                    x_next = jnp.einsum("bij,bj->bi", _Ad, _x) + _Bd * _u[:, None]
                    if _p.shape[1] > 0:
                        x_next = x_next + jnp.einsum("bij,bj->bi", _Cd, _p)
                    return x_next

                self._compiled_step = jax.jit(_linear_step, donate_argnums=(0,))
            else:
                raise ValueError(f"Unknown integrator: {self.integrator}")
        if self.integrator == "linear":
            mats = linear_mats if linear_mats is not None else self._linear_mats
            Ad, Bd, Cd = mats
            ys = self._compiled_step(x, u, p, Ad, Bd, Cd)
        else:
            ys = self._compiled_step(x, u, p, params)
        if self.integrator == "diffrax" and self.batch_mode == "stacked" and ys.ndim == 3:
            ys = jnp.swapaxes(ys, 0, 1)
        if self.saveat_mode == "t1" and ys.ndim == 2:
            ys = ys[:, None, :]
        next_state = ys[:, -1, :]
        return ys, next_state

    def calc_cost(self, T_hp_ret_series, T_hp_sup, T_amb, params):
        return calc_hp(T_hp_ret_series, T_hp_sup, T_amb, self.timestep, params)

    def apply_hp_constraints(self, T_hp_sup, T_hp_ret, params):
        return check_hp(T_hp_sup, T_hp_ret, params)
