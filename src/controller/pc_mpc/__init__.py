"""Polynomial chaos MPC utilities.

This package intentionally exposes its public API lazily. Eager imports caused a
cycle between ``src.controller.pc_mpc.core`` and ``src.models`` during
subpackage initialisation, which then surfaced as misleading ImportErrors from
notebook imports.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "PcMpcRunner": ".core",
    "PolynomialChaosBasis": ".core",
    "UncertaintyModel": ".core",
    "ensure_disturbance_order": ".core",
    "MultiplicativePlantBuilder": ".core",
    "pc_moments": ".core",
    "simulate_closed_loop_mpc": ".core",
    "PcePropagationResult": ".pce_nonintrusive",
    "build_collocation_nodes": ".pce_nonintrusive",
    "compute_pce_coeffs": ".pce_nonintrusive",
    "IntrusiveControllerCase": ".intrusive_evaluation",
    "aggregate_intrusive_benchmark": ".intrusive_evaluation",
    "benchmark_intrusive_open_loop_pce_orders": ".intrusive_evaluation",
    "benchmark_intrusive_pce_orders": ".intrusive_evaluation",
    "build_intrusive_mpc_solver": ".intrusive_evaluation",
    "extract_intrusive_trajectory": ".intrusive_evaluation",
    "lower_quantile_from_moments": ".intrusive_evaluation",
    "room_moments_from_state_traj": ".intrusive_evaluation",
    "summarize_intrusive_closed_loop": ".intrusive_evaluation",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

