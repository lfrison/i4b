"""Evaluation helpers for intrusive PCE MPC experiments.

The helpers in this module are designed for the intrusive-PCE controller path:

- mean-constrained MPC, where only the predicted mean room temperature is
  constrained,
- chance-constrained MPC, where a Gaussian lower quantile is constrained.

They provide consistent post-processing across PCE orders and uncertainty
samples, with metrics focused on solver runtime, objective value, and realized
comfort violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.controller.mpc.casadi_framework import MPC_solver
from src.simulator import Model_simulator


@dataclass(frozen=True)
class IntrusiveControllerCase:
    """One controller formulation used in an intrusive-PCE comparison."""

    label: str
    chance_epsilon: Optional[float] = None


def build_intrusive_mpc_solver(
    *,
    hp_model: Any,
    building_model: Any,
    disturbance_count: int,
    h: int,
    nk: int,
    ws: float = 1.0,
    resultdir: str = "results_intrusive_pce",
    resultfile: str = "intrusive_pce",
    chance_epsilon: Optional[float] = None,
    soft_lower_constraint: bool = False,
    nu: int = 1,
    ns: int = 1,
    nc: int = 3,
) -> MPC_solver:
    """Build an MPC solver configured for intrusive-PCE states."""
    return MPC_solver(
        resultdir=resultdir,
        resultfile=resultfile,
        hp_model=hp_model,
        building_model=building_model,
        nx=len(building_model.state_keys),
        nu=nu,
        npar=disturbance_count,
        ns=ns,
        nc=nc,
        h=h,
        nk=nk,
        ws=ws,
        chance_epsilon=chance_epsilon,
        soft_lower_constraint=soft_lower_constraint,
    )


def extract_intrusive_trajectory(res_x: np.ndarray, mpc: MPC_solver) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract state, control, and slack trajectories from the NLP vector."""
    dim = mpc.dim
    nx, ns, nu, d = dim["nx"], dim["ns"], dim["nu"], dim["d"]
    res_arr = np.asarray(res_x).reshape(-1)

    states = []
    slacks = []
    controls = []
    offset = 0

    for _ in range(mpc.nk):
        states.append(res_arr[offset : offset + nx])
        offset += nx
        slacks.append(res_arr[offset : offset + ns])
        offset += ns

        for _ in range(1, d + 1):
            offset += nx
            offset += ns

        controls.append(res_arr[offset : offset + nu])
        offset += nu

    states.append(res_arr[offset : offset + nx])
    offset += nx
    slacks.append(res_arr[offset : offset + ns])

    return np.vstack(states), np.vstack(controls), np.vstack(slacks)


def room_moments_from_state_traj(x_traj_pc: np.ndarray, intrusive_building: Any, state_name: str = "T_room") -> tuple[np.ndarray, np.ndarray]:
    """Return mean and standard deviation trajectories for one physical state."""
    state_idx = intrusive_building.base_state_keys.index(state_name)
    mean_vals = []
    std_vals = []

    for x_pc in x_traj_pc:
        mean_state = intrusive_building.mean_state(x_pc)
        var_state = intrusive_building.variance_state(x_pc)
        mean_vals.append(mean_state[state_idx])
        std_vals.append(np.sqrt(var_state[state_idx]))

    return np.asarray(mean_vals), np.asarray(std_vals)


def lower_quantile_from_moments(mean_vals: np.ndarray, std_vals: np.ndarray, epsilon: float) -> np.ndarray:
    """Evaluate the lower Gaussian quantile ``mean - z sigma``."""
    z_value = NormalDist().inv_cdf(1.0 - epsilon)
    return np.asarray(mean_vals) - z_value * np.asarray(std_vals)


def realized_stage_cost(hp_model: Any, building_model: Any, x: np.ndarray, u: float, p: np.ndarray) -> float:
    """Evaluate the stage cost on a realized deterministic state trajectory."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    u_val = float(np.asarray(u).reshape(-1)[0])

    # Fall back to the last state for legacy Building instances still loaded in
    # notebook kernels that predate the generic return_temperature_expr hook.
    if hasattr(building_model, "return_temperature_expr"):
        T_RL = float(building_model.return_temperature_expr(x_arr))
    else:
        T_RL = float(x_arr[-1])
    T_amb = float(p_arr[0])
    grid = float(p_arr[-1])
    COP = hp_model.COP(u_val, T_amb)
    Qth = hp_model.mdot_HP * hp_model.c_water * (u_val - T_RL) / 1000.0
    return float(Qth / (COP * 100.0) * grid)


def summarize_intrusive_closed_loop(traj: pd.DataFrame) -> dict[str, float | int | str]:
    """Aggregate one closed-loop run into runtime, objective, and violation metrics."""
    if traj.empty:
        return {
            "controller": "",
            "pce_order": -1,
            "sample_index": -1,
            "n_steps": 0,
            "solver_calls": 0,
            "runtime_total_s": 0.0,
            "runtime_mean_s": 0.0,
            "objective_total": 0.0,
            "objective_mean_per_step": 0.0,
            "realized_objective_total": 0.0,
            "realized_objective_mean_per_step": 0.0,
            "violating_steps": 0,
            "violation_probability": 0.0,
            "mean_violation_C": 0.0,
            "max_violation_C": 0.0,
            "min_predicted_mean_margin_C": 0.0,
            "min_predicted_lower_quantile_margin_C": 0.0,
            "min_formulation_margin_C": 0.0,
        }

    violation = np.maximum(traj["T_set_lower"].to_numpy() - traj["T_room_true"].to_numpy(), 0.0)
    runtime = traj["solve_runtime_s"].to_numpy()
    objective = traj["objective_value"].to_numpy()
    realized_objective = (
        traj["realized_objective_value"].to_numpy()
        if "realized_objective_value" in traj.columns
        else objective
    )

    return {
        "controller": str(traj["controller"].iloc[0]),
        "pce_order": int(traj["pce_order"].iloc[0]),
        "sample_index": int(traj["sample_index"].iloc[0]),
        "n_steps": int(len(traj)),
        "solver_calls": int(len(traj)),
        "runtime_total_s": float(np.sum(runtime)),
        "runtime_mean_s": float(np.mean(runtime)),
        "objective_total": float(np.sum(objective)),
        "objective_mean_per_step": float(np.mean(objective)),
        "realized_objective_total": float(np.sum(realized_objective)),
        "realized_objective_mean_per_step": float(np.mean(realized_objective)),
        "violating_steps": int(np.count_nonzero(violation > 1e-9)),
        "violation_probability": float(np.mean(violation > 1e-9)),
        "mean_violation_C": float(np.mean(violation)),
        "max_violation_C": float(np.max(violation)),
        "min_predicted_mean_margin_C": float(np.min(traj["predicted_mean_margin_C"])),
        "min_predicted_lower_quantile_margin_C": float(np.min(traj["predicted_lower_quantile_margin_C"])),
        "min_formulation_margin_C": float(np.min(traj["formulation_margin_C"])),
    }


def aggregate_intrusive_benchmark(sample_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-sample summaries to one row per controller and PCE order."""
    if sample_summary.empty:
        return sample_summary.copy()

    grouped = (
        sample_summary
        .groupby(["pce_order", "controller"], as_index=False)
        .agg(
            samples=("sample_index", "nunique"),
            n_steps_mean=("n_steps", "mean"),
            runtime_total_s_mean=("runtime_total_s", "mean"),
            runtime_total_s_max=("runtime_total_s", "max"),
            runtime_mean_s_mean=("runtime_mean_s", "mean"),
            objective_total_mean=("objective_total", "mean"),
            objective_total_max=("objective_total", "max"),
            objective_mean_per_step_mean=("objective_mean_per_step", "mean"),
            realized_objective_total_mean=("realized_objective_total", "mean"),
            realized_objective_total_max=("realized_objective_total", "max"),
            realized_objective_mean_per_step_mean=("realized_objective_mean_per_step", "mean"),
            violation_probability_mean=("violation_probability", "mean"),
            violation_probability_max=("violation_probability", "max"),
            mean_violation_C_mean=("mean_violation_C", "mean"),
            max_violation_C_max=("max_violation_C", "max"),
            min_predicted_mean_margin_C_min=("min_predicted_mean_margin_C", "min"),
            min_predicted_lower_quantile_margin_C_min=("min_predicted_lower_quantile_margin_C", "min"),
            min_formulation_margin_C_min=("min_formulation_margin_C", "min"),
        )
    )
    return grouped.sort_values(["pce_order", "controller"]).reset_index(drop=True)


def _normalize_controller_cases(
    controller_cases: Mapping[str, Optional[float]] | Sequence[IntrusiveControllerCase],
) -> list[IntrusiveControllerCase]:
    if isinstance(controller_cases, Mapping):
        return [IntrusiveControllerCase(label=label, chance_epsilon=eps) for label, eps in controller_cases.items()]
    return list(controller_cases)


def benchmark_intrusive_pce_orders(
    *,
    orders: Sequence[int],
    controller_cases: Mapping[str, Optional[float]] | Sequence[IntrusiveControllerCase],
    intrusive_building_factory: Callable[[int], Any],
    true_building_factory: Callable[[np.ndarray], Any],
    nominal_building: Any,
    hp_model: Any,
    disturbances: pd.DataFrame,
    disturbance_cols: Sequence[str],
    x0_det: np.ndarray,
    nk: int,
    h: int,
    closed_loop_steps: int,
    eval_epsilon: float,
    xi_samples: Optional[np.ndarray] = None,
    ws: float = 1.0,
    resultdir: str = "results_intrusive_pce",
    soft_lower_constraint: bool = False,
    nu: int = 1,
    ns: int = 1,
    nc: int = 3,
) -> dict[str, pd.DataFrame]:
    """Benchmark intrusive mean/chance-constrained MPC across PCE orders.

    Parameters
    ----------
    orders :
        PCE orders to benchmark.
    controller_cases :
        Either ``{label: chance_epsilon}`` or ``IntrusiveControllerCase`` items.
        Use ``chance_epsilon=None`` for mean-constrained MPC.
    intrusive_building_factory :
        ``factory(order) -> BuildingPCEIntrusive``.
    true_building_factory :
        ``factory(xi_sample) -> deterministic Building`` used by the simulator.
    xi_samples :
        Array of shape ``(n_samples, n_xi)``. If omitted, a single nominal sample
        ``xi=0`` is used.

    Returns
    -------
    dict
        ``step_metrics`` contains one row per MPC solve, ``sample_summary`` one
        row per closed-loop run, and ``summary`` the aggregate per order and
        controller.
    """
    cases = _normalize_controller_cases(controller_cases)
    x0_det = np.asarray(x0_det, dtype=float).reshape(-1)
    if xi_samples is None:
        first_building = intrusive_building_factory(int(orders[0]))
        xi_samples = np.zeros((1, first_building.uncertain_dim), dtype=float)
    else:
        xi_samples = np.asarray(xi_samples, dtype=float)
        if xi_samples.ndim == 1:
            xi_samples = xi_samples.reshape(1, -1)

    room_idx = nominal_building.state_keys.index("T_room")
    setpoint_col = list(disturbance_cols).index("T_room_set_lower")
    rows: list[dict[str, Any]] = []

    for order in orders:
        intrusive_building = intrusive_building_factory(int(order))
        x0_pc = intrusive_building.lift_deterministic_state(x0_det)

        for controller in cases:
            for sample_index, xi_sample in enumerate(xi_samples):
                # Keep the closed-loop warm start within one sample, but reset it
                # between uncertainty realizations so runtime comparisons stay fair.
                mpc = build_intrusive_mpc_solver(
                    hp_model=hp_model,
                    building_model=intrusive_building,
                    disturbance_count=len(disturbance_cols),
                    h=h,
                    nk=nk,
                    ws=ws,
                    resultdir=resultdir,
                    resultfile=f"intrusive_pce_order{order}_{controller.label.lower().replace(' ', '_')}_sample{sample_index}",
                    chance_epsilon=controller.chance_epsilon,
                    soft_lower_constraint=soft_lower_constraint,
                    nu=nu,
                    ns=ns,
                    nc=nc,
                )
                simulator = Model_simulator(hp_model, true_building_factory(xi_sample), h)
                x_true = x0_det.copy()
                x_pc = x0_pc.copy()

                for step in range(closed_loop_steps):
                    window = disturbances.iloc[step : step + nk + 1].copy()
                    if len(window) < nk + 1:
                        break

                    P = window.loc[:, list(disturbance_cols)].to_numpy()
                    mpc.update_NLP(x_pc)

                    solve_t0 = perf_counter()
                    uk, _, res = mpc.solve_NLP(P, return_res=True)
                    solve_runtime = perf_counter() - solve_t0

                    uk = float(np.asarray(uk).squeeze())
                    objective_value = float(np.asarray(res["f"]).squeeze())
                    realized_objective_value = realized_stage_cost(
                        hp_model,
                        simulator.bldg_model,
                        x_true,
                        uk,
                        P[0],
                    )
                    x_pred_traj, _, s_pred_traj = extract_intrusive_trajectory(res["x"], mpc)
                    mean_room, std_room = room_moments_from_state_traj(x_pred_traj, intrusive_building)
                    lower_quantile = lower_quantile_from_moments(mean_room, std_room, eval_epsilon)

                    next_state = simulator.get_next_state(
                        x_init={key: float(val) for key, val in zip(nominal_building.state_keys, x_true)},
                        uk=uk,
                        pk=window.iloc[0].to_dict(),
                    )["state"]
                    x_true = np.asarray([next_state[key] for key in nominal_building.state_keys], dtype=float)
                    x_pc = np.asarray(x_pred_traj[1], dtype=float)

                    set_low_next = float(P[1, setpoint_col])
                    true_room_next = float(x_true[room_idx])
                    predicted_mean_margin = float(mean_room[1] - set_low_next)
                    predicted_lower_quantile_margin = float(lower_quantile[1] - set_low_next)
                    formulation_margin = (
                        predicted_lower_quantile_margin
                        if controller.chance_epsilon is not None
                        else predicted_mean_margin
                    )

                    row = {
                        "controller": controller.label,
                        "chance_epsilon": controller.chance_epsilon,
                        "pce_order": int(order),
                        "sample_index": int(sample_index),
                        "step": int(step + 1),
                        "time_h": float((step + 1) * h / 3600.0),
                        "solve_runtime_s": float(solve_runtime),
                        "objective_value": objective_value,
                        "realized_objective_value": float(realized_objective_value),
                        "T_room_true": true_room_next,
                        "T_room_mean_pred": float(mean_room[1]),
                        "T_room_std_pred": float(std_room[1]),
                        "T_room_lower_quantile_pred": float(lower_quantile[1]),
                        "T_set_lower": set_low_next,
                        "predicted_mean_margin_C": predicted_mean_margin,
                        "predicted_lower_quantile_margin_C": predicted_lower_quantile_margin,
                        "formulation_margin_C": formulation_margin,
                        "violation_C": float(max(set_low_next - true_room_next, 0.0)),
                        "slack_lower_pred": float(s_pred_traj[1, 0]),
                        "T_hp_sup": uk,
                    }
                    for xi_idx, xi_value in enumerate(np.asarray(xi_sample).reshape(-1)):
                        row[f"xi_{xi_idx}"] = float(xi_value)
                    rows.append(row)

    step_metrics = pd.DataFrame(rows)
    if step_metrics.empty:
        return {
            "step_metrics": step_metrics,
            "sample_summary": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }

    sample_summary = pd.DataFrame(
        summarize_intrusive_closed_loop(group)
        for _, group in step_metrics.groupby(["pce_order", "controller", "sample_index"], sort=True)
    ).sort_values(["pce_order", "controller", "sample_index"]).reset_index(drop=True)

    summary = aggregate_intrusive_benchmark(sample_summary)
    return {
        "step_metrics": step_metrics,
        "sample_summary": sample_summary,
        "summary": summary,
    }


def benchmark_intrusive_open_loop_pce_orders(
    *,
    orders: Sequence[int],
    controller_cases: Mapping[str, Optional[float]] | Sequence[IntrusiveControllerCase],
    intrusive_building_factory: Callable[[int], Any],
    true_building_factory: Callable[[np.ndarray], Any],
    nominal_building: Any,
    hp_model: Any,
    disturbances: pd.DataFrame,
    disturbance_cols: Sequence[str],
    x0_det: np.ndarray,
    nk: int,
    h: int,
    eval_epsilon: float,
    xi_samples: Optional[np.ndarray] = None,
    ws: float = 1.0,
    resultdir: str = "results_intrusive_pce",
    soft_lower_constraint: bool = False,
    nu: int = 1,
    ns: int = 1,
    nc: int = 3,
) -> dict[str, pd.DataFrame]:
    """Benchmark fixed open-loop intrusive MPC plans via Monte Carlo simulation."""
    cases = _normalize_controller_cases(controller_cases)
    x0_det = np.asarray(x0_det, dtype=float).reshape(-1)
    if xi_samples is None:
        first_building = intrusive_building_factory(int(orders[0]))
        xi_samples = np.zeros((1, first_building.uncertain_dim), dtype=float)
    else:
        xi_samples = np.asarray(xi_samples, dtype=float)
        if xi_samples.ndim == 1:
            xi_samples = xi_samples.reshape(1, -1)

    room_idx = nominal_building.state_keys.index("T_room")
    setpoint_col = list(disturbance_cols).index("T_room_set_lower")
    rows: list[dict[str, Any]] = []

    for order in orders:
        intrusive_building = intrusive_building_factory(int(order))
        x0_pc = intrusive_building.lift_deterministic_state(x0_det)
        window = disturbances.iloc[: nk + 1].copy()
        if len(window) < nk + 1:
            continue

        P = window.loc[:, list(disturbance_cols)].to_numpy()

        for controller in cases:
            mpc = build_intrusive_mpc_solver(
                hp_model=hp_model,
                building_model=intrusive_building,
                disturbance_count=len(disturbance_cols),
                h=h,
                nk=nk,
                ws=ws,
                resultdir=resultdir,
                resultfile=f"open_loop_intrusive_pce_order{order}_{controller.label.lower().replace(' ', '_')}",
                chance_epsilon=controller.chance_epsilon,
                soft_lower_constraint=soft_lower_constraint,
                nu=nu,
                ns=ns,
                nc=nc,
            )
            mpc.update_NLP(x0_pc)

            solve_t0 = perf_counter()
            uk0, _, res = mpc.solve_NLP(P, return_res=True)
            solve_runtime = perf_counter() - solve_t0

            x_pred_traj, u_seq, s_pred_traj = extract_intrusive_trajectory(res["x"], mpc)
            mean_room, std_room = room_moments_from_state_traj(x_pred_traj, intrusive_building)
            lower_quantile = lower_quantile_from_moments(mean_room, std_room, eval_epsilon)
            u_seq = np.asarray(u_seq, dtype=float).reshape(-1)
            open_loop_objective = float(np.asarray(res["f"]).squeeze())

            for sample_index, xi_sample in enumerate(xi_samples):
                simulator = Model_simulator(hp_model, true_building_factory(xi_sample), h)
                x_true = x0_det.copy()

                for step in range(min(nk, len(u_seq))):
                    uk = float(u_seq[step])
                    realized_objective_value = realized_stage_cost(
                        hp_model,
                        simulator.bldg_model,
                        x_true,
                        uk,
                        P[step],
                    )

                    next_state = simulator.get_next_state(
                        x_init={key: float(val) for key, val in zip(nominal_building.state_keys, x_true)},
                        uk=uk,
                        pk=window.iloc[step].to_dict(),
                    )["state"]
                    x_true = np.asarray([next_state[key] for key in nominal_building.state_keys], dtype=float)

                    set_low_next = float(P[step + 1, setpoint_col])
                    true_room_next = float(x_true[room_idx])
                    predicted_mean_margin = float(mean_room[step + 1] - set_low_next)
                    predicted_lower_quantile_margin = float(lower_quantile[step + 1] - set_low_next)
                    formulation_margin = (
                        predicted_lower_quantile_margin
                        if controller.chance_epsilon is not None
                        else predicted_mean_margin
                    )

                    row = {
                        "controller": controller.label,
                        "chance_epsilon": controller.chance_epsilon,
                        "pce_order": int(order),
                        "sample_index": int(sample_index),
                        "step": int(step + 1),
                        "time_h": float((step + 1) * h / 3600.0),
                        "solve_runtime_s": float(solve_runtime if step == 0 else 0.0),
                        "objective_value": float(open_loop_objective if step == 0 else 0.0),
                        "realized_objective_value": float(realized_objective_value),
                        "T_room_true": true_room_next,
                        "T_room_mean_pred": float(mean_room[step + 1]),
                        "T_room_std_pred": float(std_room[step + 1]),
                        "T_room_lower_quantile_pred": float(lower_quantile[step + 1]),
                        "T_set_lower": set_low_next,
                        "predicted_mean_margin_C": predicted_mean_margin,
                        "predicted_lower_quantile_margin_C": predicted_lower_quantile_margin,
                        "formulation_margin_C": formulation_margin,
                        "violation_C": float(max(set_low_next - true_room_next, 0.0)),
                        "slack_lower_pred": float(s_pred_traj[step + 1, 0]),
                        "T_hp_sup": uk,
                    }
                    for xi_idx, xi_value in enumerate(np.asarray(xi_sample).reshape(-1)):
                        row[f"xi_{xi_idx}"] = float(xi_value)
                    rows.append(row)

    step_metrics = pd.DataFrame(rows)
    if step_metrics.empty:
        return {
            "step_metrics": step_metrics,
            "sample_summary": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }

    sample_summary = pd.DataFrame(
        summarize_intrusive_closed_loop(group)
        for _, group in step_metrics.groupby(["pce_order", "controller", "sample_index"], sort=True)
    ).sort_values(["pce_order", "controller", "sample_index"]).reset_index(drop=True)

    summary = aggregate_intrusive_benchmark(sample_summary)
    return {
        "step_metrics": step_metrics,
        "sample_summary": sample_summary,
        "summary": summary,
    }
