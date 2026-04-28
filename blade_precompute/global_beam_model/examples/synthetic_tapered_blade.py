"""
Synthetic tapered-blade stiffness fixture — solver regression testing only.

All stiffness values (EA, EIy, EIz, GJ, kAy, kAz, Kww) are polynomial *taper* laws
with magnitudes tuned for numerical stability in the 7-DOF solver (same order as
integration-test / GBT smoke templates). They are NOT derived from real sections.

For production use, build :class:`~blade_precompute.global_beam_model.core.types.SectionStation`
rows from ``section_properties`` outputs (``stations_from_arrays``) or from GBT in ``examples/section_beam_model``.

Public API
----------
``smoke_model`` / ``_smoke_model``
    Build the :class:`~blade_precompute.global_beam_model.BeamModel` (no solve).
``default_beam_loads``
    Root-clamped BCs and uniform lateral distributed load (same pattern as CLI smoke).
``default_tip_loads``
    Root-clamped BCs and a lateral tip force (+y); used by ``run_synthetic_tapered_convergence_case`` by default.
``default_solver_options_for_synthetic_tapered``
    Solver defaults tuned so the synthetic case converges reliably.
``run_synthetic_tapered_convergence_case``
    One-shot solve (default: tip load; pass ``q_y_Npm=...`` for UDL) returning :class:`~blade_precompute.global_beam_model.core.types.BeamSolveResult`.
``convergence_verdict`` / ``print_convergence_summary``
    Shared rules for examples and pytest (SciPy flag + residual threshold + history sanity).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.core.types import BeamLoads, BeamSolveResult, BoundaryCondition, SolverOptions
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays

# Default lateral line load [N/m] in **global** axes (see ``assembly.external_load_vector``).
# Large UDL + coarse taper on a straight 12 m cantilever is stiff in the 7-DOF corotational
# NR path; defaults are chosen so ``run_synthetic_tapered_convergence_case`` is a fast regression.
# For heavier smoke (e.g. ~350 N/m), pass ``q_y_Npm=...`` and/or more load steps / lower relax_factor.
_DEFAULT_Q_Y_NPM = 45.0
# Conservative regression bound (same order as test_precompute_example_blade10_beam_converges).
_DEFAULT_RESIDUAL_THRESHOLD = 1.0
RESIDUAL_THRESHOLD_REGRESSION = _DEFAULT_RESIDUAL_THRESHOLD


def _tapered_K7(z_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal K6/K7 taper laws with magnitudes comparable to ``test_integration`` / GBT smoke.

    Very large axial/shear entries (e.g. EA ~ 1e10) vs warping Kww ~ 1e4–1e6 badly condition
    the 7-DOF Newton system for this mesh; keep the same *relative* taper but SI scales
    closer to the stable template ``diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6])`` + ``Kww=1e4``.
    """
    z0, z1 = float(z_nodes[0]), float(z_nodes[-1])
    n = z_nodes.shape[0]
    mats6 = np.zeros((n, 6, 6), dtype=np.float64)
    mats7 = np.zeros((n, 7, 7), dtype=np.float64)
    for i, z in enumerate(z_nodes):
        t = (z - z0) / (z1 - z0 + 1e-30)
        EA = 1.0e8 * (1.0 - 0.55 * t)
        EIy = 1.0e6 * (1.0 - 0.45 * t)
        EIz = 1.2e6 * (1.0 - 0.40 * t)
        GJ = 1.0e5 * (1.0 - 0.35 * t)
        kAy = 1.0e6 * (1.0 - 0.2 * t)
        kAz = 1.0e6 * (1.0 - 0.2 * t)
        Kww = 1.0e4 * (1.0 - 0.25 * t)
        mats6[i, 0, 0] = EA
        mats6[i, 1, 1] = EIy
        mats6[i, 2, 2] = EIz
        mats6[i, 3, 3] = GJ
        mats6[i, 4, 4] = kAy
        mats6[i, 5, 5] = kAz
        mats7[i, :6, :6] = mats6[i]
        mats7[i, 6, 6] = Kww
    return mats6, mats7


def _smoke_model() -> bm.BeamModel:
    L = 12.0
    n_st = 5
    z_st = np.linspace(0.0, L, n_st)
    # Straight reference line + zero initial curvature: a curved prebend reference with
    # large distributed lateral load was routinely driving the corotational NR into
    # divergence for this coarse station layout; taper is carried only in K6/K7.
    r_ref = np.stack([np.zeros_like(z_st), np.zeros_like(z_st), z_st], axis=1)
    kappa0 = np.zeros((n_st, 3), dtype=np.float64)
    geom = BladeGeometry(
        z_stations=z_st,
        r_ref=r_ref,
        kappa0=kappa0,
        chord=np.ones(n_st) * 0.5,
        twist=np.zeros(n_st),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2)),
        subcomponent_materials={},
        chi0=np.zeros(n_st),
    )
    n_nodes = 11
    K6s, K7s = _tapered_K7(z_st)
    stations = stations_from_arrays(z_st, K6s, K7s)
    return bm.BeamModel.from_blade_geometry(geom, n_nodes, stations, span_axis=2)


def smoke_model() -> bm.BeamModel:
    """Public alias for the synthetic tapered :class:`~blade_precompute.global_beam_model.BeamModel`."""
    return _smoke_model()


def default_beam_loads(model: bm.BeamModel, q_y_Npm: float = _DEFAULT_Q_Y_NPM) -> BeamLoads:
    """Uniform distributed lateral load (global +y) with a fully fixed root (7 DOF)."""
    n = model.n_nodes
    q_line = np.zeros((len(model.elements), 3), dtype=np.float64)
    q_line[:, 1] = float(q_y_Npm)
    return BeamLoads(
        nodal_F=np.zeros((n, 3)),
        nodal_M=np.zeros((n, 3)),
        distributed_q=q_line,
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )


def default_tip_loads(model: bm.BeamModel, fy_tip_N: float = 25.0) -> BeamLoads:
    """Lateral tip force (global +y) with a fully fixed root — robust for the regression solve."""
    n = model.n_nodes
    F = np.zeros((n, 3), dtype=np.float64)
    F[-1, 1] = float(fy_tip_N)
    return BeamLoads(
        nodal_F=F,
        nodal_M=np.zeros((n, 3)),
        distributed_q=None,
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )


def default_solver_options_for_synthetic_tapered(
    *,
    max_iter: int = 25,
    n_load_steps: int = 12,
    verbose: bool = False,
    **kwargs: Any,
) -> SolverOptions:
    """Solver options tuned for the synthetic tapered stiffness law (proportional loading + stabilization).

    Slightly more under-relaxation and Tikhonov on the tangent than the GBT precompute recipe:
    the straight synthetic cantilever + uniform lateral load is a stiff geometric path for
    the material-only Hessian unless increments are kept moderate.
    """
    base: dict[str, Any] = {
        "max_iter": max_iter,
        "tol_res": 5e-2,
        "tol_res_rel": 5e-3,
        "tol_du": 1e-6,
        "n_gauss": 2,
        "n_load_steps": n_load_steps,
        "full_fd_hessian": False,
        "spin_stabilization": 1e-5,
        "warping_stabilization": 1e-3,
        "relax_factor": 0.72,
        "tangent_rho": 0.0,
        "verbose": verbose,
        "line_search": False,
        "tol_res_rel_rhs": 0.035,
        "cap_floor_rel": 0.055,
    }
    base.update(kwargs)
    return SolverOptions(**base)


def _first_last_residual(history: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not history:
        return None, None
    r0 = history[0].get("residual_norm")
    r1 = history[-1].get("residual_norm")
    return (float(r0) if r0 is not None else None, float(r1) if r1 is not None else None)


def convergence_verdict(
    res: BeamSolveResult,
    *,
    residual_threshold: float = _DEFAULT_RESIDUAL_THRESHOLD,
) -> dict[str, Any]:
    """Deterministic checks shared by examples and tests (not a substitute for solver internals)."""
    r_first, r_last = _first_last_residual(res.iteration_history)
    ok_solver = bool(res.converged)
    ok_finite = np.isfinite(res.residual_norm) and np.all(np.isfinite(res.nodal_positions))
    ok_residual = float(res.residual_norm) < float(residual_threshold)
    ok_history = len(res.iteration_history) > 0
    # Not part of ``ok``: proportional loading makes early |res| tiny vs full-load NR tail.
    trend_ok: bool | None = None
    if r_first is not None and r_last is not None and r_first > 0:
        trend_ok = bool(r_last <= r_first * (1.0 + 1e-6))
    return {
        "ok": bool(ok_solver and ok_finite and ok_residual and ok_history),
        "converged": ok_solver,
        "residual_below_threshold": ok_residual,
        "residual_threshold": float(residual_threshold),
        "residual_norm": float(res.residual_norm),
        "n_iterations": int(res.n_iterations),
        "history_nonempty": ok_history,
        "residual_first": r_first,
        "residual_last": r_last,
        "residual_first_le_last_relaxed": trend_ok,
        "nodal_positions_finite": bool(ok_finite),
    }


def print_convergence_summary(res: BeamSolveResult, model: bm.BeamModel | None = None) -> None:
    """Print a short convergence summary (callback / CLI friendly)."""
    r_first, r_last = _first_last_residual(res.iteration_history)
    print("Synthetic tapered beam — convergence summary:")
    print(f"  converged={res.converged}  n_iterations={res.n_iterations}  |res|={res.residual_norm:.6e}")
    if r_first is not None and r_last is not None:
        print(f"  iteration_history residual_norm: first={r_first:.6e}  last={r_last:.6e}")
    if model is not None:
        tip = res.nodal_positions[-1] - model.X_ref[-1]
        print(f"  tip displacement [m]: {tip}")
    verdict = convergence_verdict(res)
    print(f"  verdict ok={verdict['ok']} (threshold |res|<{verdict['residual_threshold']})")


def run_synthetic_tapered_convergence_case(
    *,
    q_y_Npm: float | None = None,
    fy_tip_N: float = 25.0,
    solver_options: SolverOptions | None = None,
    print_summary: bool = False,
    **solver_kw: Any,
) -> BeamSolveResult:
    """Build the synthetic model, apply loads, and run ``solve_static``.

    By default uses a **tip lateral force** (stable for this 7-DOF path with a material-only
    Hessian). Pass ``q_y_Npm`` for a uniform distributed lateral load (may need more load
    increments / lower ``relax_factor`` than the defaults here).
    """
    model = _smoke_model()
    loads = (
        default_beam_loads(model, q_y_Npm=float(q_y_Npm))
        if q_y_Npm is not None
        else default_tip_loads(model, fy_tip_N=fy_tip_N)
    )
    opts = solver_options
    if opts is None and solver_kw:
        opts = default_solver_options_for_synthetic_tapered(**solver_kw)
    elif opts is None:
        opts = default_solver_options_for_synthetic_tapered()
    res = bm.BeamAnalysis(model).solve_static(loads, options=opts)
    if print_summary:
        print_convergence_summary(res, model=model)
    return res
