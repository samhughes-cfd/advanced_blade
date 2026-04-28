"""Per-SLSQP-iteration logging payloads and optional ``.npz`` dumps (diagnostics only).

RunLogger events (``blade_precompute._utils.run_logging``):

- ``optimizer.iteration`` — one line per SLSQP callback. Core keys include masses, KS
  slacks, thickness vectors, and ``delta_objective``. When a previous evaluation exists,
  thickness deltas (``delta_t_*``, ``stations_changed_*``) and optional
  ``constraint_deltas_max_abs_ks_slack`` are added. Beam driver lines add ``beam_*`` scalars
  when ``DesignProblem.iteration_log_beam_summary`` is True and ``beam_state`` is a
  :class:`~blade_precompute.global_beam_model.core.types.BeamSolveResult``. Optional
  ``beam_nr_history_trunc`` appears when ``iteration_log_beam_nr_history`` is True.
  ``orientation_mix_id`` correlates inner iterations with ``optimizer.orientation_combo``
  from the outer orientation enumeration.

- ``optimizer.orientation_combo`` — emitted once per discrete orientation combo in
  :meth:`BladeOptimizer.run_with_orientation` (mix dict, combo index, inner success).

- ``optimizer.setup`` — once per :meth:`BladeOptimizer.run`, lists **binding** scalar KS
  inequality ids passed to SciPy, ``stress_recovery_mode``, monotone-inequality count, and
  multistart settings. Logged FIs in ``optimizer.iteration`` that are *not* listed in
  ``slsqp_scalar_ineq_ks_ids`` are diagnostic-only for that run mode.

Machine-readable field list: ``write_iteration_payload_schema`` → ``iteration_payload_schema.json``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from blade_precompute._utils.jsonutil import to_jsonable
from blade_precompute.section_optimisation.core.types import DesignEvaluation, DesignProblem

_EPS_LOG = 1e-30


def per_station_max_fi_hashin(fi_hashin: np.ndarray) -> np.ndarray:
    """Max Hashin FI over plies (and subcomponents) at each station. Shape ``(n_stations,)``."""
    fh = np.asarray(fi_hashin, dtype=np.float64)
    if fh.size == 0:
        return np.array([], dtype=np.float64)
    return np.max(fh, axis=tuple(range(1, fh.ndim))).astype(np.float64, copy=False)


def per_station_max_fi_vm(fi_vm: np.ndarray) -> np.ndarray:
    """Max von Mises FI at each station (reduces over trailing axes). Shape ``(n_stations,)``."""
    fvm = np.asarray(fi_vm, dtype=np.float64)
    if fvm.size == 0:
        return np.array([], dtype=np.float64)
    if fvm.ndim == 1:
        return fvm.astype(np.float64, copy=False).ravel()
    return np.max(fvm, axis=tuple(range(1, fvm.ndim))).astype(np.float64, copy=False)


def ks_slack_dict(ev: DesignEvaluation, problem: DesignProblem) -> dict[str, float | None]:
    """Aggregated KS values and constraint slacks for plotting (matches optimiser logging)."""
    rho = float(problem.ks_rho)
    slack_h, slack_vm, slack_m4, ks_h, ks_vm_v, ks_m4_v = _ks_slack_hashin_vm_mitc4(ev, rho)
    rho_b = float(getattr(problem, "ks_rho_buckling", 25.0))
    slack_pb = _slack_panel_buckling(ev, rho_b)
    out: dict[str, float | None] = {
        "slack_ks_hashin": float(slack_h),
        "slack_ks_vm": float(slack_vm) if slack_vm is not None else None,
        "slack_ks_mitc4": float(slack_m4) if slack_m4 is not None else None,
        "ks_hashin": float(ks_h),
        "ks_vm": float(ks_vm_v) if ks_vm_v is not None else None,
        "ks_mitc4": float(ks_m4_v) if ks_m4_v is not None else None,
        "slack_ks_panel_buckling": float(slack_pb) if slack_pb is not None else None,
    }
    return out


def objective_scalar(ev: DesignEvaluation, objective: str) -> float:
    """Match :meth:`BladeOptimizer.run` objective passed to ``scipy.optimize.minimize``."""
    if str(objective) == "max_specific_stiffness":
        return math.log(ev.mass + _EPS_LOG) - math.log(ev.stiffness_metric + _EPS_LOG)
    return float(ev.mass)


def _ks_slack_hashin_vm_mitc4(
    ev: DesignEvaluation, rho: float
) -> tuple[float, float | None, float | None, float, float | None, float | None]:
    from blade_precompute.section_optimisation.engine.aggregation import ks_aggregate

    ks_h = float(ks_aggregate(ev.fi_hashin, rho))
    slack_h = 1.0 - ks_h
    slack_vm: float | None = None
    ks_vm_v: float | None = None
    if ev.fi_vm.size:
        ks_vm_v = float(ks_aggregate(ev.fi_vm, rho))
        slack_vm = 1.0 - ks_vm_v
    slack_m4: float | None = None
    ks_m4_v: float | None = None
    if ev.fi_mitc4 is not None and ev.fi_mitc4.size:
        ks_m4_v = float(ks_aggregate(ev.fi_mitc4, rho))
        slack_m4 = 1.0 - ks_m4_v
    return slack_h, slack_vm, slack_m4, ks_h, ks_vm_v, ks_m4_v


def _slack_panel_buckling(ev: DesignEvaluation, rho_b: float) -> float | None:
    if ev.fi_panel_buckling is None or not np.asarray(ev.fi_panel_buckling).size:
        return None
    from blade_precompute.section_optimisation.engine.aggregation import ks_aggregate as _ks

    ks_pb = float(_ks(ev.fi_panel_buckling, rho_b))
    return 1.0 - ks_pb


def _thickness_delta_block(
    prev_ev: DesignEvaluation,
    ev: DesignEvaluation,
    tol_m: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tol = float(tol_m)
    for role, p_arr, c_arr in (
        ("skin", prev_ev.dv.t_skin, ev.dv.t_skin),
        ("cap", prev_ev.dv.t_cap, ev.dv.t_cap),
        ("web", prev_ev.dv.t_web, ev.dv.t_web),
    ):
        dp = np.asarray(p_arr, dtype=np.float64).ravel()
        dc = np.asarray(c_arr, dtype=np.float64).ravel()
        if dp.shape != dc.shape:
            continue
        d = dc - dp
        out[f"delta_t_{role}_max_abs_m"] = float(np.max(np.abs(d))) if d.size else 0.0
        out[f"delta_t_{role}_l2_m"] = float(np.linalg.norm(d)) if d.size else 0.0
        changed = [int(i) for i in range(d.size) if abs(float(d[i])) > tol]
        out[f"stations_changed_{role}"] = changed
    return out


def _constraint_slack_delta_max(
    prev_ev: DesignEvaluation,
    ev: DesignEvaluation,
    problem: DesignProblem,
) -> float | None:
    rho = float(problem.ks_rho)
    sh_p, svm_p, sm4_p, _, _, _ = _ks_slack_hashin_vm_mitc4(prev_ev, rho)
    sh_c, svm_c, sm4_c, _, _, _ = _ks_slack_hashin_vm_mitc4(ev, rho)
    rho_b = float(getattr(problem, "ks_rho_buckling", 25.0))
    parts: list[float] = [abs(sh_c - sh_p)]
    if svm_p is not None and svm_c is not None:
        parts.append(abs(svm_c - svm_p))
    if sm4_p is not None and sm4_c is not None:
        parts.append(abs(sm4_c - sm4_p))
    pb_p = _slack_panel_buckling(prev_ev, rho_b)
    pb_c = _slack_panel_buckling(ev, rho_b)
    if pb_p is not None and pb_c is not None:
        parts.append(abs(pb_c - pb_p))
    return float(max(parts)) if parts else None


def _beam_iteration_payload(ev: DesignEvaluation, problem: DesignProblem) -> dict[str, Any]:
    if not bool(getattr(problem, "iteration_log_beam_summary", True)):
        return {}
    bs = ev.beam_state
    if bs is None:
        return {}
    try:
        from blade_precompute.global_beam_model.core.types import BeamSolveResult
    except ImportError:
        return {}
    if not isinstance(bs, BeamSolveResult):
        return {}
    res = np.asarray(bs.resultants, dtype=np.float64)
    strains = np.asarray(bs.strains, dtype=np.float64)
    out: dict[str, Any] = {
        "beam_converged": bool(bs.converged),
        "beam_n_iterations": int(bs.n_iterations),
        "beam_final_residual_norm": float(bs.residual_norm),
        "beam_resultants_norm": float(np.linalg.norm(res)) if res.size else 0.0,
        "beam_strains_norm": float(np.linalg.norm(strains)) if strains.size else 0.0,
    }
    hist = getattr(bs, "iteration_history", None) or []
    if hist:
        resids = [float(h.get("residual_norm", 0.0)) for h in hist if isinstance(h, dict)]
        dus = [float(h.get("displacement_norm", 0.0)) for h in hist if isinstance(h, dict)]
        out["beam_nr_iters"] = len(hist)
        out["beam_nr_residual_max"] = float(max(resids)) if resids else 0.0
        out["beam_nr_du_max"] = float(max(dus)) if dus else 0.0
    else:
        out["beam_nr_iters"] = 0
        out["beam_nr_residual_max"] = 0.0
        out["beam_nr_du_max"] = 0.0

    if bool(getattr(problem, "iteration_log_beam_nr_history", False)) and hist:

        def _json_row(h: dict[str, Any]) -> dict[str, Any]:
            row: dict[str, Any] = {}
            for k, v in h.items():
                if isinstance(v, bool):
                    row[k] = v
                elif isinstance(v, (int, float)):
                    row[k] = float(v)
                elif v is None:
                    row[k] = None
                else:
                    row[k] = v
            return row

        n = len(hist)
        if n <= 6:
            trunc = [_json_row(h) for h in hist if isinstance(h, dict)]
        else:
            first = [_json_row(h) for h in hist[:3] if isinstance(h, dict)]
            last = [_json_row(h) for h in hist[-3:] if isinstance(h, dict)]
            trunc = first + [{"_truncated": True, "_omitted": int(n - 6)}] + last
        out["beam_nr_history_trunc"] = trunc
    return out


def _k7_diag_spanwise_stats(k7_stack: np.ndarray) -> dict[str, Any]:
    """Per diagonal index (0..6), min/median/max of K7[j,j] across stations."""
    k = np.asarray(k7_stack, dtype=np.float64)
    if k.ndim != 3 or k.shape[1] != 7 or k.shape[2] != 7:
        return {}
    n_s = k.shape[0]
    out: dict[str, Any] = {}
    for j in range(7):
        col = np.array([float(k[i, j, j]) for i in range(n_s)], dtype=np.float64)
        out[f"k7_diag_{j}_min"] = float(np.min(col))
        out[f"k7_diag_{j}_median"] = float(np.median(col))
        out[f"k7_diag_{j}_max"] = float(np.max(col))
    return out


def collect_k7_stack_for_npz(evaluator: Any) -> np.ndarray | None:
    """Stack ``SectionSolveResult.K7`` per station → ``(n_s, 7, 7)`` for NPZ / spanwise stats."""
    caches = getattr(evaluator, "_caches", None)
    if not caches:
        return None
    mats: list[np.ndarray] = []
    for c in caches:
        res = getattr(c, "result", None)
        if res is None:
            return None
        k7 = np.asarray(res.K7, dtype=np.float64)
        if k7.shape != (7, 7):
            return None
        mats.append(k7)
    if not mats:
        return None
    return np.stack(mats, axis=0)


def beam_nr_residual_tail_array(ev: DesignEvaluation, *, k: int) -> np.ndarray | None:
    """Last *k* NR ``residual_norm`` values from ``BeamSolveResult.iteration_history``."""
    if k <= 0:
        return None
    bs = ev.beam_state
    if bs is None:
        return None
    try:
        from blade_precompute.global_beam_model.core.types import BeamSolveResult
    except ImportError:
        return None
    if not isinstance(bs, BeamSolveResult):
        return None
    hist = getattr(bs, "iteration_history", None) or []
    if not hist:
        return None
    resids = [float(h.get("residual_norm", 0.0)) for h in hist if isinstance(h, dict)]
    if not resids:
        return None
    tail = resids[-k:]
    return np.asarray(tail, dtype=np.float64)


def top_k_hashin_hotspots(
    fi_hashin: np.ndarray,
    k: int,
    composite_names: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Largest ``k`` ply Hashin FI entries with station / subcomponent / ply indices."""
    fh = np.asarray(fi_hashin, dtype=np.float64)
    flat = fh.ravel()
    if flat.size == 0 or k <= 0:
        return []
    kk = min(int(k), int(flat.size))
    idx = np.argpartition(flat, -kk)[-kk:]
    idx = idx[np.argsort(-flat[idx])]
    sh = fh.shape
    out: list[dict[str, Any]] = []
    for lin in idx:
        si, ci, pi = np.unravel_index(int(lin), sh)
        name = None
        if composite_names is not None and 0 <= int(ci) < len(composite_names):
            name = str(composite_names[int(ci)])
        out.append(
            {
                "station": int(si),
                "subcomp_index": int(ci),
                "subcomp_name": name,
                "ply": int(pi),
                "fi_hashin": float(flat[int(lin)]),
            }
        )
    return out


def counts_hashin_unity(fi_hashin: np.ndarray) -> dict[str, int]:
    """Counts for stations / subcomponents with max ply FI strictly above 1."""
    fh = np.asarray(fi_hashin, dtype=np.float64)
    if fh.size == 0:
        return {"n_stations_max_fi_gt_1": 0, "n_subcomp_max_fi_gt_1": 0}
    if fh.ndim == 3:
        per_st = np.max(fh, axis=(1, 2))
        per_ci = np.max(fh, axis=(0, 2))
    elif fh.ndim == 2:
        per_st = np.max(fh, axis=1)
        per_ci = np.max(fh, axis=0)
    else:
        mx = float(np.max(fh))
        return {"n_stations_max_fi_gt_1": int(mx > 1.0), "n_subcomp_max_fi_gt_1": int(mx > 1.0)}
    return {
        "n_stations_max_fi_gt_1": int(np.sum(per_st > 1.0)),
        "n_subcomp_max_fi_gt_1": int(np.sum(per_ci > 1.0)),
    }


def build_optimizer_iteration_payload(
    ev: DesignEvaluation,
    problem: DesignProblem,
    *,
    iteration: int,
    prev_objective: float | None,
    prev_ev: DesignEvaluation | None = None,
    hotspot_k: int,
    composite_names: Sequence[str] | None,
    isotropic_names: Sequence[str] | None,
    z_stations_m: np.ndarray | None,
    axis_meta_emitted: bool,
    k7_stack: np.ndarray | None = None,
    orientation_mix_id: str | None = None,
) -> dict[str, Any]:
    """JSON-serialisable fields for ``RunLogger.info_event('optimizer.iteration', ...)``."""
    rho = float(problem.ks_rho)
    stress_recovery = str(getattr(problem, "stress_recovery", "mitc4"))
    slack_h, slack_vm, slack_m4, ks_h, ks_vm_v, ks_m4_v = _ks_slack_hashin_vm_mitc4(ev, rho)
    obj = objective_scalar(ev, str(problem.objective))
    fh = np.asarray(ev.fi_hashin, dtype=np.float64)
    fvm = np.asarray(ev.fi_vm, dtype=np.float64)

    per_station_h = per_station_max_fi_hashin(fh).tolist() if fh.size else []
    per_station_vm = per_station_max_fi_vm(fvm).tolist() if fvm.size else []

    per_subcomp_h: list[float] = []
    if fh.ndim == 3:
        per_subcomp_h = np.max(fh, axis=(0, 2)).astype(float).tolist()
    elif fh.ndim == 2:
        per_subcomp_h = np.max(fh, axis=0).astype(float).tolist()
    elif fh.ndim == 1:
        per_subcomp_h = [float(np.max(fh))]

    counts = counts_hashin_unity(fh)
    hotspots = top_k_hashin_hotspots(fh, int(hotspot_k), composite_names)

    log_kw: dict[str, Any] = {
        "iteration": int(iteration),
        "mass_kg": float(ev.mass),
        "stiffness_metric": float(ev.stiffness_metric),
        "specific_stiffness": float(ev.stiffness_metric / max(ev.mass, _EPS_LOG)),
        "objective_value": float(obj),
        "delta_objective": float(obj - prev_objective) if prev_objective is not None else None,
        "max_fi_hashin": float(ev.max_fi_hashin),
        "max_fi_vm": float(ev.max_fi_vm),
        "ks_hashin": float(ks_h),
        "ks_vm": float(ks_vm_v) if ks_vm_v is not None else None,
        "ks_mitc4": float(ks_m4_v) if ks_m4_v is not None else None,
        "slack_ks_hashin": float(slack_h),
        "slack_ks_vm": float(slack_vm) if slack_vm is not None else None,
        "slack_ks_mitc4": float(slack_m4) if slack_m4 is not None else None,
        "max_fi_hashin_per_station": per_station_h,
        "max_fi_vm_per_station": per_station_vm,
        "max_fi_hashin_per_subcomp": per_subcomp_h,
        **counts,
        "top_k_hashin_hotspots": hotspots,
        "t_skin": ev.dv.t_skin.astype(float).tolist(),
        "t_cap": ev.dv.t_cap.astype(float).tolist(),
        "t_web": ev.dv.t_web.astype(float).tolist(),
    }

    ih = int(np.argmax(fh)) if fh.size else 0
    sh = fh.shape
    si, ci, pi = (np.unravel_index(ih, sh) if fh.size else (0, 0, 0))
    log_kw["critical_station"] = int(si)
    log_kw["critical_subcomponent"] = int(ci)
    log_kw["critical_ply"] = int(pi)
    if composite_names is not None and fh.size and 0 <= int(ci) < len(composite_names):
        log_kw["critical_subcomponent_name"] = str(composite_names[int(ci)])

    if ev.fi_mitc4 is not None and ev.fi_mitc4.size:
        log_kw["max_fi_mitc4"] = float(np.max(ev.fi_mitc4))
        log_kw["fi_mitc4_per_station"] = np.asarray(ev.fi_mitc4, dtype=float).ravel().tolist()
    if ev.tip_deflection is not None:
        log_kw["tip_deflection_m"] = float(ev.tip_deflection)

    if not axis_meta_emitted and z_stations_m is not None:
        log_kw["z_stations_m"] = np.asarray(z_stations_m, dtype=float).ravel().tolist()
        if composite_names is not None:
            log_kw["composite_subcomp_names"] = [str(x) for x in composite_names]
        if isotropic_names is not None:
            log_kw["isotropic_subcomp_names"] = [str(x) for x in isotropic_names]

    if ev.k7_cond_stats is not None:
        log_kw.update({"k7_" + k: float(v) for k, v in ev.k7_cond_stats.items()})

    rho_b = float(getattr(problem, "ks_rho_buckling", 25.0))
    if ev.fi_panel_buckling is not None and np.asarray(ev.fi_panel_buckling).size:
        from blade_precompute.section_optimisation.engine.aggregation import ks_aggregate as _ks

        ks_pb = float(_ks(ev.fi_panel_buckling, rho_b))
        log_kw["ks_panel_buckling"] = ks_pb
        log_kw["slack_ks_panel_buckling"] = 1.0 - ks_pb
    if ev.global_buckling_lambdas is not None and np.asarray(ev.global_buckling_lambdas).size:
        log_kw["lambda_crit"] = float(np.min(ev.global_buckling_lambdas))
        lam_min = float(getattr(problem, "global_buckling_lambda_min", 1.5))
        log_kw["n_modes_under_safe"] = int(np.sum(ev.global_buckling_lambdas < lam_min))

    tol_m = float(getattr(problem, "iteration_delta_thickness_tol_m", 1e-9))
    if prev_ev is not None:
        log_kw.update(_thickness_delta_block(prev_ev, ev, tol_m))
        if bool(getattr(problem, "iteration_log_constraint_deltas", True)):
            cd = _constraint_slack_delta_max(prev_ev, ev, problem)
            if cd is not None:
                log_kw["constraint_deltas_max_abs_ks_slack"] = cd
    else:
        for role in ("skin", "cap", "web"):
            log_kw[f"delta_t_{role}_max_abs_m"] = None
            log_kw[f"delta_t_{role}_l2_m"] = None
            log_kw[f"stations_changed_{role}"] = None

    log_kw.update(_beam_iteration_payload(ev, problem))

    if bool(getattr(problem, "iteration_log_k7_spanwise", False)) and k7_stack is not None:
        log_kw.update(_k7_diag_spanwise_stats(k7_stack))

    if orientation_mix_id is not None:
        log_kw["orientation_mix_id"] = str(orientation_mix_id)

    return to_jsonable(log_kw)


def write_iteration_payload_schema(path: Path) -> None:
    """One-time schema file for consumers of ``optimizer.iteration`` and ``*.npz`` keys."""
    schema: dict[str, Any] = {
        "version": 2,
        "events": {
            "optimizer.iteration": "Per SciPy optimiser callback; see optimizer_iteration_fields.",
            "optimizer.setup": "Once before minimise: scipy_method, optimizer_ftol, maxiter, n_restarts_planned, multistart_seed, slsqp_scalar_ineq_ks_ids (binding KS surrogate ids), slsqp_spanwise_monotone_ineq_count, stress_recovery_mode, hashin_constraint_fi_source, von_mises_constraint_fi_source.",
            "optimizer.orientation_combo": "After each inner run in run_with_orientation: mix, combo_index, n_inner_iters, inner_success, final_mass_kg, message.",
        },
        "optimizer_iteration_fields": {
            "iteration": "int, SLSQP callback order (1-based)",
            "objective_value": "float, scalar passed to scipy minimize",
            "delta_objective": "float or null vs previous callback objective",
            "orientation_mix_id": "str or absent; JSON key for active orientation combo in outer loop",
            "delta_t_skin_max_abs_m": "float or null; vs previous iterate",
            "delta_t_skin_l2_m": "float or null",
            "stations_changed_skin": "list[int] or null; station indices with |Δt_skin| > tol",
            "delta_t_cap_* / stations_changed_cap": "same pattern",
            "delta_t_web_* / stations_changed_web": "same pattern",
            "constraint_deltas_max_abs_ks_slack": "float or absent; max |Δslack| across KS surrogates when prev_ev exists",
            "beam_converged": "bool when beam_state is BeamSolveResult and iteration_log_beam_summary",
            "beam_n_iterations": "int",
            "beam_final_residual_norm": "float",
            "beam_resultants_norm": "float Frobenius norm of (n_station,7) resultants",
            "beam_strains_norm": "float Frobenius norm of (n_station,7) strains",
            "beam_nr_iters": "int, len(iteration_history)",
            "beam_nr_residual_max": "float",
            "beam_nr_du_max": "float",
            "beam_nr_history_trunc": "list[dict] when iteration_log_beam_nr_history; first 3 + marker + last 3 NR rows",
            "k7_diag_0_min": "float; per-diagonal K7[j,j] spanwise stats when iteration_log_k7_spanwise and K7_stack provided",
            "k7_diag_*_median / _max": "seven diagonals j=0..6",
            "slack_ks_hashin": "1 - KS(fi_hashin)",
            "top_k_hashin_hotspots": "list of {station, subcomp_index, subcomp_name?, ply, fi_hashin}",
            "t_skin / t_cap / t_web": "design thicknesses [m] per station",
        },
        "iteration_npz_optional_keys": {
            "fi_hashin": "(n_stations, n_comp, n_ply_max)",
            "fi_vm": "(n_stations, n_iso)",
            "resultants": "(n_stations, 7)",
            "strains": "(n_stations, 7) from BeamSolveResult when beam_state present",
            "t_skin_t_cap_t_web": "vectors length n_stations",
            "K7_stack": "(n_stations, 7, 7) when section caches expose K7",
            "beam_nr_residual_tail": "(k,) last NR residual_norm scalars",
            "strip_abd_inv": "(n_stations, n_comp, 6, 6)",
            "strip_q_bar": "(n_stations, n_comp, n_ply_max, 3, 3)",
            "strip_z_ply": "(n_stations, n_comp, n_ply_max)",
            "strip_iso_abd_inv": "(n_stations, n_iso, 6, 6)",
            "fi_mitc4": "(n_stations,) when present",
            "mitc4_panel_abd_station0": "(n_panels, 6, 6) panel-wise ABD for station 0 only when MITC4 ran",
        },
        "spanwise_monotone": "SLSQP linear inequalities t_role[i]-t_role[i+1]>=0 per skin/cap/web when enforce_spanwise_monotone",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def collect_section_clpt_for_npz(evaluator: Any) -> dict[str, np.ndarray]:
    """Stack :class:`SectionSolveResult` laminate arrays from evaluator station caches."""
    caches = getattr(evaluator, "_caches", None)
    if not caches:
        return {}
    n = len(caches)
    r0 = caches[0].result
    if r0 is None:
        return {}
    n_comp = int(r0.ABD_inv.shape[0])
    n_ply = int(r0.Q_bar.shape[1])
    n_iso = int(r0.iso_ABD_inv.shape[0]) if getattr(r0, "iso_ABD_inv", np.zeros(0)).size else 0

    abd = np.zeros((n, n_comp, 6, 6), dtype=np.float64)
    qb = np.zeros((n, n_comp, n_ply, 3, 3), dtype=np.float64)
    zp = np.zeros((n, n_comp, n_ply), dtype=np.float64)
    iso_abd = np.zeros((n, max(n_iso, 0), 6, 6), dtype=np.float64)

    for i in range(n):
        res = caches[i].result
        if res is None:
            continue
        abd[i] = np.asarray(res.ABD_inv, dtype=np.float64)
        qb[i] = np.asarray(res.Q_bar, dtype=np.float64)
        zp[i] = np.asarray(res.z_ply, dtype=np.float64)
        ia = np.asarray(res.iso_ABD_inv, dtype=np.float64)
        if ia.size and n_iso > 0:
            iso_abd[i, : ia.shape[0]] = ia

    return {
        "strip_abd_inv": abd,
        "strip_q_bar": qb,
        "strip_z_ply": zp,
        "strip_iso_abd_inv": iso_abd,
    }


# Backward-compatible name (section midsurface / strip FE CLPT tensors, not the removed stress mode).
collect_strip_clpt_for_npz = collect_section_clpt_for_npz


def write_iteration_npz(
    path: Path,
    *,
    ev: DesignEvaluation,
    strip: Mapping[str, np.ndarray] | None,
    mitc4_station0_panel_abd: np.ndarray | None,
    mitc4_station0_panel_thickness_m: np.ndarray | None,
    mitc4_station0_panel_G_eff: np.ndarray | None,
    mitc4_station0_panel_labels: np.ndarray | None,
    k7_stack: np.ndarray | None = None,
    beam_nr_residual_tail: np.ndarray | None = None,
) -> None:
    """Write one compressed archive for a single callback (arrays only)."""
    payload: dict[str, Any] = {
        "fi_hashin": np.asarray(ev.fi_hashin, dtype=np.float64),
        "fi_vm": np.asarray(ev.fi_vm, dtype=np.float64),
        "resultants": np.asarray(ev.resultants, dtype=np.float64),
        "t_skin": np.asarray(ev.dv.t_skin, dtype=np.float64),
        "t_cap": np.asarray(ev.dv.t_cap, dtype=np.float64),
        "t_web": np.asarray(ev.dv.t_web, dtype=np.float64),
    }
    if ev.fi_mitc4 is not None:
        payload["fi_mitc4"] = np.asarray(ev.fi_mitc4, dtype=np.float64)
    if strip:
        for k, v in strip.items():
            payload[k] = np.asarray(v, dtype=np.float64)
    if mitc4_station0_panel_abd is not None:
        payload["mitc4_panel_abd_station0"] = np.asarray(mitc4_station0_panel_abd, dtype=np.float64)
    if mitc4_station0_panel_thickness_m is not None:
        payload["mitc4_panel_thickness_m_station0"] = np.asarray(
            mitc4_station0_panel_thickness_m, dtype=np.float64
        )
    if mitc4_station0_panel_G_eff is not None:
        payload["mitc4_panel_G_eff_station0"] = np.asarray(mitc4_station0_panel_G_eff, dtype=np.float64)
    if mitc4_station0_panel_labels is not None:
        payload["mitc4_panel_labels_station0"] = np.asarray(mitc4_station0_panel_labels, dtype=object)

    bs = ev.beam_state
    if bs is not None:
        try:
            from blade_precompute.global_beam_model.core.types import BeamSolveResult as _BSR
        except ImportError:
            _BSR = None
        if _BSR is not None and isinstance(bs, _BSR):
            payload["strains"] = np.asarray(bs.strains, dtype=np.float64)
    if k7_stack is not None and np.asarray(k7_stack).size:
        payload["K7_stack"] = np.asarray(k7_stack, dtype=np.float64)
    if beam_nr_residual_tail is not None and np.asarray(beam_nr_residual_tail).size:
        payload["beam_nr_residual_tail"] = np.asarray(beam_nr_residual_tail, dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
