"""
beam_model/solver.py
====================
Static Newton–Raphson driver (7 DOFs per node).
"""

from __future__ import annotations

import json
import logging
from typing import List, Sequence

import numpy as np

from .assembly import (
    assemble_gradient,
    assemble_geometric_stiffness,
    assemble_hessian,
    expand_solution,
    external_load_jacobian_fd,
    external_load_vector,
    fixed_dof_set,
    reduce_linear_system,
    _precompute_K7_gp,
)
from .kinematics import quat_align_axis_to_vector, quat_to_rotmat, update_orientation
from .nodal_result_projector import project_beam_strains_resultants_to_nodes
from .postprocess import collect_station_data, compute_reactions
from ..core.types import (
    BeamLoads,
    BeamModel,
    BeamSolveResult,
    BoundaryCondition,
    NodeState,
    SolverOptions,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass  # BeamModel, SolverOptions already imported above

logger = logging.getLogger(__name__)

# Mixed residual tolerance uses several max(...) terms; allow tiny slack vs float noise / tight plateaus.
_RES_TOL_SLACK = 1.0005


def _initialize_nodes(model: BeamModel) -> List[NodeState]:
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n = model.n_nodes
    t_acc = [np.zeros(3, dtype=np.float64) for _ in range(n)]
    cnt = np.zeros(n, dtype=np.int32)
    for el in model.elements:
        i, j = el.node_ids
        t = model.X_ref[j] - model.X_ref[i]
        nt = float(np.linalg.norm(t))
        if nt < 1e-14:
            raise ValueError("Zero-length beam element in reference geometry.")
        t = t / nt
        t_acc[i] += t
        t_acc[j] += t
        cnt[i] += 1
        cnt[j] += 1
    nodes: List[NodeState] = []
    for i in range(n):
        t = t_acc[i] / max(int(cnt[i]), 1)
        nt = float(np.linalg.norm(t))
        if nt < 1e-14:
            t = ex.copy()
        else:
            t = t / nt
        q = quat_align_axis_to_vector(ex, t)
        nodes.append(
            NodeState(
                x=model.X_ref[i].copy(),
                q=q.copy(),
                psi=0.0,
                spin_accum=np.zeros(3, dtype=np.float64),
            )
        )
    return nodes


def _apply_increment(nodes: List[NodeState], du: np.ndarray) -> None:
    n_nodes = len(nodes)
    for i in range(n_nodes):
        base = 7 * i
        dx = du[base : base + 3]
        dth = du[base + 3 : base + 6]
        dpsi = float(du[base + 6])
        nodes[i].x = nodes[i].x + dx
        nodes[i].q = update_orientation(nodes[i].q, dth)
        q_n = float(np.linalg.norm(nodes[i].q))
        if q_n > 1e-14:
            nodes[i].q = nodes[i].q / q_n
        nodes[i].psi += dpsi
        nodes[i].spin_accum = nodes[i].spin_accum + dth


def _restore_nodes(nodes: List[NodeState], snapshot: List[NodeState]) -> None:
    for i, sn in enumerate(snapshot):
        nodes[i].x = sn.x.copy()
        nodes[i].q = sn.q.copy()
        nodes[i].psi = float(sn.psi)
        nodes[i].spin_accum = sn.spin_accum.copy()


def _reduced_equilibrium_residual_norm(
    model: BeamModel,
    nodes: List[NodeState],
    stations: Sequence[object],
    F_ext: np.ndarray,
    free_idx: np.ndarray,
    n_gauss: int,
    fd_eps: float,
    K7_gp=None,
    use_cs: bool = True,
) -> float:
    g = assemble_gradient(model, nodes, stations, n_gauss, fd_eps, K7_gp, use_cs=use_cs)
    rhs = F_ext - g
    rhs_f = rhs[free_idx]
    return float(np.linalg.norm(rhs_f))


def _symmetric_spd_floor(K: np.ndarray, *, eig_floor_rel: float) -> np.ndarray:
    """Symmetrise ``K`` and clamp eigenvalues to ``>= eig_floor_rel * max(lam_max, 1)``."""
    Ks = 0.5 * (K + K.T)
    lam, Q = np.linalg.eigh(Ks)
    lam_max = float(np.max(lam))
    floor = float(eig_floor_rel) * max(lam_max, 1.0)
    lam2 = np.maximum(lam, floor)
    return (Q * lam2) @ Q.T


def _bcs_from_loads(loads: BeamLoads, explicit: List[BoundaryCondition] | None) -> List[BoundaryCondition]:
    out: List[BoundaryCondition] = []
    if loads.bcs:
        out.extend(loads.bcs)
    if explicit:
        out.extend(explicit)
    return out


def _pack_7dofs(
    model: BeamModel, nodes: List[NodeState]
) -> np.ndarray:
    """7-DOF / node: translations, stored spin-accum, warping (arc-length / diagnostics)."""
    n = model.n_nodes
    v = np.empty(7 * n, dtype=np.float64)
    for i in range(n):
        v[7 * i : 7 * i + 3] = nodes[i].x
        v[7 * i + 3 : 7 * i + 6] = nodes[i].spin_accum
        v[7 * i + 6] = float(nodes[i].psi)
    return v


def _tangent_aug_riks(
    K_ff: np.ndarray,
    f_ref_f: np.ndarray,
    d_w_f: np.ndarray,
    lam: float,
    lam0: float,
    c_lam: float,
) -> np.ndarray:
    """
    (n+1) x (n+1) augmented matrix for Riks corrector:
    [ K_ff, -F_ref; 2 d_w^T, 2 c^2 (lam - lam0) ]
    """
    n = K_ff.shape[0]
    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[0:n, 0:n] = K_ff
    A[0:n, n] = -f_ref_f
    A[n, 0:n] = 2.0 * d_w_f
    A[n, n] = 2.0 * (c_lam ** 2) * (lam - lam0)
    return A


def _solve_riks_crisfield(
    model: BeamModel,
    loads: BeamLoads,
    options: SolverOptions,
    nodes: List[NodeState],
    free_idx: np.ndarray,
    ndof: int,
    K7_gp,
    stations: list,
) -> tuple:
    """
    Riks continuation with spherical arc in (free DOF, λ) with ``λ`` scaled by ``c``.
    Reference: ``F_ext = λ F_ref`` with ``F_ref`` at ``λ = 1`` (same assembly as static solve).
    """
    F_ext_ref = external_load_vector(model, loads, options.n_gauss, nodes=nodes)
    f_ref_f = F_ext_ref[free_idx]
    c = max(float(options.arc_length_scale_lambda), 1e-12)
    ds0 = float(options.arc_length_target)
    if ds0 <= 0.0:
        ds0 = 1.0 / max(1, int(options.n_load_steps))
    n_arc_max = max(1, int(options.n_load_steps))
    iteration_history: List[dict] = []
    it_total = 0
    lam = 0.0
    k_t_buck: np.ndarray | None = None
    converged = False

    for _arc in range(n_arc_max):
        if lam >= 1.0 - 1e-9:
            converged = True
            break
        w0 = _pack_7dofs(model, nodes)
        lam0 = float(lam)
        ds = ds0

        Kf = assemble_hessian(
            model, nodes, stations, options.n_gauss, options.fd_eps, options.hess_eps,
            options.full_fd_hessian, options.fd_eps, options.spin_stabilization,
            options.warping_stabilization, K7_gp,
        )[np.ix_(free_idx, free_idx)]
        if options.full_fd_hessian and options.project_fd_hessian_spd:
            Kf = _symmetric_spd_floor(
                Kf, eig_floor_rel=float(options.fd_hessian_eig_floor_rel)
            )
        try:
            du_h = np.linalg.solve(Kf, f_ref_f)
        except np.linalg.LinAlgError:
            du_h, _, _, _ = np.linalg.lstsq(Kf, f_ref_f, rcond=None)
        norm_t = float(np.sqrt(np.dot(du_h, du_h) + c ** 2))
        if norm_t < 1e-20:
            break
        dlam = ds / norm_t
        du_pred = dlam * du_h
        lam = lam0 + dlam
        _apply_increment(nodes, expand_solution(free_idx, du_pred, ndof))

        maxcorr = max(1, int(options.arc_length_max_iter), int(options.max_iter))
        step_ok = False
        for _it in range(1, maxcorr + 1):
            it_total += 1
            F_ext = F_ext_ref * float(lam)
            g = assemble_gradient(
                model, nodes, stations, options.n_gauss, options.fd_eps, K7_gp,
                use_cs=options.use_cs_gradient,
            )
            Rq = g - F_ext
            R_f = Rq[free_idx]
            w = _pack_7dofs(model, nodes)
            d_w = w - w0
            d_w_f = d_w[free_idx]
            c_val = float(np.dot(d_w_f, d_w_f) + c ** 2 * (lam - lam0) ** 2 - ds ** 2)

            Kff = assemble_hessian(
                model, nodes, stations, options.n_gauss, options.fd_eps, options.hess_eps,
                options.full_fd_hessian, options.fd_eps, options.spin_stabilization,
                options.warping_stabilization, K7_gp,
            )[np.ix_(free_idx, free_idx)]
            if options.extract_buckling and k_t_buck is None:
                k_t_buck = Kff.copy()
            if options.full_fd_hessian and options.project_fd_hessian_spd:
                Kff = _symmetric_spd_floor(
                    Kff, eig_floor_rel=float(options.fd_hessian_eig_floor_rel)
                )

            res_norm = float(np.linalg.norm(R_f))
            ref = max(1.0, float(np.linalg.norm(F_ext)))
            tol_mixed = max(float(options.tol_res), float(options.tol_res_rel) * ref)
            tol_c = max(1e-8, 1e-3 * (ds * ds) if ds > 0 else 1e-6)
            if res_norm <= tol_mixed * _RES_TOL_SLACK and abs(c_val) <= tol_c:
                step_ok = True
                break

            b_aug = np.empty(Kff.shape[0] + 1, dtype=np.float64)
            b_aug[:-1] = -R_f
            b_aug[-1] = -c_val
            A_aug = _tangent_aug_riks(Kff, f_ref_f, d_w_f, lam, lam0, c)
            try:
                sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            except np.linalg.LinAlgError:
                break
            du_c = sol[:-1]
            dlamv = float(sol[-1])
            _apply_increment(nodes, expand_solution(free_idx, du_c, ndof))
            lam = lam + dlamv
        if not step_ok:
            if options.verbose:
                print(f"  Riks arc step {_arc+1} failed to converge.")
            break
    if lam >= 1.0 - 1e-6:
        converged = True
    return converged, it_total, iteration_history, k_t_buck


def solve_static(
    model: BeamModel,
    loads: BeamLoads,
    options: SolverOptions | None = None,
    *,
    bcs: List[BoundaryCondition] | None = None,
) -> BeamSolveResult:
    """
    Nonlinear static equilibrium with optional BCs in ``loads.bcs`` or ``bcs``.

    Riks (``use_arc_length``) uses a spherical arc in ``(7n-DOF, λ)`` (with λ scaled by
    ``arc_length_scale_lambda``) for continuation to ``λ = 1``. Proportional loading
    uses ``n_load_steps`` and optional sub-step halving to reach each target load factor.
    """
    if options is None:
        options = SolverOptions()
    bcs_all = _bcs_from_loads(loads, bcs)
    if not model.section_stations or len(model.section_stations) < 2:
        raise ValueError("BeamModel.section_stations must contain at least two stations.")
    stations = model.section_stations

    nodes = _initialize_nodes(model)
    fixed = fixed_dof_set(bcs_all)
    ndof = 7 * model.n_nodes
    free_idx = np.array([i for i in range(ndof) if i not in fixed], dtype=np.int64)
    F_ext_ref = external_load_vector(model, loads, options.n_gauss, nodes=nodes)
    K7_gp = _precompute_K7_gp(model, stations, options.n_gauss)
    if stations:
        k7_diag_rows = []
        for st in stations:
            if st.K7 is not None:
                k7 = np.asarray(st.K7, dtype=np.float64)
            else:
                k6 = np.asarray(st.K6, dtype=np.float64)
                kw = max(float(k6[3, 3]), 1e-6)
                k7 = np.zeros((7, 7), dtype=np.float64)
                k7[:6, :6] = k6
                k7[6, 6] = kw
            k7_diag_rows.append(np.diag(k7))
        k7_diag = np.stack(k7_diag_rows, axis=0)
        logger.info(
            "k7_diagnostics %s",
            json.dumps(
                {
                    "diag_min": np.min(k7_diag, axis=0).tolist(),
                    "diag_max": np.max(k7_diag, axis=0).tolist(),
                    "diag_median": np.median(k7_diag, axis=0).tolist(),
                }
            ),
        )

    iteration_history: List[dict] = []
    converged = False
    res_norm = 1e30
    du_norm = 1e30
    it_total = 0
    k_t_buck_converged: np.ndarray | None = None

    def _tangent_plus_follower() -> np.ndarray:
        Kg = assemble_hessian(
            model,
            nodes,
            stations,
            options.n_gauss,
            options.fd_eps,
            options.hess_eps,
            options.full_fd_hessian,
            options.fd_eps,
            options.spin_stabilization,
            options.warping_stabilization,
            K7_gp,
        )
        if loads.frame == "node_corotated":
            Kg = Kg + external_load_jacobian_fd(
                model,
                loads,
                nodes,
                options.n_gauss,
                options.follower_jacobian_eps,
            )
        return Kg

    if options.use_arc_length:
        if loads.frame == "node_corotated":
            raise ValueError(
                "Riks (use_arc_length) is not supported with node_corotated loads; use proportional load stepping."
            )
        converged, it_total, iteration_history, k_t_buck_converged = _solve_riks_crisfield(
            model, loads, options, nodes, free_idx, ndof, K7_gp, stations
        )
    else:
        n_steps = max(1, int(options.n_load_steps))
        lam_sequence = [k / float(n_steps) for k in range(1, n_steps + 1)]
        prev = 0.0
        converged = True
        min_step = float(options.adaptive_load_min_step)
        max_bis = max(0, int(options.adaptive_load_bisect_max))
        for target in lam_sequence:
            if prev + 1e-12 >= target:
                continue
            while prev < target - 1e-9:
                step = target - prev
                sub_ok = False
                bis = 0
                while step >= min_step and bis <= max_bis:
                    lam = float(prev + step)
                    # Equilibrium at load factor ``prev``; NR attempts for ``lam`` must start here.
                    # Without this restore, a failed / maxed-out NR leaves ``nodes`` corrupted and
                    # bisected sub-steps (smaller ``lam``) incorrectly continue from that state.
                    snap_equilibrium = [n.copy() for n in nodes]
                    F_ext = external_load_vector(
                        model, loads, options.n_gauss, nodes=nodes
                    ) * float(lam)
                    step_conv = False
                    prev_res = 1e300
                    stall = 0
                    rhs_ref0 = None
                    rho_ls = float(options.tangent_rho)
                    last_res = float("inf")
                    bad_streak = 0

                    for it in range(1, options.max_iter + 1):
                        it_total += 1
                        F_ext = external_load_vector(
                            model, loads, options.n_gauss, nodes=nodes
                        ) * float(lam)
                        g = assemble_gradient(
                            model, nodes, stations, options.n_gauss, options.fd_eps, K7_gp,
                            use_cs=options.use_cs_gradient,
                        )
                        rhs = F_ext - g
                        g_norm = float(np.linalg.norm(g))
                        f_ext_norm = float(np.linalg.norm(F_ext))
                        K_ff, rhs_f, _ = reduce_linear_system(
                            _tangent_plus_follower(),
                            rhs,
                            fixed,
                            free_idx,
                        )
                        if options.extract_buckling:
                            k_t_buck_converged = K_ff.copy()
                        if options.full_fd_hessian and options.project_fd_hessian_spd:
                            K_ff = _symmetric_spd_floor(
                                K_ff, eig_floor_rel=float(options.fd_hessian_eig_floor_rel)
                            )
                        res_norm = float(np.linalg.norm(rhs_f))
                        diag_vals = np.diag(K_ff) if K_ff.size else np.array([], dtype=np.float64)
                        kff_diag_min = float(np.min(diag_vals)) if diag_vals.size else 0.0
                        kff_diag_max = float(np.max(diag_vals)) if diag_vals.size else 0.0
                        if options.log_full_condition_number and K_ff.size:
                            try:
                                kff_cond = float(np.linalg.cond(K_ff))
                            except np.linalg.LinAlgError:
                                kff_cond = float("inf")
                        else:
                            if not K_ff.size:
                                kff_cond = 0.0
                            else:
                                d = np.abs(np.diag(K_ff))
                                dpos = d[d > 0]
                                kff_cond = float(np.max(d) / max(float(np.min(dpos)), 1e-30)) if dpos.size else float("inf")
                        if rhs_ref0 is None:
                            f_ext_norm_step = float(np.linalg.norm(F_ext))
                            rhs_ref0 = max(res_norm, f_ext_norm_step, 1e-30)

                        ref = max(1.0, float(np.linalg.norm(F_ext)))
                        tol_mixed = max(float(options.tol_res), float(options.tol_res_rel) * ref)
                        if options.tol_res_rel_rhs is not None:
                            tol_mixed = max(tol_mixed, float(options.tol_res_rel_rhs) * rhs_ref0)

                        psi_max = max([abs(n.psi) for n in nodes], default=0.0)
                        if res_norm <= tol_mixed * _RES_TOL_SLACK:
                            step_conv = True
                            du_norm = 0.0
                            iteration_history.append(
                                {
                                    "iter": it_total,
                                    "residual_norm": res_norm,
                                    "displacement_norm": du_norm,
                                    "warping_amplitude_max": float(psi_max),
                                    "f_ext_norm": f_ext_norm,
                                    "g_norm": g_norm,
                                    "kff_cond": kff_cond,
                                    "kff_diag_min": kff_diag_min,
                                    "kff_diag_max": kff_diag_max,
                                }
                            )
                            logger.info(
                                "newton_iter %s",
                                json.dumps(
                                    {
                                        "iter": it_total,
                                        "residual_norm": res_norm,
                                        "displacement_norm": du_norm,
                                        "warping_amplitude_max": float(psi_max),
                                        "f_ext_norm": f_ext_norm,
                                        "g_norm": g_norm,
                                        "kff_cond": kff_cond,
                                        "kff_diag_min": kff_diag_min,
                                        "kff_diag_max": kff_diag_max,
                                    }
                                ),
                            )
                            if options.verbose:
                                print(
                                    f"  load lam={lam:.3f} converged in {it} NR it, |res|={res_norm:.3e}"
                                )
                            break

                        if options.adaptive_tangent_rho and it > 1 and res_norm >= last_res * 0.9995:
                            bad_streak += 1
                        else:
                            bad_streak = 0
                        if options.adaptive_tangent_rho and bad_streak >= 4:
                            rho_ls = min(
                                rho_ls * float(options.adaptive_rho_growth),
                                float(options.adaptive_rho_max),
                            )
                            bad_streak = 0
                        last_res = res_norm

                        nf = K_ff.shape[0]
                        K_eff = K_ff
                        if rho_ls > 0.0:
                            K_eff = K_ff + rho_ls * np.eye(nf, dtype=np.float64)
                        try:
                            du_f, *_ = np.linalg.lstsq(K_eff, rhs_f, rcond=None)
                        except np.linalg.LinAlgError:
                            # lstsq SVD can fail on ill‑conditioned tangents; damp and retry lstsq (avoid pinv/SVD stalls).
                            eps = 1e-8 * max(float(kff_diag_max), 1.0)
                            Kd = K_eff + eps * np.eye(nf, dtype=np.float64)
                            try:
                                du_f = np.linalg.solve(Kd, rhs_f)
                            except np.linalg.LinAlgError:
                                du_f, *_ = np.linalg.lstsq(Kd, rhs_f, rcond=1e-10)
                        du_f *= float(np.clip(options.relax_factor, 0.0, 1.0))
                        du_norm = float(np.linalg.norm(du_f))
                        iteration_history.append(
                            {
                                "iter": it_total,
                                "residual_norm": res_norm,
                                "displacement_norm": du_norm,
                                "warping_amplitude_max": float(psi_max),
                                "f_ext_norm": f_ext_norm,
                                "g_norm": g_norm,
                                "kff_cond": kff_cond,
                                "kff_diag_min": kff_diag_min,
                                "kff_diag_max": kff_diag_max,
                            }
                        )
                        logger.info(
                            "newton_iter %s",
                            json.dumps(
                                {
                                    "iter": it_total,
                                    "residual_norm": res_norm,
                                    "displacement_norm": du_norm,
                                    "warping_amplitude_max": float(psi_max),
                                    "f_ext_norm": f_ext_norm,
                                    "g_norm": g_norm,
                                    "kff_cond": kff_cond,
                                    "kff_diag_min": kff_diag_min,
                                    "kff_diag_max": kff_diag_max,
                                }
                            ),
                        )
                        if options.verbose and abs(lam - 1.0) < 1e-6:
                            print(
                                f"  NR iter {it:3d}  |res|={res_norm:.3e}  |du|={du_norm:.3e}"
                            )

                        du = expand_solution(free_idx, du_f, ndof)
                        if options.line_search:
                            snap = [n.copy() for n in nodes]
                            merit0 = res_norm
                            s_try = 1.0
                            shrink = float(options.line_search_shrink)
                            s_min = float(options.line_search_min_scale)
                            n_trial = max(1, int(options.line_search_max_trials))
                            best_s = s_min
                            best_merit = float("inf")
                            for _trial in range(n_trial):
                                _restore_nodes(nodes, snap)
                                _apply_increment(nodes, du * s_try)
                                F_trial = external_load_vector(
                                    model, loads, options.n_gauss, nodes=nodes
                                ) * float(lam)
                                res_t = _reduced_equilibrium_residual_norm(
                                    model, nodes, stations, F_trial, free_idx, options.n_gauss,
                                    options.fd_eps, K7_gp, use_cs=options.use_cs_gradient,
                                )
                                if res_t < best_merit:
                                    best_merit = res_t
                                    best_s = s_try
                                if res_t <= merit0 * 0.995:
                                    best_s = s_try
                                    best_merit = res_t
                                    break
                                s_try = max(s_min, s_try * shrink)
                            _restore_nodes(nodes, snap)
                            _apply_increment(nodes, du * best_s)
                            du_norm = float(np.linalg.norm(du_f * best_s))
                        else:
                            _apply_increment(nodes, du)

                        if du_norm < options.tol_du and res_norm <= tol_mixed * _RES_TOL_SLACK:
                            step_conv = True
                            break

                        if options.accept_stagnation:
                            if abs(res_norm - prev_res) < 0.05 * max(prev_res, 1.0) and du_norm < max(
                                options.tol_du, 1e-9
                            ):
                                stall += 1
                            else:
                                stall = 0
                            prev_res = res_norm
                            if stall >= options.stagnation_window:
                                rho_ls = min(
                                    rho_ls + 1e-4 * max(1.0, kff_diag_max),
                                    float(options.adaptive_rho_max),
                                )
                                stall = 0
                                if options.verbose:
                                    print(f"  NR stagnated: rho_ls bumped to {rho_ls:.2e}")

                    if step_conv:
                        prev = lam
                        sub_ok = True
                        break
                    _restore_nodes(nodes, snap_equilibrium)
                    step *= 0.5
                    bis += 1
                if not sub_ok:
                    converged = False
                    if options.verbose:
                        print(
                            f"  could not reach lam toward {target:.3f} (sub-step < {min_step} or max bisect)."
                        )
                    break
            if not converged:
                break

    z_out, es, rs = collect_station_data(model, nodes, stations, options.n_gauss, options.fd_eps)
    z_nodal, es_nodal, rs_nodal = project_beam_strains_resultants_to_nodes(
        model, nodes, stations, options.n_gauss, options.fd_eps
    )
    reacts = compute_reactions(model, nodes, loads, bcs_all, stations, options.n_gauss, options.fd_eps)

    n_n = model.n_nodes
    pos = np.stack([n.x for n in nodes], axis=0)
    rotvec = np.stack([n.spin_accum for n in nodes], axis=0)
    Rm = np.stack([quat_to_rotmat(n.q) for n in nodes], axis=0)
    psi = np.array([n.psi for n in nodes], dtype=np.float64)

    # J.3: global buckling eigenvalue extraction at converged state
    tangent_K: np.ndarray | None = None
    geometric_K: np.ndarray | None = None
    buckling_lambdas: np.ndarray | None = None
    buckling_modes: np.ndarray | None = None

    if converged and options.extract_buckling:
        try:
            tangent_K, geometric_K, buckling_lambdas, buckling_modes = _extract_buckling(
                model, nodes, stations, fixed, options, n_n, K7_gp, free_idx, k_t_buck_converged
            )
        except Exception as exc:
            logger.warning("buckling_extraction_failed: %s", exc)

    return BeamSolveResult(
        nodal_positions=pos,
        nodal_rotations=rotvec,
        nodal_R=Rm,
        nodal_warping=psi,
        resultants=rs,
        strains=es,
        converged=converged,
        n_iterations=it_total,
        residual_norm=res_norm,
        iteration_history=iteration_history,
        nodes=nodes,
        reactions=reacts,
        z_stations_out=z_out,
        z_nodal_out=z_nodal,
        strains_nodal=es_nodal,
        resultants_nodal=rs_nodal,
        tangent_stiffness=tangent_K,
        geometric_stiffness=geometric_K,
        global_buckling_lambdas=buckling_lambdas,
        global_buckling_modeshapes=buckling_modes,
    )


def _extract_buckling(
    model: "BeamModel",
    nodes: list,
    stations: list,
    fixed: set,
    options: "SolverOptions",
    n_nodes: int,
    K7_gp=None,
    free_idx_pre: np.ndarray | None = None,
    k_t_ff_converged: np.ndarray | None = None,
) -> tuple:
    """Extract ``K_t``, assembled stress (initial) geometric stiffness ``K_g``, and eigenpair (J.3).

    ``K_t`` is the converged path tangent: analytic ``B^T K7 B + r_m ∂²e_m/∂q∂q^T`` when
    ``full_fd_hessian`` is False, or full FD element Hessian when True.  ``K_g`` is
    the stress part only, from :func:`assemble_geometric_stiffness`.  The generalized
    problem ``K_t φ = μ K_g φ`` is solved in the free subspace.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    ndof = 7 * n_nodes
    if free_idx_pre is not None:
        free_idx = np.asarray(free_idx_pre, dtype=np.int64).ravel()
    else:
        free_idx = np.array([i for i in range(ndof) if i not in fixed], dtype=np.int64)

    if k_t_ff_converged is not None:
        K_t_ff = k_t_ff_converged
    else:
        K_t_full = assemble_hessian(
            model, nodes, stations,
            options.n_gauss, options.fd_eps, options.hess_eps,
            options.full_fd_hessian, options.fd_eps,
            options.spin_stabilization, options.warping_stabilization,
            K7_gp,
        )
        K_t_ff = K_t_full[np.ix_(free_idx, free_idx)]

    K_g_full = assemble_geometric_stiffness(
        model, nodes, stations, options.n_gauss, options.fd_eps,
        options.spin_stabilization, options.warping_stabilization, K7_gp,
    )
    K_g_ff = K_g_full[np.ix_(free_idx, free_idx)]

    n_free = K_t_ff.shape[0]
    n_modes = min(int(options.n_buckling_modes), max(1, n_free - 2))

    # scipy eigsh: solve K_t @ phi = lambda * K_g @ phi
    # Smallest positive eigenvalues via shift-invert mode.
    try:
        K_t_sp = csr_matrix(K_t_ff)
        K_g_sp = csr_matrix(K_g_ff)
        lambdas_raw, vecs_raw = eigsh(
            K_t_sp, k=n_modes, M=K_g_sp, which="LM", sigma=1.0,
            tol=1e-8, maxiter=500,
        )
        # Sort by ascending eigenvalue (smallest positive = most critical)
        order = np.argsort(lambdas_raw.real)
        lambdas = lambdas_raw.real[order]
        vecs = vecs_raw.real[:, order]
    except Exception:
        # Fallback: direct dense solve when sparse solver fails
        try:
            lambdas_arr, vecs_arr = np.linalg.eig(np.linalg.solve(K_g_ff, K_t_ff))
            real_pos_mask = (lambdas_arr.real > 0) & (np.abs(lambdas_arr.imag) < 1e-3 * np.abs(lambdas_arr.real + 1e-30))
            if real_pos_mask.any():
                idx_sorted = np.argsort(lambdas_arr.real[real_pos_mask])[:n_modes]
                lambdas = lambdas_arr.real[real_pos_mask][idx_sorted]
                vecs = vecs_arr.real[:, real_pos_mask][:, idx_sorted]
            else:
                lambdas = np.full(n_modes, np.inf)
                vecs = np.zeros((n_free, n_modes))
        except Exception:
            lambdas = np.full(n_modes, np.inf)
            vecs = np.zeros((n_free, n_modes))

    # Expand modeshapes to (n_nodes, 7) format (J.5)
    modeshapes = np.zeros((n_modes, n_nodes, 7), dtype=np.float64)
    for mi in range(min(n_modes, vecs.shape[1])):
        phi_full = np.zeros(ndof, dtype=np.float64)
        phi_full[free_idx] = vecs[:, mi]
        modeshapes[mi] = phi_full.reshape(n_nodes, 7)

    return K_t_ff, K_g_ff, lambdas[:n_modes], modeshapes
