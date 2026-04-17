"""
beam_model/solver.py
====================
Static Newton–Raphson driver (7 DOFs per node).
"""

from __future__ import annotations

import json
import logging
from typing import List

import numpy as np

from .assembly import (
    assemble_gradient,
    assemble_hessian,
    expand_solution,
    external_load_vector,
    fixed_dof_set,
    reduce_linear_system,
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

logger = logging.getLogger(__name__)


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
        nodes[i].psi += dpsi
        nodes[i].spin_accum = nodes[i].spin_accum + dth


def _bcs_from_loads(loads: BeamLoads, explicit: List[BoundaryCondition] | None) -> List[BoundaryCondition]:
    out: List[BoundaryCondition] = []
    if loads.bcs:
        out.extend(loads.bcs)
    if explicit:
        out.extend(explicit)
    return out


def solve_static(
    model: BeamModel,
    loads: BeamLoads,
    options: SolverOptions | None = None,
    *,
    bcs: List[BoundaryCondition] | None = None,
) -> BeamSolveResult:
    """
    Nonlinear static equilibrium with optional BCs in ``loads.bcs`` or ``bcs``.

    Arc-length: when ``options.use_arc_length`` is True, load stepping still
    applies; full arc-length continuation is not implemented in this version.
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
    F_ext_full = external_load_vector(model, loads, options.n_gauss)

    iteration_history: List[dict] = []
    converged = False
    res_norm = 1e30
    du_norm = 1e30
    it_total = 0

    n_steps = max(1, int(options.n_load_steps))
    lam_values = [k / float(n_steps) for k in range(1, n_steps + 1)]

    for lam in lam_values:
        F_ext = F_ext_full * lam
        step_conv = False
        prev_res = 1e300
        stall = 0
        for it in range(1, options.max_iter + 1):
            it_total += 1
            g = assemble_gradient(model, nodes, stations, options.n_gauss, options.fd_eps)
            rhs = F_ext - g
            K_ff, rhs_f, free_idx = reduce_linear_system(
                assemble_hessian(
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
                ),
                rhs,
                fixed,
            )
            res_norm = float(np.linalg.norm(rhs_f))
            ref = max(1.0, float(np.linalg.norm(F_ext)))
            tol_mixed = max(options.tol_res, options.tol_res_rel * ref)
            psi_max = max([abs(n.psi) for n in nodes], default=0.0)
            if res_norm < tol_mixed:
                step_conv = True
                du_norm = 0.0
                iteration_history.append(
                    {
                        "iter": it_total,
                        "residual_norm": res_norm,
                        "displacement_norm": du_norm,
                        "warping_amplitude_max": float(psi_max),
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
                        }
                    ),
                )
                if options.verbose:
                    print(f"  load lam={lam:.3f} converged in {it} NR it, |res|={res_norm:.3e}")
                break

            nf = K_ff.shape[0]
            rho = float(options.tangent_rho)
            if rho > 0.0:
                K_ff = K_ff + rho * np.eye(nf, dtype=np.float64)
            du_f, *_ = np.linalg.lstsq(K_ff, rhs_f, rcond=None)
            du_f *= float(np.clip(options.relax_factor, 0.0, 1.0))
            du_norm = float(np.linalg.norm(du_f))
            iteration_history.append(
                {
                    "iter": it_total,
                    "residual_norm": res_norm,
                    "displacement_norm": du_norm,
                    "warping_amplitude_max": float(psi_max),
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
                    }
                ),
            )
            if options.verbose and lam == lam_values[-1]:
                print(f"  NR iter {it:3d}  |res|={res_norm:.3e}  |du|={du_norm:.3e}")

            du = expand_solution(free_idx, du_f, ndof)
            _apply_increment(nodes, du)

            if du_norm < options.tol_du and res_norm < tol_mixed:
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
                cap = 0.35 * max(1.0, float(np.linalg.norm(F_ext)))
                if stall >= options.stagnation_window and res_norm < cap:
                    step_conv = True
                    if options.verbose:
                        print(
                            f"  load lam={lam:.3f} stagnation stop "
                            f"(|res|={res_norm:.3e}, |du|={du_norm:.3e})"
                        )
                    break

        converged = step_conv
        if not step_conv:
            if options.verbose:
                print(f"  load lam={lam:.3f} failed to converge in {options.max_iter} NR iterations.")
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
    )
