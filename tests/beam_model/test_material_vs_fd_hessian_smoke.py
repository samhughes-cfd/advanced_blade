"""
Smoke: material-only vs full finite-difference Hessian on a very small mesh.

The FD Hessian path is more expensive and can be noisier than the material tangent on tiny
problems; we only assert both runs complete with finite state and meet a loose force residual.
"""

from __future__ import annotations

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays


def _make_cantilever(*, n_nodes: int) -> tuple[bm.BeamModel, bm.BeamLoads, np.ndarray]:
    n = n_nodes
    L = 2.0
    z = np.linspace(0.0, L, n)
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 2] = z
    elems: list[bm.BeamElement] = []
    for e in range(n - 1):
        L0 = float(np.linalg.norm(X[e + 1] - X[e]))
        elems.append(bm.BeamElement((e, e + 1), L0, 0.5 * (z[e] + z[e + 1])))
    EI, EA, GJ = 2.0e6, 5.0e9, 4.0e5
    K6 = np.zeros((n, 6, 6), dtype=np.float64)
    for k in range(n):
        K6[k, 0, 0] = EA
        K6[k, 1, 1] = EI
        K6[k, 2, 2] = EI
        K6[k, 3, 3] = GJ
        K6[k, 4, 4] = K6[k, 5, 5] = 5e5
    K7 = np.zeros((n, 7, 7), dtype=np.float64)
    for k in range(n):
        K7[k, :6, :6] = K6[k]
        K7[k, 6, 6] = 1e3
    st = stations_from_arrays(z, K6, K7)
    model = bm.BeamModel(X_ref=X, elements=elems, section_stations=st, span_axis=2, z_node=z)
    loads = bm.BeamLoads(
        nodal_F=np.zeros((n, 3)),
        nodal_M=np.zeros((n, 3)),
        bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
    )
    loads.nodal_F[-1, 0] = 10.0
    return model, loads, X


def test_small_cantilever_material_and_full_fd_both_sane() -> None:
    model, loads, X = _make_cantilever(n_nodes=5)
    base = bm.SolverOptions(
        max_iter=60,
        tol_res=1e-3,
        tol_du=1e-6,
        n_load_steps=2,
        accept_stagnation=False,
        full_fd_hessian=False,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        hess_eps=1e-5,
        verbose=False,
    )
    res_m = bm.solve_static(model, loads, base)
    assert res_m.converged
    assert float(res_m.residual_norm) < 1e-2
    assert np.isfinite(res_m.nodal_positions).all()

    model2, loads2, _ = _make_cantilever(n_nodes=5)
    opts_fd = bm.SolverOptions(
        max_iter=60,
        tol_res=1e-3,
        tol_du=1e-6,
        n_load_steps=2,
        accept_stagnation=False,
        full_fd_hessian=True,
        project_fd_hessian_spd=True,
        fd_hessian_eig_floor_rel=1e-8,
        hess_eps=1e-5,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res_f = bm.solve_static(model2, loads2, opts_fd)
    assert res_f.converged
    assert float(res_f.residual_norm) < 1e-2
    assert np.isfinite(res_f.nodal_positions).all()
    # Same tip order of magnitude (both equilibriated to loose tol)
    d_m = float(res_m.nodal_positions[-1, 0] - X[-1, 0])
    d_f = float(res_f.nodal_positions[-1, 0] - X[-1, 0])
    if abs(d_m) > 1e-9:
        assert abs(d_f - d_m) / abs(d_m) < 0.2
