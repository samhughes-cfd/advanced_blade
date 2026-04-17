"""Root warping restraint under torque: finite bimoment column, bounded warping."""

from __future__ import annotations

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays


def test_warping_root_fixed_under_torque() -> None:
    L = 2.0
    n = 9
    z = np.linspace(0.0, L, n)
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 2] = z
    elems = []
    for e in range(n - 1):
        i, j = e, e + 1
        L0 = float(np.linalg.norm(X[j] - X[i]))
        elems.append(bm.BeamElement((i, j), L0, 0.5 * (z[i] + z[j])))
    K6 = np.zeros((n, 6, 6))
    for k in range(n):
        K6[k, 0, 0] = 1e9
        K6[k, 1, 1] = K6[k, 2, 2] = 1e6
        K6[k, 3, 3] = 8e5
        K6[k, 4, 4] = K6[k, 5, 5] = 2e5
    kw = np.zeros(6)
    kw[3] = 1.0e4
    Kww = 5.0e5
    K7 = np.zeros((n, 7, 7))
    for k in range(n):
        K7[k, :6, :6] = K6[k]
        K7[k, :6, 6] = kw
        K7[k, 6, :6] = kw
        K7[k, 6, 6] = Kww
    st = stations_from_arrays(z, K6, K7)
    model = bm.BeamModel(X_ref=X, elements=elems, section_stations=st, span_axis=2, z_node=z)
    T_tip = 800.0
    loads = bm.BeamLoads(
        nodal_F=np.zeros((n, 3)),
        nodal_M=np.zeros((n, 3)),
        distributed_mz=np.zeros(len(elems)),
        bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
    )
    loads.nodal_M[-1, 2] = T_tip
    opts = bm.SolverOptions(
        max_iter=60,
        tol_res=1e-4,
        n_load_steps=6,
        spin_stabilization=1e-4,
        warping_stabilization=1e-2,
        verbose=False,
    )
    res = bm.solve_static(model, loads, opts)
    assert res.converged
    assert abs(res.nodal_warping[0]) < 1e-8
    B = res.resultants[:, 6]
    assert np.all(np.isfinite(B))
    assert float(np.max(np.abs(B))) < 1e7
