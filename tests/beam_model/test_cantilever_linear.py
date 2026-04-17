"""Small tip load: tip deflection vs Euler–Bernoulli (same-order accuracy)."""

from __future__ import annotations

import numpy as np

import beam_model as bm
from beam_model.engine.constitutive import resultants_to_recovery6
from beam_model.engine.interp import stations_from_arrays


def euler_tip(F: float, L: float, EI: float) -> float:
    return F * L**3 / (3.0 * EI)


def test_cantilever_tip_vs_euler() -> None:
    L = 4.0
    n = 25
    z = np.linspace(0.0, L, n)
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 2] = z
    elems = []
    for e in range(n - 1):
        i, j = e, e + 1
        L0 = float(np.linalg.norm(X[j] - X[i]))
        elems.append(bm.BeamElement((i, j), L0, 0.5 * (z[i] + z[j])))
    EI = 2.0e6
    EA = 5.0e9
    GJ = 4.0e5
    K6 = np.zeros((n, 6, 6))
    for k in range(n):
        K6[k, 0, 0] = EA
        K6[k, 1, 1] = EI
        K6[k, 2, 2] = EI
        K6[k, 3, 3] = GJ
        K6[k, 4, 4] = K6[k, 5, 5] = 5e5
    K7 = np.zeros((n, 7, 7))
    for k in range(n):
        K7[k, :6, :6] = K6[k]
        K7[k, 6, 6] = 1e3
    st = stations_from_arrays(z, K6, K7)
    model = bm.BeamModel(X_ref=X, elements=elems, section_stations=st, span_axis=2, z_node=z)
    F = 50.0
    loads = bm.BeamLoads(
        nodal_F=np.zeros((n, 3)),
        nodal_M=np.zeros((n, 3)),
        bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
    )
    loads.nodal_F[-1, 0] = F
    opts = bm.SolverOptions(
        max_iter=50,
        tol_res=1e-6,
        tol_du=1e-9,
        n_load_steps=4,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res = bm.solve_static(model, loads, opts)
    assert res.converged
    tip_x = float(res.nodal_positions[-1, 0] - X[-1, 0])
    u_eb = euler_tip(F, L, EI)
    rel = abs(tip_x - u_eb) / max(abs(u_eb), 1e-12)
    assert rel < 0.85
    r6 = resultants_to_recovery6(res.resultants[-1])
    assert r6.shape == (6,)
