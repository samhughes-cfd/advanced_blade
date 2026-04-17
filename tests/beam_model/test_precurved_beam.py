"""Stress-free straight reference with zero precurvature has vanishing internal residual."""

from __future__ import annotations

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.assembly import assemble_gradient
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
from blade_precompute.global_beam_model.engine.solver import _initialize_nodes


def test_straight_reference_zero_load_gradient() -> None:
    n = 5
    L = 3.0
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
        K6[k, 3, 3] = 5e5
        K6[k, 4, 4] = K6[k, 5, 5] = 1e5
    st = stations_from_arrays(z, K6)
    model = bm.BeamModel(
        X_ref=X,
        elements=elems,
        section_stations=st,
        span_axis=2,
        z_node=z,
        kappa0_node=np.zeros((n, 3)),
        chi0_node=np.zeros(n),
    )
    nodes = _initialize_nodes(model)
    g = assemble_gradient(model, nodes, st, n_gauss=2, fd_h=1e-7)
    assert np.linalg.norm(g) < 1e-5
