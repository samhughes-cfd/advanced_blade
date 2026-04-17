"""Mechanical strains vanish when state matches precurvature reference."""

from __future__ import annotations

import numpy as np

import beam_model as bm
from beam_model.engine.element import e7_vector
from beam_model.engine.interp import stations_from_arrays
from beam_model.engine.solver import _initialize_nodes


def test_zero_mechanical_strain_with_kappa0_match() -> None:
    n = 4
    L = 2.0
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
        K6[k, 3, 3] = 1e5
        K6[k, 4, 4] = K6[k, 5, 5] = 1e5
    st = stations_from_arrays(z, K6)
    kappa0 = np.zeros((n, 3), dtype=np.float64)
    model = bm.BeamModel(
        X_ref=X,
        elements=elems,
        section_stations=st,
        span_axis=2,
        z_node=z,
        kappa0_node=kappa0,
        chi0_node=np.zeros(n, dtype=np.float64),
    )
    nodes = _initialize_nodes(model)
    e = e7_vector(model, elems[0], nodes, 0.0, 1e-7)
    assert np.max(np.abs(e[:6])) < 1e-9
    assert abs(e[6]) < 1e-9
