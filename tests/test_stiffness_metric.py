"""Integrated K7 trace helper."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_optimisation.engine.stiffness_metric import integrated_k7_trace


def test_integrated_k7_trace_trapezoid_matches_hand() -> None:
    z = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    # identity * i at each station -> trace = 7 * scale
    K = np.stack([np.eye(7, dtype=np.float64) * (i + 1) for i in range(3)], axis=0)
    tr = np.array([7.0, 14.0, 21.0])
    dz = np.diff(z)
    expect = float(np.sum(0.5 * (tr[:-1] + tr[1:]) * dz))
    got = integrated_k7_trace(K, z)
    np.testing.assert_allclose(got, expect, rtol=0, atol=1e-12)


def test_integrated_k7_trace_single_station() -> None:
    z = np.array([2.0])
    K = np.eye(7, dtype=np.float64)[None, :, :] * 3.0
    assert integrated_k7_trace(K, z) == 21.0
