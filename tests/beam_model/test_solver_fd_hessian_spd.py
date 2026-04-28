"""Regression: SPD projection for full FD Hessian path."""

from __future__ import annotations

import numpy as np

from blade_precompute.global_beam_model.engine.solver import _symmetric_spd_floor


def test_symmetric_spd_floor_clamps_negative_eigenvalues() -> None:
    K = np.array([[2.0, -1.0], [-1.0, 0.05]], dtype=np.float64)
    K2 = _symmetric_spd_floor(K, eig_floor_rel=1e-6)
    assert np.allclose(K2, K2.T)
    w = np.linalg.eigvalsh(K2)
    assert float(w.min()) > 0.0
    assert float(w.min()) >= 1e-6 * max(float(w.max()), 1.0) * (1.0 - 1e-12)
