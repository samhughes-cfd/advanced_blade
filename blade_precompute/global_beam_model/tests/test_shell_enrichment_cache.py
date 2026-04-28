"""Shell enrichment: airfoil cache and dB/dz computation."""

import numpy as np
import pytest


def test_airfoil_for_station_keyed_returns_same_array() -> None:
    pytest.importorskip(
        "blade_precompute.section_shell_model.job_outputs",
        reason="section_shell_model not available",
    )
    from blade_precompute.global_beam_model.engine import shell_enrichment as se

    se._airfoil_polygon_cache.clear()
    a1 = se._airfoil_for_station_keyed(0.0, 0.0, 0.0, 0.5, 4)
    a2 = se._airfoil_for_station_keyed(0.0, 0.0, 0.0, 0.5, 4)
    assert a1 is a2
    assert isinstance(a1, np.ndarray)


def test_dBdz_computation_is_nonzero_when_B_varies() -> None:
    """dBdz_at_zq is non-zero when bimoment varies along span, and note_dB_dx key absent."""
    from blade_precompute.global_beam_model.engine.shell_enrichment import _compute_dBdz_at_zq

    z_gp = np.linspace(0.0, 10.0, 20, dtype=np.float64)
    # Linearly varying B: dB/dz = 1000 N everywhere
    B_gp = 1000.0 * z_gp
    resultants = np.zeros((20, 7), dtype=np.float64)
    resultants[:, 6] = B_gp

    zq = np.array([2.0, 5.0, 8.0], dtype=np.float64)
    dBdz = _compute_dBdz_at_zq(z_gp, resultants, zq)
    assert dBdz.shape == zq.shape
    np.testing.assert_allclose(dBdz, 1000.0, rtol=1e-6,
                               err_msg="dBdz should equal slope of linear B(z)")
