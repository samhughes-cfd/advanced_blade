"""Tier 3 produces bounded output and zero at n_s < 2."""
import numpy as np

from section_model.engine.interlaminar_recovery import interlaminar_stress_recovery


def test_interlaminar_zero_when_single_station():
    sig = np.zeros((1, 1, 2, 3))
    z = np.array([0.0])
    zp = np.array([[0.0, 1e-4]])
    out = interlaminar_stress_recovery(sig, z, zp)
    assert out.shape == (1, 1, 3, 3)
    np.testing.assert_allclose(out, 0.0)


def test_interlaminar_two_stations():
    n_s, n_c, n_p = 3, 1, 2
    sig = np.random.default_rng(0).standard_normal((n_s, n_c, n_p, 3)) * 1e5
    z = np.linspace(0.0, 2.0, n_s)
    zp = np.array([[0.0, 1e-4]])
    out = interlaminar_stress_recovery(sig, z, zp)
    assert out.shape == (n_s, n_c, n_p + 1, 3)
    assert np.all(np.isfinite(out))
