"""Plane-stress von Mises FI vs closed form."""
import numpy as np

from section_model.engine.isotropic_recovery import isotropic_membrane_stresses, von_mises_plane_stress


def test_von_mises_fi():
    N = np.array([[[3e6, 4e6, 2e6]]])  # (1,1,3)
    t = np.array([0.01])
    sig = isotropic_membrane_stresses(N, t)
    svm = np.sqrt(sig[..., 0] ** 2 - sig[..., 0] * sig[..., 1] + sig[..., 1] ** 2 + 3.0 * sig[..., 2] ** 2)
    allow = np.array([300e6])
    fi = von_mises_plane_stress(sig, allow)
    np.testing.assert_allclose(fi, svm / allow, rtol=1e-10)
