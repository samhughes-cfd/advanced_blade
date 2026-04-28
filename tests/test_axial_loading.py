"""Unit tests for centrifugal + gravity axial line load and N(z) integration."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.global_beam_model.engine.axial_loading import (
    AxialLoadingConfig,
    axial_force_distribution,
    q_x_distributed,
)
from blade_precompute.global_beam_model.engine.distributed_load_integrator import (
    DistributedLoadIntegrator,
)


def test_qx_gravity_only_uniform_mu() -> None:
    z = np.linspace(0.0, 10.0, 11, dtype=np.float64)
    n = z.size
    mu = np.full(n, 2.0, dtype=np.float64)
    r = np.linspace(1.0, 11.0, n, dtype=np.float64)
    g = 9.81
    for az, sign in [(0.0, 1.0), (90.0, 0.0), (180.0, -1.0)]:
        cfg = AxialLoadingConfig(
            u_inf_m_s=0.0,
            tip_speed_ratio=0.0,
            r_tip_m=float(r[-1]),
            gravity_m_s2=g,
            azimuth_deg=az,
            enabled=True,
        )
        qx = q_x_distributed(z, r, mu, cfg)
        expect = mu * (g * sign) if abs(az - 90.0) > 1e-9 else 0.0
        if abs(az - 90.0) < 1e-9:
            assert np.allclose(qx, 0.0, atol=1e-9)
        else:
            assert np.allclose(qx, expect, rtol=1e-9)


def test_n_constant_qx_is_linear_ramp() -> None:
    z = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    q0 = 100.0
    qx = np.full_like(z, q0)
    N = axial_force_distribution(z, qx)
    assert N[-1] == pytest.approx(0.0, abs=1e-9)
    L = float(z[-1] - z[0])
    assert N[0] == pytest.approx(q0 * L, rel=1e-6)
    # piecewise: at z=4, remaining length 4
    assert N[2] == pytest.approx(q0 * 4.0, rel=1e-6)


def test_integrator_with_qx_matches_axial_force_distribution() -> None:
    z = np.linspace(0.0, 5.0, 20, dtype=np.float64)
    q_y = np.zeros_like(z)
    q_z = np.zeros_like(z)
    m_x = np.zeros_like(z)
    qx = 3.0 * (1.0 + 0.1 * z)
    r1 = DistributedLoadIntegrator.integrate(z, q_y, q_z, m_x, q_x=qx)
    r2 = axial_force_distribution(z, qx)
    assert np.allclose(r1.N, r2, rtol=1e-9, atol=1e-8)


def test_rotation_closed_form_uniform_mu_r_squared() -> None:
    """Uniform mu, omega>0, no gravity: N(r0) = 0.5*mu*omega^2*(R_tip^2 - r0^2) for constant q_x along r when r=z (span=radial)."""
    r_root = 2.0
    r_tip = 10.0
    z = np.linspace(r_root, r_tip, 50, dtype=np.float64)
    mu = 1.5
    mu_z = np.full_like(z, mu)
    cfg = AxialLoadingConfig(
        u_inf_m_s=8.0,
        tip_speed_ratio=6.0,
        r_tip_m=r_tip,
        gravity_m_s2=0.0,
        azimuth_deg=90.0,
        enabled=True,
    )
    w = float(cfg.omega_rad_s())
    qx = q_x_distributed(z, z, mu_z, cfg)
    N = axial_force_distribution(z, qx)
    for i, r0 in enumerate(z):
        N_an = 0.5 * mu * w * w * (r_tip * r_tip - float(r0) * float(r0))
        assert N[i] == pytest.approx(N_an, rel=1e-2, abs=0.01)
