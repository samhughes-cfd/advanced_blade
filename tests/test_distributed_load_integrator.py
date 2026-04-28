"""Sanity checks for distributed load integration and ``.dat`` parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_precompute.global_beam_model.engine.distributed_load_integrator import DistributedLoadIntegrator
from blade_precompute.section_optimisation.io.distributed_load_dat import (
    extreme_loads_from_distributed,
    load_extreme_distributed_loads_dat,
    load_operational_distributed_loads_dat,
    resultant_history_from_operational_dat,
)


def test_constant_qy_shear_tip_zero() -> None:
    z = np.linspace(0.0, 8.0, 41, dtype=np.float64)
    q = 2500.0
    q_y = np.full_like(z, q)
    q_z = np.zeros_like(z)
    m_x = np.zeros_like(z)
    r = DistributedLoadIntegrator.integrate(z, q_y, q_z, m_x)
    assert r.Vy[-1] == pytest.approx(0.0, abs=1e-9)
    L = float(z[-1] - z[0])
    assert r.Vy[0] == pytest.approx(q * L, rel=1e-9, abs=1e-6)
    # Piecewise-linear Vy in z for constant q
    assert np.allclose(r.Vy, q * (z[-1] - z), rtol=1e-9, atol=1e-6)


def test_constant_qy_moment_curvature_sign() -> None:
    z = np.linspace(0.0, 8.0, 201, dtype=np.float64)
    q = 2500.0
    q_y = np.full_like(z, q)
    q_z = np.zeros_like(z)
    m_x = np.zeros_like(z)
    r = DistributedLoadIntegrator.integrate(z, q_y, q_z, m_x)
    L = z[-1]
    mz_analytic = -0.5 * q * (L - z) ** 2
    assert np.allclose(r.Mz, mz_analytic, rtol=2e-3, atol=1e-3)


def test_constant_qx_axial_force() -> None:
    z = np.linspace(0.0, 4.0, 9, dtype=np.float64)
    q_x = np.full_like(z, 80.0)
    q_y = np.zeros_like(z)
    q_z = np.zeros_like(z)
    m_x = np.zeros_like(z)
    r = DistributedLoadIntegrator.integrate(z, q_y, q_z, m_x, q_x=q_x)
    L = float(z[-1] - z[0])
    assert r.N[-1] == pytest.approx(0.0, abs=1e-9)
    assert r.N[0] == pytest.approx(80.0 * L, rel=1e-9, abs=1e-6)
    assert np.allclose(r.N, 80.0 * (z[-1] - z), rtol=1e-9, atol=1e-6)


def test_constant_mx_torque() -> None:
    z = np.linspace(0.0, 5.0, 51, dtype=np.float64)
    m = 400.0
    q_y = np.zeros_like(z)
    q_z = np.zeros_like(z)
    m_x = np.full_like(z, m)
    r = DistributedLoadIntegrator.integrate(z, q_y, q_z, m_x)
    L = float(z[-1] - z[0])
    assert r.T[-1] == pytest.approx(0.0, abs=1e-9)
    assert r.T[0] == pytest.approx(m * L, rel=1e-9, abs=1e-6)
    assert np.allclose(r.T, m * (z[-1] - z), rtol=1e-9, atol=1e-6)


def test_extreme_dat_round_trip(tmp_path: Path) -> None:
    z = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    lines = [
        "# test",
        "spanwise_pos q_y_Npm q_z_Npm m_x_Nmpm",
        "0.0 1000 0 0",
        "2.0 1000 0 0",
        "4.0 1000 0 0",
        "6.0 1000 0 0",
    ]
    p = tmp_path / "extreme.dat"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    zl, qy, qz, mx = load_extreme_distributed_loads_dat(p)
    el = extreme_loads_from_distributed(zl, qy, qz, mx)
    assert el.z_stations.shape == z.shape
    assert el.N.shape == z.shape
    assert np.allclose(el.N, 0.0)
    assert el.Vy[0] > 0.0 and el.Vy[-1] == pytest.approx(0.0)


def test_operational_unique_tz(tmp_path: Path) -> None:
    bad = tmp_path / "bad.dat"
    bad.write_text(
        "t_s spanwise_pos q_y_Npm q_z_Npm m_x_Nmpm\n"
        "0.0 0.0 1 0 0\n"
        "0.0 0.0 2 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_operational_distributed_loads_dat(bad)


def test_data_library_extreme_integrates() -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "data_library" / "extreme_load_distribution.dat"
    if not p.is_file():
        pytest.skip("Regenerated data_library extreme file not present.")
    z, qy, qz, mx = load_extreme_distributed_loads_dat(p)
    el = extreme_loads_from_distributed(z, qy, qz, mx)
    assert el.z_stations.shape == z.shape
    assert el.T.shape == z.shape


def test_data_library_operational_history() -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "data_library" / "operational_load_timeseries.dat"
    if not p.is_file():
        pytest.skip("Regenerated data_library operational file not present.")
    rh = resultant_history_from_operational_dat(p)
    assert rh.Vy.ndim == 2
    assert rh.Vy.shape[0] == rh.time.shape[0]
    assert rh.Vy.shape[1] == rh.z_stations.shape[0]
