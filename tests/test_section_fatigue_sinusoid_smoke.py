"""Smoke: short sinusoid + smoke RecoveryCache + fatigue run produces finite damage."""

from __future__ import annotations

import numpy as np

from blade_analysis.fatigue_damage import FatigueAnalysis
from blade_analysis.fatigue_damage._smoke_fixtures import (
    build_smoke_recovery_cache,
    default_fatigue_sn_curves,
    smoke_sinusoidal_resultant_history,
)
from blade_analysis.fatigue_damage.engine.conversion import resultants_to_stress_history
from numpy.testing import assert_array_equal


def test_smoke_sinusoid_fatigue_finishes() -> None:
    cache = build_smoke_recovery_cache()
    z = np.asarray(cache.z_stations, dtype=np.float64)
    hist = smoke_sinusoidal_resultant_history(
        z,
        n_t=32,
        t_end=0.5,
        f_hz=2.0,
        amplitude=3.0e3,
        load_component="My",
        spanwise_envelope=True,
    )
    sh = resultants_to_stress_history(hist, cache, chunk_size=8)
    assert sh.sigma_composite.shape[0] == 32
    assert sh.sigma_composite.shape[1] == int(z.size)
    assert sh.sigma_isotropic.shape[0] == 32

    res = FatigueAnalysis.from_cache(
        cache, default_fatigue_sn_curves(), design_life_years=25.0
    ).run(hist, memory_limit_mb=64.0)

    assert np.isfinite(res.max_damage_composite)
    assert np.isfinite(res.max_damage_isotropic)
    assert res.damage_composite.shape[0] == int(z.size)
    assert res.damage_composite.shape == res.fi_static_tw.shape
    assert res.memory_mode in ("full", "incremental")
    assert res.rainflow_bins is not None
    assert res.rainflow_bins.counts_comp.shape[1] == int(z.size)


def test_sinusoidal_history_to_array_stacking() -> None:
    z = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    h = smoke_sinusoidal_resultant_history(
        z, n_t=5, t_end=1.0, f_hz=1.0, load_component="Mz", spanwise_envelope=False
    )
    a = h.to_array()
    assert a.shape == (5, 3, 7)
    assert_array_equal(a[..., 0], h.N)
    assert_array_equal(a[..., 1], h.Vy)
    assert_array_equal(a[..., 4], h.Mz)
    assert float(np.max(np.abs(h.Mz))) > 0.0
