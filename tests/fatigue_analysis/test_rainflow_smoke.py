"""Rainflow binning produces nonzero cycle mass on a multiaxial-like signal."""

from __future__ import annotations

import numpy as np

from blade_analysis.fatigue_damage.core.loads import StressHistory
from blade_analysis.fatigue_damage.engine.rainflow import count_cycles_ply_stresses


def test_count_cycles_ply_nonzero_on_oscillatory_history():
    n_t, n_s, n_cp, n_ply = 400, 1, 1, 1
    t = np.linspace(0.0, 40.0, n_t, dtype=np.float64)
    s11 = 50e6 * np.sin(t) + 30e6 * np.sin(2.7 * t + 0.3)
    s22 = 20e6 * np.cos(1.1 * t)
    tau = 10e6 * np.sin(0.8 * t)
    sig = np.stack(
        [
            s11.reshape(n_t, n_s, n_cp, n_ply),
            s22.reshape(n_t, n_s, n_cp, n_ply),
            tau.reshape(n_t, n_s, n_cp, n_ply),
        ],
        axis=-1,
    )
    sh = StressHistory(
        z_stations=np.array([0.0], dtype=np.float64),
        time=t,
        sigma_composite=sig,
        sigma_isotropic=np.zeros((n_t, n_s, 1, 3), dtype=np.float64),
        composite_subcomp_names=["c0"],
        isotropic_subcomp_names=["i0"],
    )
    bins = count_cycles_ply_stresses(sh, component=0, n_range_bins=64)
    assert float(np.sum(bins.counts_comp)) > 0.0
