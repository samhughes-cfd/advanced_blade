"""Smoke: blade_analysis.fatigue_damage.interface.plot."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from blade_analysis.fatigue_damage.core.types import FatigueResult, RainflowBins
from blade_analysis.fatigue_damage.engine.sn_curves import SNcurve
from blade_analysis.fatigue_damage.interface import plot as fplot


def test_fatigue_plots_run():
    n_bin, n_s, n_cp, n_ply, n_iso = 12, 2, 1, 2, 1
    rc = (1e4 + np.arange(n_bin, dtype=np.float64) * 4e4).reshape(n_bin, 1, 1, 1)
    ranges_comp = np.broadcast_to(rc, (n_bin, n_s, n_cp, n_ply)).copy()
    cc = np.linspace(0.0, 10.0, n_bin, dtype=np.float64).reshape(n_bin, 1, 1, 1)
    counts_comp = np.broadcast_to(cc, (n_bin, n_s, n_cp, n_ply)).copy()
    ri = (1e4 + np.arange(n_bin, dtype=np.float64) * 3e4).reshape(n_bin, 1, 1)
    ranges_iso = np.broadcast_to(ri, (n_bin, n_s, n_iso)).copy()
    ci = np.linspace(0.0, 5.0, n_bin, dtype=np.float64).reshape(n_bin, 1, 1)
    counts_iso = np.broadcast_to(ci, (n_bin, n_s, n_iso)).copy()
    bins = RainflowBins(
        ranges_comp=ranges_comp,
        means_comp=np.zeros((n_bin, n_s, n_cp, n_ply)),
        counts_comp=counts_comp,
        ranges_iso=ranges_iso,
        means_iso=np.zeros((n_bin, n_s, n_iso)),
        counts_iso=counts_iso,
    )
    fr = FatigueResult(
        damage_composite=np.zeros((n_s, n_cp, n_ply)),
        damage_isotropic=np.zeros((n_s, n_iso)),
        damage_delam=None,
        life_composite=np.full((n_s, n_cp, n_ply), 25.0),
        life_isotropic=np.full((n_s, n_iso), 25.0),
        max_damage_composite=0.0,
        max_damage_isotropic=0.0,
        worst_composite=(0, "skin", 0),
        worst_isotropic=(0, "al"),
        fatigue_critical_material="composite",
        fi_static_tw=np.zeros((n_s, n_cp, n_ply)),
        fi_static_vm=np.zeros((n_s, n_iso)),
        stress_component_used=0,
        goodman_applied=False,
        design_life_years=25.0,
        memory_mode="test",
        rainflow_bins=bins,
    )
    z = np.linspace(0.0, 1.0, n_s)
    fig, _ = fplot.plot_damage_life_vs_span(fr, z)
    plt.close(fig)
    fig, _ = fplot.plot_static_fi_vs_span(fr, z)
    plt.close(fig)
    fig, _ = fplot.plot_rainflow_composite(bins, station=0, subcomp=0, ply=0)
    plt.close(fig)
    fig, _ = fplot.plot_rainflow_isotropic(bins, station=0, subcomp=0)
    plt.close(fig)
    rc = np.asarray(bins.ranges_comp[:, 0, 0, 0], dtype=np.float64)
    fig, _ = fplot.plot_sn_curve_with_ranges(SNcurve.gfrp_blade(), rc)
    plt.close(fig)
