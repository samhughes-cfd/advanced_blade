"""Smoke: data_library.plot_inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from data_library.plot_inputs import (
    plot_blade_spanwise_dat,
    plot_extreme_load_distribution_dat,
    plot_operational_load_heatmap,
    plot_operational_timeseries_dat,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_data_library_plots_run():
    root = _repo_root()
    fig, _ = plot_blade_spanwise_dat(root / "data_library" / "blade_spanwise_distribution.dat")
    plt.close(fig)
    fig, _ = plot_extreme_load_distribution_dat(root / "data_library" / "extreme_load_distribution.dat")
    plt.close(fig)
    fig, _ = plot_operational_timeseries_dat(root / "data_library" / "operational_load_timeseries.dat")
    plt.close(fig)
    fig, _ = plot_operational_load_heatmap(root / "data_library" / "operational_load_timeseries.dat")
    plt.close(fig)
