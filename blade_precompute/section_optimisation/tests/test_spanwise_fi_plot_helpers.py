"""Tests for iteration_report spanwise FI helpers and optional plot smoke."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_optimisation.engine.iteration_report import (
    per_station_max_fi_hashin,
    per_station_max_fi_vm,
)


def test_per_station_max_fi_hashin_3d() -> None:
    fh = np.array(
        [
            [[0.1, 0.5], [0.2, 0.1]],
            [[1.0, 0.0], [0.3, 0.4]],
        ],
        dtype=np.float64,
    )
    out = per_station_max_fi_hashin(fh)
    np.testing.assert_allclose(out, [0.5, 1.0])


def test_per_station_max_fi_hashin_2d() -> None:
    fh = np.array([[0.2, 0.9], [0.4, 0.1]], dtype=np.float64)
    out = per_station_max_fi_hashin(fh)
    np.testing.assert_allclose(out, [0.9, 0.4])


def test_per_station_max_fi_vm_2d() -> None:
    fvm = np.array([[0.1, 0.3], [0.0, 0.5]], dtype=np.float64)
    out = per_station_max_fi_vm(fvm)
    np.testing.assert_allclose(out, [0.3, 0.5])


def test_per_station_max_fi_vm_1d() -> None:
    fvm = np.array([0.2, 0.7], dtype=np.float64)
    out = per_station_max_fi_vm(fvm)
    np.testing.assert_allclose(out, [0.2, 0.7])


def test_plot_smoke_max_fi_vs_span() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from blade_precompute.section_optimisation.core.types import DesignEvaluation, DesignVector
    from blade_precompute.section_optimisation.interface import plot as dplot

    n = 4
    z = np.linspace(0.0, 1.0, n)
    dv = DesignVector(
        t_skin=np.full(n, 0.01),
        t_cap=np.full(n, 0.02),
        t_web=np.full(n, 0.015),
    )
    fh = np.random.RandomState(0).rand(n, 2, 3).astype(np.float64) * 0.5
    ev = DesignEvaluation(
        dv=dv,
        mass=1.0,
        stiffness_metric=1e6,
        resultants=np.zeros((n, 7)),
        fi_hashin=fh,
        fi_vm=np.zeros((n, 0)),
        max_fi_hashin=float(np.max(fh)),
        max_fi_vm=0.0,
    )
    fig, _ = dplot.plot_max_fi_vs_span(z, ev, None, problem=None)
    fig.canvas.draw()
    matplotlib.pyplot.close(fig)
