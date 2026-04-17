"""Smoke: design_optimisation.interface.plot."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from design_optimisation.core.types import DesignEvaluation, DesignVector, OptimisationResult
from design_optimisation.interface import plot as dplot


def test_design_optimisation_plots_run():
    n = 4
    z = np.linspace(0.0, 2.0, n)
    dv0 = DesignVector(
        t_skin=np.full(n, 0.01),
        t_cap=np.full(n, 0.04),
        t_web=np.full(n, 0.012),
    )
    dv1 = DesignVector(
        t_skin=np.full(n, 0.011),
        t_cap=np.full(n, 0.039),
        t_web=np.full(n, 0.013),
    )
    sh_tw = (n, 2, 3)
    sh_vm = (n, 1)
    ev0 = DesignEvaluation(
        dv=dv0,
        mass=120.0,
        stiffness_metric=1.5e9,
        resultants=np.zeros((n, 7)),
        fi_tw=np.full(sh_tw, 0.4),
        fi_vm=np.full(sh_vm, 0.35),
        fi_delam=None,
        max_fi_tw=0.4,
        max_fi_vm=0.35,
        max_fi_delam=None,
    )
    ev1 = DesignEvaluation(
        dv=dv1,
        mass=118.0,
        stiffness_metric=1.52e9,
        resultants=np.zeros((n, 7)),
        fi_tw=np.full(sh_tw, 0.42),
        fi_vm=np.full(sh_vm, 0.33),
        fi_delam=None,
        max_fi_tw=0.42,
        max_fi_vm=0.33,
        max_fi_delam=None,
    )
    opt = OptimisationResult(
        success=True,
        message="ok",
        dv_opt=dv1,
        evaluations=[ev0, ev1],
        n_iter=2,
    )
    fig, _ = dplot.plot_design_vector_vs_span(z, dv0)
    plt.close(fig)
    fig, _ = dplot.plot_design_vector_vs_span(z, dv1, dv_compare=dv0)
    plt.close(fig)
    fig, _ = dplot.plot_optimisation_history(opt)
    plt.close(fig)
