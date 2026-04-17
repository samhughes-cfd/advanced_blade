"""Smoke: beam_model.interface.plot functions run without error."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import beam_model as bm
from beam_model.engine.blade_geometry import BladeGeometry
from beam_model.engine.interp import stations_from_arrays
from beam_model.interface import plot as bmplot


def _tiny_model_loads() -> tuple[bm.BeamModel, bm.BeamLoads]:
    L = 4.0
    z_st = np.linspace(0.0, L, 4)
    r_ref = np.zeros((z_st.shape[0], 3))
    r_ref[:, 2] = z_st
    geom = BladeGeometry(
        z_stations=z_st,
        r_ref=r_ref,
        kappa0=np.zeros((z_st.shape[0], 3)),
        tau0=np.zeros_like(z_st),
        chord=np.ones_like(z_st),
        twist=np.zeros_like(z_st),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2)),
        subcomponent_materials={},
        chi0=np.zeros_like(z_st),
    )
    n = z_st.shape[0]
    mats6 = np.zeros((n, 6, 6))
    mats7 = np.zeros((n, 7, 7))
    for i in range(n):
        mats6[i, 0, 0] = 1e9
        mats6[i, 1, 1] = 1e6
        mats6[i, 2, 2] = 1e6
        mats6[i, 3, 3] = 1e6
        mats6[i, 4, 4] = 1e5
        mats6[i, 5, 5] = 1e5
        mats7[i, :6, :6] = mats6[i]
        mats7[i, 6, 6] = 1e5
    stations = stations_from_arrays(z_st, mats6, mats7)
    model = bm.BeamModel.from_blade_geometry(geom, 5, stations, span_axis=2)
    n_nodes = model.n_nodes
    q = np.zeros((len(model.elements), 3))
    q[:, 1] = 50.0
    loads = bm.BeamLoads(
        nodal_F=np.zeros((n_nodes, 3)),
        nodal_M=np.zeros((n_nodes, 3)),
        distributed_q=q,
        bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
    )
    return model, loads


def test_beam_plot_functions_run():
    model, loads = _tiny_model_loads()
    res = bm.BeamAnalysis(model).solve_static(
        loads,
        options=bm.SolverOptions(
            max_iter=40,
            tol_res=1e-1,
            tol_res_rel=1e-2,
            n_gauss=2,
            n_load_steps=4,
            verbose=False,
        ),
    )
    nn = model.n_nodes
    assert res.z_nodal_out is not None
    assert res.z_nodal_out.shape == (nn,)
    assert res.strains_nodal is not None and res.strains_nodal.shape == (nn, 7)
    assert res.resultants_nodal is not None and res.resultants_nodal.shape == (nn, 7)
    import matplotlib.pyplot as plt

    fig, _ = bmplot.plot_centerline_ref_def(model, res)
    plt.close(fig)
    fig, _ = bmplot.plot_spanwise_resultants(res)
    plt.close(fig)
    fig, _ = bmplot.plot_spanwise_strains(res)
    plt.close(fig)
    fig, _ = bmplot.plot_spanwise_resultants_nodal(res)
    plt.close(fig)
    fig, _ = bmplot.plot_spanwise_strains_nodal(res)
    plt.close(fig)
    fig, _ = bmplot.plot_nodal_warping(model, res)
    plt.close(fig)
    fig, _ = bmplot.plot_iteration_history(res)
    plt.close(fig)
    fig, _ = bmplot.plot_reactions(res)
    plt.close(fig)
    fig, _ = bmplot.plot_distributed_loads(model, loads)
    plt.close(fig)
