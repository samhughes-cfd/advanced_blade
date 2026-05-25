from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_precompute.section_optimisation.core.types import (
    DistributedLoadCurves,
    ExtremeLoads,
    OptimBladeGeometry,
)
from blade_precompute.section_optimisation.engine import beam_distributed
from blade_precompute.section_optimisation.engine.beam_distributed import GlobalBeamResultantDriver


def test_global_beam_driver_returns_section_recovery_order(monkeypatch) -> None:
    z = np.array([0.0, 1.0], dtype=np.float64)
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=np.zeros((2, 3), dtype=np.float64),
        kappa0=np.zeros((2, 3), dtype=np.float64),
        chord=np.ones(2, dtype=np.float64),
        twist=np.zeros(2, dtype=np.float64),
        airfoil_profiles=[],
        web_positions=np.array([-0.35, 0.0], dtype=np.float64),
        subcomponent_materials={},
    )
    curves = DistributedLoadCurves(
        loads_r_z_m=z,
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )
    loads = ExtremeLoads(
        z_stations=z,
        N=np.zeros(2, dtype=np.float64),
        Vy=np.zeros(2, dtype=np.float64),
        Vz=np.zeros(2, dtype=np.float64),
        My=np.zeros(2, dtype=np.float64),
        Mz=np.zeros(2, dtype=np.float64),
        T=np.zeros(2, dtype=np.float64),
    )
    model = SimpleNamespace(
        n_nodes=2,
        elements=[SimpleNamespace(z_mid=0.5)],
        span_axis=2,
        X_ref=np.zeros((2, 3), dtype=np.float64),
    )
    beam_order = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
        ],
        dtype=np.float64,
    )
    solve_res = SimpleNamespace(
        z_stations_out=z,
        resultants=beam_order,
        z_nodal_out=None,
        nodal_R=None,
        nodal_positions=np.zeros((2, 3), dtype=np.float64),
    )

    monkeypatch.setattr(
        beam_distributed.BeamAnalysis,
        "from_blade_geometry",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(model=model)),
    )
    monkeypatch.setattr(beam_distributed, "solve_static", lambda *args, **kwargs: solve_res)
    monkeypatch.setattr(
        beam_distributed,
        "sample_resultants_at_z",
        lambda z_query, z_out, resultants: np.asarray(resultants, dtype=np.float64),
    )

    state = GlobalBeamResultantDriver(curves, n_beam_nodes=2).drive(
        np.repeat(np.eye(7, dtype=np.float64)[None, :, :], 2, axis=0),
        loads,
        bg,
        K6_stack=np.repeat(np.eye(6, dtype=np.float64)[None, :, :], 2, axis=0),
    )

    np.testing.assert_allclose(
        state.resultants,
        np.array(
            [
                [10.0, 40.0, 50.0, 60.0, 20.0, 30.0, 70.0],
                [11.0, 41.0, 51.0, 61.0, 21.0, 31.0, 71.0],
            ],
            dtype=np.float64,
        ),
    )
