"""Global beam resultant driver matches direct :class:`BeamAnalysis` (distributed loads)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.core.types import SolverOptions
from blade_precompute.global_beam_model.engine.constitutive import beam_resultants_to_section_recovery_order
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
from blade_precompute.global_beam_model.engine.postprocess import sample_resultants_at_z
from blade_precompute.global_beam_model.engine.solver import solve_static
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import OrthotropicPly
from blade_precompute.section_optimisation.core.types import (
    DistributedLoadCurves,
    ExtremeLoads,
    OptimBladeGeometry,
)
from blade_precompute.global_beam_model.engine.axial_loading import AxialLoadingConfig, q_x_distributed
from blade_precompute.section_optimisation.engine import beam_distributed
from blade_precompute.section_optimisation.engine.beam_distributed import (
    GlobalBeamResultantDriver,
    build_beam_loads_distributed,
)
from blade_precompute.section_optimisation.engine.beam_k7 import PrescribedResultantDriver


def _synthetic_bg(n: int = 4) -> OptimBladeGeometry:
    z = np.linspace(0.0, 2.0, n, dtype=np.float64)
    r_ref = np.stack([np.zeros(n), np.zeros(n), z], axis=1)
    ply = OrthotropicPly(
        name="p",
        t_ply=0.0002,
        rho=1500.0,
        E1=40e9,
        E2=10e9,
        G12=3.5e9,
        nu12=0.3,
        Xt=1500e6,
        Xc=1200e6,
        Yt=50e6,
        Yc=200e6,
        S12=70e6,
        Zt=50e6,
        S13=30e6,
        S23=30e6,
    )
    lam = LaminateDefinition(plies=[(ply, 0.0), (ply, 90.0)])
    skin = "skin_0"
    radial_r_m = 5.0 + z * 2.0
    return OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=np.zeros((n, 3), dtype=np.float64),
        chord=np.full(n, 0.5, dtype=np.float64),
        twist=np.zeros(n, dtype=np.float64),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2), dtype=np.float64),
        subcomponent_materials={skin: lam},
        thickness_role={skin: "skin"},
        radial_r_m=radial_r_m,
    )


def test_global_beam_driver_returns_section_resultant_order(monkeypatch) -> None:
    bg = _synthetic_bg(2)
    z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    curves = DistributedLoadCurves(
        loads_r_z_m=z,
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )
    ex = ExtremeLoads(
        z_stations=z,
        N=np.zeros(2, dtype=np.float64),
        Vy=np.zeros(2, dtype=np.float64),
        Vz=np.zeros(2, dtype=np.float64),
        My=np.zeros(2, dtype=np.float64),
        Mz=np.zeros(2, dtype=np.float64),
        T=np.zeros(2, dtype=np.float64),
    )
    beam_order = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
        ],
        dtype=np.float64,
    )
    model = SimpleNamespace(
        n_nodes=2,
        elements=[SimpleNamespace(z_mid=0.5)],
        span_axis=2,
        X_ref=np.zeros((2, 3), dtype=np.float64),
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
        ex,
        bg,
        K6_stack=np.repeat(np.eye(6, dtype=np.float64)[None, :, :], 2, axis=0),
    )

    np.testing.assert_allclose(
        state.resultants,
        beam_resultants_to_section_recovery_order(beam_order),
        rtol=0,
        atol=0,
    )
    assert state.beam_solve is solve_res


def test_global_beam_driver_matches_direct_solve() -> None:
    bg = _synthetic_bg(4)
    z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    n = int(z.shape[0])
    K6_template = np.diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6]).astype(np.float64)
    K7_template = np.zeros((7, 7), dtype=np.float64)
    K7_template[:6, :6] = K6_template
    K7_template[6, 6] = 1e4
    K6 = np.stack([K6_template.copy() for _ in range(n)], axis=0)
    K7 = np.stack([K7_template.copy() for _ in range(n)], axis=0)
    n_beam = 16
    stations = stations_from_arrays(z, K6, K7)
    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        chord=np.asarray(bg.chord, dtype=np.float64),
        twist=np.asarray(bg.twist, dtype=np.float64),
        airfoil_profiles=list(bg.airfoil_profiles),
        web_positions=np.asarray(bg.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg.subcomponent_materials),
        chi0=None,
    )
    analysis = BeamAnalysis.from_blade_geometry(geom, n_beam, stations, span_axis=2)
    model = analysis.model
    curves = DistributedLoadCurves(
        loads_r_z_m=z,
        q_y_Npm=50.0 * np.ones(n, dtype=np.float64),
        q_z_Npm=np.zeros(n, dtype=np.float64),
        m_x_Nmpm=np.zeros(n, dtype=np.float64),
    )
    opt = SolverOptions(
        max_iter=50,
        tol_res=1e-1,
        tol_res_rel=1e-2,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=12,
        full_fd_hessian=False,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    loads = build_beam_loads_distributed(geom, model, curves)
    res = solve_static(model, loads, options=opt)
    assert res.z_stations_out is not None
    R_direct = sample_resultants_at_z(z, res.z_stations_out, res.resultants)

    ex = ExtremeLoads(
        z_stations=z,
        N=np.zeros(n),
        Vy=np.zeros(n),
        Vz=np.zeros(n),
        My=np.zeros(n),
        Mz=np.zeros(n),
        T=np.zeros(n),
    )
    drv = GlobalBeamResultantDriver(
        curves, n_beam_nodes=n_beam, solver_options=opt
    )
    st = drv.drive(K7, ex, bg, K6_stack=K6)
    np.testing.assert_allclose(
        st.resultants,
        beam_resultants_to_section_recovery_order(R_direct),
        rtol=1e-4,
        atol=0.1,
    )


def test_prescribed_driver_ignores_k6_kwargs() -> None:
    bg = _synthetic_bg(3)
    n = int(bg.z_stations.shape[0])
    K6 = np.stack([np.eye(6) for _ in range(n)], axis=0) * 1e6
    K7 = np.zeros((n, 7, 7), dtype=np.float64)
    K7[:, :6, :6] = K6
    K7[:, 6, 6] = 1e4
    ex = ExtremeLoads(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        N=np.zeros(n),
        Vy=np.zeros(n),
        Vz=np.zeros(n),
        My=np.zeros(n),
        Mz=np.zeros(n),
        T=np.zeros(n),
    )
    d = PrescribedResultantDriver()
    a = d.drive(K7, ex, bg)
    b = d.drive(K7, ex, bg, K6_stack=K6)
    np.testing.assert_array_equal(a.resultants, b.resultants)


def test_build_beam_loads_merges_spanwise_axial_into_global_span_axis() -> None:
    """`q_x_Npm` (spanwise) must add to ``distributed_q[:, span_axis]`` (``span_axis=2`` → global z)."""
    bg = _synthetic_bg(4)
    z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    n = int(z.shape[0])
    r_tip = float(np.max(np.asarray(bg.radial_r_m, dtype=np.float64)))
    K6_template = np.diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6]).astype(np.float64)
    K7_template = np.zeros((7, 7), dtype=np.float64)
    K7_template[:6, :6] = K6_template
    K7_template[6, 6] = 1e4
    K6 = np.stack([K6_template.copy() for _ in range(n)], axis=0)
    K7 = np.stack([K7_template.copy() for _ in range(n)], axis=0)
    n_beam = 12
    stations = stations_from_arrays(z, K6, K7)
    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        chord=np.asarray(bg.chord, dtype=np.float64),
        twist=np.asarray(bg.twist, dtype=np.float64),
        airfoil_profiles=list(bg.airfoil_profiles),
        web_positions=np.asarray(bg.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg.subcomponent_materials),
        chi0=None,
    )
    analysis = BeamAnalysis.from_blade_geometry(geom, n_beam, stations, span_axis=2)
    model = analysis.model
    cfg = AxialLoadingConfig(
        u_inf_m_s=20.0,
        tip_speed_ratio=2.0,
        r_tip_m=r_tip,
        gravity_m_s2=0.0,
        azimuth_deg=90.0,
        enabled=True,
    )
    n_s = n
    mu_line = np.full(n_s, 0.3, dtype=np.float64)
    curves0 = DistributedLoadCurves(
        loads_r_z_m=z,
        q_y_Npm=50.0 * np.ones(n, dtype=np.float64),
        q_z_Npm=np.zeros(n, dtype=np.float64),
        m_x_Nmpm=np.zeros(n, dtype=np.float64),
    )
    zq = np.asarray(curves0.loads_r_z_m, dtype=np.float64).ravel()
    r_b = np.asarray(bg.radial_r_m, dtype=np.float64).ravel()
    mu_q = np.interp(zq, z, mu_line)
    r_q = np.interp(zq, z, r_b)
    qx = q_x_distributed(zq, r_q, mu_q, cfg)
    curves1 = replace(curves0, q_x_Npm=qx)
    loads0 = build_beam_loads_distributed(geom, model, curves0)
    loads1 = build_beam_loads_distributed(geom, model, curves1)
    sa = int(getattr(model, "span_axis", 2))
    # Spanwise line load is merged into the global axis of the 1D reference, not into index 0.
    np.testing.assert_allclose(
        loads1.distributed_q[:, sa] - loads0.distributed_q[:, sa],
        np.interp(
            np.asarray([el.z_mid for el in model.elements], dtype=np.float64),
            zq,
            qx,
        ),
        rtol=0,
        atol=1e-4,
    )
    assert float(np.max(np.abs(loads1.distributed_q[:, 0]))) < 1e-9
