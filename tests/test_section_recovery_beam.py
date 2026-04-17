"""Beam ↔ section recovery ordering and optional enrich smoke."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.design_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_precompute.design_optimisation.engine.section_builder import SectionBuilder
from blade_precompute.global_beam_model.core.types import BeamSolveResult
from blade_precompute.global_beam_model.engine.constitutive import beam_resultants_to_section_recovery_order
from blade_precompute.global_beam_model.engine.section_recovery import enrich_beam_result_with_section_stress
from blade_precompute.section_properties.api import SectionAnalysis


def test_beam_resultants_to_section_recovery_order_rows() -> None:
    r = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        ],
        dtype=np.float64,
    )
    out = beam_resultants_to_section_recovery_order(r)
    np.testing.assert_allclose(
        out[0],
        np.array([1.0, 4.0, 5.0, 6.0, 2.0, 3.0, 7.0], dtype=np.float64),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        out[1],
        np.array([10.0, 40.0, 50.0, 60.0, 20.0, 30.0, 70.0], dtype=np.float64),
        rtol=0,
        atol=0,
    )


def _tiny_geometry(n_s: int = 3) -> tuple[OptimBladeGeometry, DesignVector]:
    z = np.linspace(0.0, 2.0, n_s, dtype=np.float64)
    r_ref = np.zeros((n_s, 3), dtype=np.float64)
    r_ref[:, 2] = z
    from blade_precompute.section_properties.engine.laminate import LaminateDefinition
    from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly

    t0 = 0.0002
    p = OrthotropicPly(
        name="p",
        E1=40e9,
        E2=10e9,
        G12=4e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=t0,
        Xt=900e6,
        Xc=650e6,
        Yt=65e6,
        Yc=120e6,
        S12=75e6,
        Zt=45e6,
        S13=40e6,
        S23=40e6,
    )
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 45.0)])
    al = IsotropicMaterial(name="al", E=70e9, nu=0.33, rho=2700.0, sigma_allow=260e6)
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=np.zeros((n_s, 3)),
        tau0=np.zeros_like(z),
        chord=np.ones_like(z),
        twist=np.zeros_like(z),
        airfoil_profiles=[],
        web_positions=np.array([-0.3, 0.3], dtype=np.float64),
        subcomponent_materials={
            "skin": lam,
            "cap_ps": lam,
            "web": lam,
            "leading_edge_insert": al,
        },
        thickness_role={"leading_edge_insert": "fixed"},
        box_height_frac=0.11,
    )
    dv = DesignVector(
        t_skin=np.full(n_s, 0.010),
        t_cap=np.full(n_s, 0.040),
        t_web=np.full(n_s, 0.012),
    )
    return bg, dv


def test_enrich_beam_result_with_section_stress_smoke() -> None:
    pytest.importorskip("blade_utilities.recovery_operators")
    pytest.importorskip("blade_utilities.stress_recovery")

    bg, dv = _tiny_geometry(3)
    section_defs = SectionBuilder.build(dv, bg)
    analysis = SectionAnalysis()
    results = tuple(analysis.solve(sd) for sd in section_defs)
    z_sec = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)
    n_s = z_sec.shape[0]

    n_nodes = 4
    n_gp = n_s * 2
    z_gp = np.linspace(float(z_sec[0]), float(z_sec[-1]), n_gp)
    rng = np.random.default_rng(0)
    nodal_r = rng.standard_normal((n_nodes, 7)) * 1e3
    res0 = BeamSolveResult(
        nodal_positions=np.zeros((n_nodes, 3)),
        nodal_rotations=np.zeros((n_nodes, 3)),
        nodal_R=np.stack([np.eye(3)] * n_nodes),
        nodal_warping=np.zeros(n_nodes),
        resultants=np.tile(nodal_r.mean(axis=0), (n_gp, 1)),
        strains=np.zeros((n_gp, 7)),
        converged=True,
        n_iterations=1,
        residual_norm=0.0,
        iteration_history=[],
        z_stations_out=z_gp,
        z_nodal_out=np.linspace(float(z_sec[0]), float(z_sec[-1]), n_nodes),
        resultants_nodal=nodal_r,
        strains_nodal=np.zeros((n_nodes, 7)),
    )

    out = enrich_beam_result_with_section_stress(
        res0,
        station_z=z_sec,
        section_results=results,
        section_definitions=tuple(section_defs),
    )
    assert out.z_section_recovery is not None
    assert out.section_stress_voigt_gp is not None and out.section_stress_voigt_gp.shape == (n_s, 3)
    assert out.section_stress_voigt_nodal is not None
    assert out.section_strain_maxabs_gp is not None and out.section_strain_maxabs_gp.shape == (n_s, 6)
    assert out.section_tsai_wu_fi_max_gp is not None and out.section_tsai_wu_fi_max_gp.shape == (n_s,)
    assert out.section_tsai_wu_fi_ply_envelope_gp is not None
    assert out.section_tsai_wu_fi_ply_envelope_gp.shape[0] == n_s
    assert out.section_tsai_wu_fi_ply_envelope_gp.shape == out.section_tsai_wu_fi_ply_envelope_nodal.shape
    assert out.section_von_mises_fi_max_gp is not None and out.section_von_mises_fi_max_gp.shape == (n_s,)
    assert out.section_delamination_fi_max_gp is not None
    assert out.section_delamination_fi_max_gp.shape == (n_s,)
    assert out.section_stress_voigt_secframe_gp is not None and out.section_stress_voigt_secframe_gp.shape == (n_s, 3)
    assert out.section_d_tsai_wu_fi_dz_gp is not None and out.section_d_tsai_wu_fi_dz_gp.shape == (n_s,)


def test_recover_all_fi_matches_individual_evaluators() -> None:
    pytest.importorskip("blade_utilities.stress_recovery")
    import dataclasses

    from blade_utilities.stress_recovery import RecoveryCache, RecoveryCacheBuilder

    bg, dv = _tiny_geometry(4)
    section_defs = SectionBuilder.build(dv, bg)
    analysis = SectionAnalysis()
    results = tuple(analysis.solve(sd) for sd in section_defs)
    z_sec = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)
    sub0 = section_defs[0].subcomponents
    storage = RecoveryCacheBuilder.build(
        section_results=list(results),
        section0_subcomponents=sub0,
        z_stations=z_sec,
        nodal_R_stack=None,
        enable_tier3=True,
    )
    cache = RecoveryCache(**dataclasses.asdict(storage))
    rng = np.random.default_rng(42)
    r_case = rng.standard_normal((1, z_sec.shape[0], 7)) * 500.0
    tw_a, vm_a, del_a = cache.recover_all_fi(r_case)
    np.testing.assert_allclose(tw_a, cache.eval_tsai_wu_fi(r_case), rtol=0, atol=1e-12)
    np.testing.assert_allclose(vm_a, cache.eval_von_mises_fi(r_case), rtol=0, atol=1e-12)
    del_b = cache.eval_delamination_fi(r_case)
    assert del_b is not None
    np.testing.assert_allclose(del_a, del_b, rtol=0, atol=1e-12)


def test_save_section_recovery_cache_to_npz(tmp_path) -> None:
    pytest.importorskip("blade_utilities.stress_recovery")

    from blade_precompute.global_beam_model.engine.section_recovery import save_section_recovery_cache_to_npz

    bg, dv = _tiny_geometry(3)
    section_defs = SectionBuilder.build(dv, bg)
    analysis = SectionAnalysis()
    results = tuple(analysis.solve(sd) for sd in section_defs)
    z_sec = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)
    n_nodes = 3
    res0 = BeamSolveResult(
        nodal_positions=np.zeros((n_nodes, 3)),
        nodal_rotations=np.zeros((n_nodes, 3)),
        nodal_R=np.stack([np.eye(3)] * n_nodes),
        nodal_warping=np.zeros(n_nodes),
        resultants=np.zeros((6, 7)),
        strains=np.zeros((6, 7)),
        converged=True,
        n_iterations=1,
        residual_norm=0.0,
        iteration_history=[],
        z_stations_out=np.linspace(float(z_sec[0]), float(z_sec[-1]), 6),
        z_nodal_out=np.linspace(float(z_sec[0]), float(z_sec[-1]), n_nodes),
        resultants_nodal=np.zeros((n_nodes, 7)),
        strains_nodal=np.zeros((n_nodes, 7)),
    )
    outp = tmp_path / "cache.npz"
    save_section_recovery_cache_to_npz(
        res0,
        station_z=z_sec,
        section_results=results,
        section_definitions=tuple(section_defs),
        path=outp,
    )
    assert outp.is_file()


def test_section_recovery_plots_smoke() -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("blade_utilities.recovery_operators")

    bg, dv = _tiny_geometry(3)
    section_defs = SectionBuilder.build(dv, bg)
    analysis = SectionAnalysis()
    results = tuple(analysis.solve(sd) for sd in section_defs)
    z_sec = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)
    n_s = z_sec.shape[0]
    n_nodes = 4
    n_gp = n_s * 2
    z_gp = np.linspace(float(z_sec[0]), float(z_sec[-1]), n_gp)
    rng = np.random.default_rng(1)
    nodal_r = rng.standard_normal((n_nodes, 7)) * 1e3
    res0 = BeamSolveResult(
        nodal_positions=np.zeros((n_nodes, 3)),
        nodal_rotations=np.zeros((n_nodes, 3)),
        nodal_R=np.stack([np.eye(3)] * n_nodes),
        nodal_warping=np.zeros(n_nodes),
        resultants=np.tile(nodal_r.mean(axis=0), (n_gp, 1)),
        strains=np.zeros((n_gp, 7)),
        converged=True,
        n_iterations=1,
        residual_norm=0.0,
        iteration_history=[],
        z_stations_out=z_gp,
        z_nodal_out=np.linspace(float(z_sec[0]), float(z_sec[-1]), n_nodes),
        resultants_nodal=nodal_r,
        strains_nodal=np.zeros((n_nodes, 7)),
    )
    out = enrich_beam_result_with_section_stress(
        res0,
        station_z=z_sec,
        section_results=results,
        section_definitions=tuple(section_defs),
    )
    import matplotlib.pyplot as plt

    from blade_precompute.global_beam_model.interface import plot as bmplot

    for pfn in (
        bmplot.plot_spanwise_section_stress,
        bmplot.plot_spanwise_section_strain_laminate,
        bmplot.plot_spanwise_section_tsai_wu,
        bmplot.plot_spanwise_section_von_mises_fi,
        bmplot.plot_spanwise_section_delamination_fi,
        bmplot.plot_spanwise_section_stress_secframe,
        bmplot.plot_spanwise_section_d_tsai_wu_dz,
        lambda r: bmplot.plot_spanwise_section_tsai_wu_fi_heatmap(r, source="gp"),
        lambda r: bmplot.plot_spanwise_section_tsai_wu_fi_heatmap(r, source="nodal"),
    ):
        fig, _ = pfn(out)
        plt.close(fig)
