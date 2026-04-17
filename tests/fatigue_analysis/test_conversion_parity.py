"""Parity: fatigue conversion vs RecoveryCache fused recovery."""

from __future__ import annotations

import numpy as np

from blade_precompute.design_optimisation.engine.section_builder import SectionBuilder
from blade_precompute.design_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_analysis.fatigue_damage.core.loads import ResultantHistory
from blade_analysis.fatigue_damage.engine.conversion import resultants_to_stress_history
from blade_utilities.stress_recovery import RecoveryCache, build_recovery_cache
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver


def _tiny_blade(n_s: int = 3) -> tuple[OptimBladeGeometry, DesignVector]:
    z = np.linspace(0.0, 2.0, n_s, dtype=np.float64)
    r_ref = np.zeros((n_s, 3), dtype=np.float64)
    r_ref[:, 2] = z
    r_ref[:, 1] = 0.01 * (z / z[-1]) ** 2
    kappa0 = np.zeros((n_s, 3), dtype=np.float64)
    kappa0[:, 1] = 0.002
    tau0 = np.zeros_like(z)
    chord = np.full_like(z, 1.5)
    twist = np.zeros_like(z)
    web_positions = np.array([-0.3, 0.3], dtype=np.float64)
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
    lam = LaminateDefinition(plies=[(p, 0.0), (p, 45.0), (p, -45.0)])
    al = IsotropicMaterial(name="al", E=70e9, nu=0.33, rho=2700.0, sigma_allow=260e6)
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=kappa0,
        tau0=tau0,
        chord=chord,
        twist=twist,
        airfoil_profiles=[],
        web_positions=web_positions,
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


def test_resultants_to_stress_matches_cache_recovery():
    bg, dv = _tiny_blade(3)
    sections = SectionBuilder.build(dv, bg)
    solver = MidsurfaceSectionSolver()
    results = [solver.solve_one(s) for s in sections]
    nodal_R = np.stack([np.eye(3)] * len(results), axis=0)
    cache = RecoveryCache(
        **build_recovery_cache(
            section_results=results,
            z_stations=bg.z_stations,
            nodal_R=nodal_R,
            section0_subcomponents=sections[0].subcomponents,
            enable_tier3=False,
        ).__dict__
    )

    rng = np.random.default_rng(42)
    n_t = 8
    R_beam = rng.standard_normal((n_t, len(results), 7)).astype(np.float64)
    history = ResultantHistory(
        z_stations=bg.z_stations,
        time=np.linspace(0.0, 1.0, n_t, dtype=np.float64),
        N=R_beam[..., 0],
        Vy=R_beam[..., 1],
        Vz=R_beam[..., 2],
        My=R_beam[..., 3],
        Mz=R_beam[..., 4],
        T=R_beam[..., 5],
        B=R_beam[..., 6],
    )
    sh = resultants_to_stress_history(history, cache, chunk_size=4)

    R_cache = np.stack(
        [R_beam[..., 0], R_beam[..., 3], R_beam[..., 4], R_beam[..., 5], R_beam[..., 1], R_beam[..., 2], R_beam[..., 6]],
        axis=-1,
    )
    sig_ref = cache.recover_ply_stresses(R_cache)
    iso_ref = cache.recover_iso_stresses(R_cache)

    np.testing.assert_allclose(sh.sigma_composite, sig_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sh.sigma_isotropic, iso_ref, rtol=1e-10, atol=1e-10)
