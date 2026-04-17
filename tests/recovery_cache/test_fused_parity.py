"""Parity of fused recovery vs explicit Tier-1 + Tier-2 chain."""

from __future__ import annotations

import numpy as np

from blade_precompute.design_optimisation.engine.section_builder import SectionBuilder
from blade_precompute.design_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_utilities.recovery import (
    RecoveryCache,
    build_recovery_cache,
    load_cache,
    plane_stress_voigt_from_R,
    save_cache,
)
from blade_precompute.section_properties.engine.clpt_recovery import clpt_ply_stresses_section_frame, rotate_plies_to_material
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


def test_fused_ply_and_iso_match_explicit_chain_identity_R(tmp_path):
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
            enable_tier3=True,
        ).__dict__
    )

    rng = np.random.default_rng(0)
    r = rng.standard_normal((5, len(results), 7)).astype(np.float64)
    sig_fused = cache.recover_ply_stresses(r)

    comp_basis = np.stack([res.composite_resultant_basis for res in results], axis=0)
    iso_basis = np.stack([res.isotropic_resultant_basis for res in results], axis=0)
    comp_res = np.einsum("csm,spmr->cspr", r, comp_basis, optimize=True)
    iso_res = np.einsum("csm,spmr->cspr", r, iso_basis, optimize=True)
    ABD_inv = np.stack([res.ABD_inv for res in results], axis=0)
    Q_bar = np.stack([res.Q_bar for res in results], axis=0)
    T_ply = np.stack([res.T_ply for res in results], axis=0)
    z_ply = np.stack([res.z_ply for res in results], axis=0)
    sigma_sec = clpt_ply_stresses_section_frame(comp_res, ABD_inv, Q_bar, z_ply)
    sigma_mat = rotate_plies_to_material(sigma_sec, T_ply)
    np.testing.assert_allclose(sig_fused, sigma_mat, rtol=1e-10, atol=1e-10)

    iso_t = np.stack([res.iso_thickness for res in results], axis=0)
    sigma_iso_ref = iso_res / np.maximum(iso_t[:, :, None], 1e-18)
    sigma_iso = cache.recover_iso_stresses(r)
    np.testing.assert_allclose(sigma_iso, sigma_iso_ref, rtol=1e-10, atol=1e-10)

    path = tmp_path / "cache.npz"
    save_cache(cache, str(path))
    loaded = load_cache(str(path))
    np.testing.assert_allclose(loaded.L_rec, cache.L_rec)
    np.testing.assert_allclose(loaded.L_iso, cache.L_iso)
    assert loaded.enable_tier3 == cache.enable_tier3


def test_plane_stress_rotation_matches_tensor_formula():
    rng = np.random.default_rng(1)
    for _ in range(5):
        a = rng.standard_normal((3, 3))
        r, _ = np.linalg.qr(a)
        if np.linalg.det(r) < 0:
            r[:, 0] *= -1.0
        M = plane_stress_voigt_from_R(r)
        voigt = rng.standard_normal(3)
        s11, s22, t12 = voigt
        full = np.array([[s11, t12, 0.0], [t12, s22, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        spr = r @ full @ r.T
        ref = np.array([spr[0, 0], spr[1, 1], spr[0, 1]], dtype=np.float64)
        np.testing.assert_allclose(M @ voigt, ref, rtol=1e-10, atol=1e-10)
