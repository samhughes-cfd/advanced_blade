"""Parity and consistency checks for blade_utilities.recovery operator API."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver

from blade_utilities.recovery import (
    apply_section_stress_operator,
    apply_span_derivative,
    apply_strain_operator,
    build_recovery_operator_bundle,
)


def _tiny_blade(n_s: int = 3) -> tuple[OptimBladeGeometry, DesignVector]:
    z = np.linspace(0.0, 2.0, n_s, dtype=np.float64)
    r_ref = np.zeros((n_s, 3), dtype=np.float64)
    r_ref[:, 2] = z
    r_ref[:, 1] = 0.01 * (z / z[-1]) ** 2
    kappa0 = np.zeros((n_s, 3), dtype=np.float64)
    kappa0[:, 1] = 0.002
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
    lam_cap = LaminateDefinition(plies=[(p, 0.0), (p, 0.0), (p, 0.0)])
    al = IsotropicMaterial(name="al", E=70e9, nu=0.33, rho=2700.0, sigma_allow=260e6)
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=kappa0,
        chord=chord,
        twist=twist,
        airfoil_profiles=[],
        web_positions=web_positions,
        subcomponent_materials={
            "skin": lam,
            "cap_ps": lam_cap,
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


def test_strain_and_section_stress_operator_match_explicit_chain() -> None:
    bg, dv = _tiny_blade(3)
    sections = SectionBuilder.build(dv, bg)
    solver = MidsurfaceSectionSolver()
    results = [solver.solve_one(s) for s in sections]
    bundle = build_recovery_operator_bundle(
        section_results=results,
        z_stations=bg.z_stations,
        nodal_R=np.stack([np.eye(3)] * len(results), axis=0),
        section0_subcomponents=sections[0].subcomponents,
    )

    rng = np.random.default_rng(0)
    r = rng.standard_normal((5, len(results), 7)).astype(np.float64)

    eps = apply_strain_operator(bundle, r)
    sig_sec = apply_section_stress_operator(bundle, r)

    B = np.stack([res.composite_resultant_basis for res in results], axis=0)
    Ainv = np.stack([res.ABD_inv for res in results], axis=0)
    Q_bar = np.stack([res.Q_bar for res in results], axis=0)
    z_ply = np.stack([res.z_ply for res in results], axis=0)

    n_case = r.shape[0]
    n_s = r.shape[1]
    n_comp = B.shape[1]
    n_ply = Q_bar.shape[2]
    eps_ref = np.zeros((n_case, n_s, n_comp, 6), dtype=np.float64)
    sig_ref = np.zeros((n_case, n_s, n_comp, n_ply, 3), dtype=np.float64)
    for c in range(n_case):
        for s in range(n_s):
            for p in range(n_comp):
                for j in range(7):
                    strain6 = Ainv[s, p] @ B[s, p, j]
                    eps_ref[c, s, p] += r[c, s, j] * strain6
                    for k in range(n_ply):
                        e = strain6[:3] + z_ply[s, p, k] * strain6[3:6]
                        sig_ref[c, s, p, k] += r[c, s, j] * (Q_bar[s, p, k] @ e)

    np.testing.assert_allclose(eps, eps_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sig_sec, sig_ref, rtol=1e-10, atol=1e-10)


def test_m_voigt_matches_rotation_tensor_formula() -> None:
    rng = np.random.default_rng(1)
    n_s = 4
    r_stack = np.zeros((n_s, 3, 3), dtype=np.float64)
    for i in range(n_s):
        a = rng.standard_normal((3, 3))
        q, _ = np.linalg.qr(a)
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        r_stack[i] = q

    bg, dv = _tiny_blade(n_s)
    sections = SectionBuilder.build(dv, bg)
    solver = MidsurfaceSectionSolver()
    results = [solver.solve_one(s) for s in sections]
    bundle = build_recovery_operator_bundle(
        section_results=results,
        z_stations=bg.z_stations,
        nodal_R=r_stack,
        section0_subcomponents=sections[0].subcomponents,
    )

    for s in range(n_s):
        voigt = rng.standard_normal(3)
        s11, s22, t12 = voigt
        full = np.array([[s11, t12, 0.0], [t12, s22, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        spr = r_stack[s] @ full @ r_stack[s].T
        ref = np.array([spr[0, 0], spr[1, 1], spr[0, 1]], dtype=np.float64)
        np.testing.assert_allclose(bundle.M_voigt[s] @ voigt, ref, rtol=1e-10, atol=1e-10)


def test_dz_matches_quadratic_derivative_on_nonuniform_grid() -> None:
    z = np.array([0.0, 0.3, 0.9, 1.5, 2.2], dtype=np.float64)
    f = z**2
    dfdz_ref = 2.0 * z

    bg, dv = _tiny_blade(len(z))
    bg = OptimBladeGeometry(**{**bg.__dict__, "z_stations": z})
    sections = SectionBuilder.build(dv, bg)
    solver = MidsurfaceSectionSolver()
    results = [solver.solve_one(s) for s in sections]
    bundle = build_recovery_operator_bundle(
        section_results=results,
        z_stations=z,
        nodal_R=None,
        section0_subcomponents=sections[0].subcomponents,
    )

    dfdz = apply_span_derivative(bundle, f)
    np.testing.assert_allclose(dfdz, dfdz_ref, rtol=1e-10, atol=1e-10)
