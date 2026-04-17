"""CLI: build or load a fused recovery cache → :class:`~recovery_cache.engine.cache.RecoveryCache`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blade_precompute.beam_model.engine.kinematics import rotmat_from_small_curvature
from blade_precompute.design_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_precompute.design_optimisation.engine.section_builder import SectionBuilder
from blade_utilities.stress_recovery import RecoveryCache, RecoveryCacheBuilder, load_cache
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ply_gfrp(t_ply: float) -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp",
        E1=42e9,
        E2=12e9,
        G12=4.5e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=t_ply,
        Xt=900e6,
        Xc=650e6,
        Yt=65e6,
        Yc=120e6,
        S12=75e6,
        Zt=45e6,
        S13=40e6,
        S23=40e6,
    )


def _ply_cfrp(t_ply: float) -> OrthotropicPly:
    return OrthotropicPly(
        name="cfrp",
        E1=135e9,
        E2=9e9,
        G12=4.8e9,
        nu12=0.28,
        rho=1600.0,
        t_ply=t_ply,
        Xt=1800e6,
        Xc=1200e6,
        Yt=60e6,
        Yc=220e6,
        S12=90e6,
        Zt=55e6,
        S13=45e6,
        S23=45e6,
    )


def _build_smoke_cache() -> RecoveryCache:
    n_s = 3
    z = np.linspace(0.0, 4.0, n_s, dtype=np.float64)
    L = float(z[-1])
    r_ref = np.zeros((n_s, 3), dtype=np.float64)
    r_ref[:, 2] = z
    r_ref[:, 1] = 0.012 * (z / max(L, 1e-12)) ** 2
    kappa0 = np.zeros((n_s, 3), dtype=np.float64)
    kappa0[:, 1] = 0.0012
    tau0 = np.zeros_like(z)
    chord = np.linspace(1.8, 1.2, n_s, dtype=np.float64)
    twist = np.zeros_like(z)
    web_positions = np.array([-0.32, 0.32], dtype=np.float64)
    t0 = 0.0002
    lam_skin = LaminateDefinition(
        plies=[
            (_ply_gfrp(t0), 0.0),
            (_ply_gfrp(t0), 45.0),
            (_ply_gfrp(t0), -45.0),
        ],
        shear_lag_correction=True,
    )
    lam_cap = LaminateDefinition(plies=[(_ply_cfrp(t0), 0.0)] * 4, shear_lag_correction=True)
    lam_web = LaminateDefinition(
        plies=[(_ply_gfrp(t0), 45.0), (_ply_gfrp(t0), -45.0)],
        shear_lag_correction=True,
    )
    al = IsotropicMaterial(
        name="al6082",
        E=70e9,
        nu=0.33,
        rho=2700.0,
        sigma_allow=260e6,
    )
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
            "skin": lam_skin,
            "cap_ps": lam_cap,
            "web": lam_web,
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
    sections = SectionBuilder.build(dv, bg)
    solver = MidsurfaceSectionSolver()
    section_results = [solver.solve_one(s) for s in sections]
    nodal_R = np.stack([rotmat_from_small_curvature(bg.kappa0[i]) for i in range(n_s)], axis=0)
    storage = RecoveryCacheBuilder.build(
        section_results,
        sections[0].subcomponents,
        bg.z_stations,
        nodal_R_stack=nodal_R,
        enable_tier3=False,
    )
    return RecoveryCache(**storage.__dict__)


def _summarise(cache: RecoveryCache) -> None:
    print("RecoveryCache:")
    print(f"  z_stations: {cache.z_stations.shape}")
    print(f"  L_rec shape: {cache.L_rec.shape}")
    print(f"  L_iso shape: {cache.L_iso.shape}")
    print(f"  composite_subcomp_names: {cache.composite_subcomp_names}")
    print(f"  isotropic_subcomp_names: {cache.isotropic_subcomp_names}")


def main() -> None:
    p = argparse.ArgumentParser(description="Load or build a smoke recovery cache (canonical: RecoveryCache).")
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional .npz written by recovery_cache.save_cache; default builds an in-memory smoke cache.",
    )
    args = p.parse_args()
    if args.cache is not None:
        cache = load_cache(args.cache)
    else:
        cache = _build_smoke_cache()
    _summarise(cache)


if __name__ == "__main__":
    main()
