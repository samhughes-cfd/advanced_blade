"""CLI: stress history → rainflow → Miner; canonical :class:`~blade_analysis.fatigue_damage.core.types.FatigueResult`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blade_precompute.beam_model.engine.kinematics import rotmat_from_small_curvature
from blade_precompute.design_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_precompute.design_optimisation.engine.section_builder import SectionBuilder
from blade_analysis.fatigue_damage import FatigueAnalysis, SNcurve
from blade_analysis.fatigue_damage.core.loads import ResultantHistory
from blade_utilities.stress_recovery import RecoveryCache, RecoveryCacheBuilder, load_cache
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver


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


def _build_smoke_recovery_cache() -> RecoveryCache:
    """Same compact geometry as ``recovery_cache.__main__`` (keep smoke paths aligned)."""
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


def _default_sn_curves() -> dict[str, SNcurve]:
    return {
        "GFRP": SNcurve.gfrp_blade(),
        "CFRP": SNcurve.cfrp_blade(),
        "default": SNcurve.gfrp_blade(),
        "aluminium": SNcurve.steel_dnv(),
    }


def _smoke_history(z_stations: np.ndarray) -> ResultantHistory:
    n_t = 256
    t = np.linspace(0.0, 1.0, n_t, dtype=np.float64)
    n_s = int(z_stations.shape[0])
    base = 5.0e3 * np.sin(2.0 * np.pi * 7.0 * t)[:, None]
    zf = z_stations.reshape(1, n_s) / max(float(z_stations[-1]), 1e-12)
    my = base * (0.5 + 0.5 * zf)
    zeros = np.zeros((n_t, n_s), dtype=np.float64)
    return ResultantHistory(
        z_stations=z_stations,
        time=t,
        N=zeros,
        Vy=zeros,
        Vz=zeros,
        My=my,
        Mz=zeros,
        T=zeros,
        B=zeros,
    )


def _summarise(fr: object) -> None:
    print("FatigueResult:")
    print(f"  max_damage_composite: {getattr(fr, 'max_damage_composite'):.6g}")
    print(f"  max_damage_isotropic: {getattr(fr, 'max_damage_isotropic'):.6g}")
    print(f"  fatigue_critical_material: {getattr(fr, 'fatigue_critical_material')!r}")
    print(f"  design_life_years: {getattr(fr, 'design_life_years')}")
    print(f"  memory_mode: {getattr(fr, 'memory_mode')!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run fatigue pipeline smoke (canonical: FatigueResult).")
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional recovery cache .npz; default matches recovery_cache built-in smoke.",
    )
    p.add_argument("--plot", action="store_true", help="Show matplotlib figures (requires matplotlib).")
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Save fatigue plots to a multi-page PDF at this path instead of showing.",
    )
    args = p.parse_args()
    if args.cache is not None:
        cache = load_cache(args.cache)
    else:
        cache = _build_smoke_recovery_cache()
    hist = _smoke_history(np.asarray(cache.z_stations, dtype=np.float64))
    analysis = FatigueAnalysis.from_cache(cache, _default_sn_curves(), design_life_years=25.0)
    res = analysis.run(hist, memory_limit_mb=512.0)
    _summarise(res)

    if args.plot or args.plot_out is not None:
        from blade_analysis.fatigue_damage.interface import plot as fplot

        z = np.asarray(cache.z_stations, dtype=np.float64)
        figs = []
        fig, _ = fplot.plot_damage_life_vs_span(res, z)
        figs.append(fig)
        fig, _ = fplot.plot_static_fi_vs_span(res, z)
        figs.append(fig)
        bins = res.rainflow_bins
        if bins is not None:
            fig, _ = fplot.plot_rainflow_composite(bins, station=0, subcomp=0, ply=0)
            figs.append(fig)
            if bins.counts_iso.size and float(np.sum(bins.counts_iso)) > 0.0:
                fig, _ = fplot.plot_rainflow_isotropic(bins, station=0, subcomp=0)
                figs.append(fig)
            rc = np.asarray(bins.ranges_comp[:, 0, 0, 0], dtype=np.float64)
            fig, _ = fplot.plot_sn_curve_with_ranges(SNcurve.gfrp_blade(), rc)
            figs.append(fig)
        if args.plot_out is not None:
            outp = args.plot_out.resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(outp) as pdf:
                for f in figs:
                    pdf.savefig(f, bbox_inches="tight")
                    plt.close(f)
            print(f"Saved fatigue plots to {outp}")
        else:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
