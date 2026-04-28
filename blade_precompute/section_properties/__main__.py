"""CLI: minimal midsurface section solve → :class:`~section_model.core.types.SectionSolveResult`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blade_precompute.section_properties.api import SectionAnalysis
from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly


def _smoke_section() -> SectionDefinition:
    ply = OrthotropicPly(
        name="glass_ud",
        E1=40e9,
        E2=10e9,
        G12=4e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=0.0003,
        Xt=1.0e9,
        Xc=0.8e9,
        Yt=40e6,
        Yc=120e6,
        S12=50e6,
        Zt=50e6,
        S13=40e6,
        S23=40e6,
    )
    lam = LaminateDefinition(plies=[(ply, 0.0), (ply, 45.0), (ply, -45.0), (ply, 0.0)])
    skin = SubcomponentGeometry(
        name="skin_ps",
        midsurface_coords=np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.02]]),
        material=lam,
        thickness=lam.total_thickness(),
        strip_width_m=0.05,
    )
    al = IsotropicMaterial(
        name="aluminium_6082",
        E=70e9,
        nu=0.33,
        rho=2700.0,
        sigma_allow=270e6,
    )
    insert = SubcomponentGeometry(
        name="leading_edge_insert",
        midsurface_coords=np.array([[0.5, 0.02], [0.52, 0.06]]),
        material=al,
        thickness=0.004,
        strip_width_m=0.02,
    )
    return SectionDefinition(station_z=0.0, subcomponents=[skin, insert])


def _summarise(res: object) -> None:
    print("SectionSolveResult:")
    print(f"  K6 shape: {getattr(res, 'K6').shape}")
    print(f"  K7 shape: {getattr(res, 'K7').shape}")
    print(f"  mass_per_length [kg/m]: {getattr(res, 'mass_per_length'):.6g}")
    print(f"  area [m²]: {getattr(res, 'area'):.6g}")
    print(f"  composite_subcomp_names: {getattr(res, 'composite_subcomp_names')}")
    print(f"  isotropic_subcomp_names: {getattr(res, 'isotropic_subcomp_names')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run a minimal section solve (canonical: SectionSolveResult).")
    p.add_argument(
        "--section-spec",
        type=Path,
        default=None,
        help="Optional path to section spec JSON; default is an in-memory smoke section.",
    )
    args = p.parse_args()
    analysis = SectionAnalysis()
    if args.section_spec is not None:
        res = analysis.load_and_solve(args.section_spec)
    else:
        res = analysis.solve(_smoke_section())
    _summarise(res)


if __name__ == "__main__":
    main()
