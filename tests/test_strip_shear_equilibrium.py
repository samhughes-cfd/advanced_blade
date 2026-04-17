"""Strip-graph scaling for interlaminar screening."""

import numpy as np

from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import OrthotropicPly
from blade_precompute.section_properties.engine.mesh import build_line_mesh
from blade_precompute.section_properties.engine.elements import build_strip_fe_data
from blade_precompute.section_properties.engine.interlaminar_recovery import recover_interlaminar
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver
from blade_precompute.section_properties.engine.strip_shear_equilibrium import (
    recover_interlaminar_strip_equilibrium,
)


def _ud_ply() -> OrthotropicPly:
    return OrthotropicPly(
        name="ud",
        E1=40e9,
        E2=10e9,
        G12=4e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=2e-3,
        Xt=1e9,
        Xc=1e9,
        Yt=1e9,
        Yc=1e9,
        S12=1e9,
        Zt=50e6,
        S13=40e6,
        S23=40e6,
    )


def test_single_edge_strip_matches_legacy_interlaminar() -> None:
    lam = LaminateDefinition(plies=[(_ud_ply(), 0.0)])
    pts = np.array([[0.0, 0.0], [0.12, 0.0]], dtype=np.float64)
    sub = SubcomponentGeometry(
        name="strip",
        midsurface_coords=pts,
        material=lam,
        thickness=lam.total_thickness(),
        strip_width_m=0.04,
    )
    sec = SectionDefinition(station_z=0.0, subcomponents=[sub])
    res = MidsurfaceSectionSolver().solve_one(sec)
    mesh = build_line_mesh(sec, 1e-6)
    fe = build_strip_fe_data(sec, mesh)
    eiy = float(res.K6[1, 1])
    eiz = float(res.K6[2, 2])
    vy, vz = 0.3, -0.7
    leg = recover_interlaminar([0], [lam], vy, vz, eiy, eiz)
    strip, _summ = recover_interlaminar_strip_equilibrium(
        [0], [lam], mesh, fe, sec, res, vy, vz, eiy, eiz
    )
    assert np.isclose(leg.IFI_global, strip.IFI_global, rtol=0.0, atol=1e-9)
    for a, b in zip(leg.edge_results[0].interfaces, strip.edge_results[0].interfaces):
        assert np.isclose(a.sigma_13, b.sigma_13, rtol=0.0, atol=1e-9)
        assert np.isclose(a.sigma_23, b.sigma_23, rtol=0.0, atol=1e-9)
