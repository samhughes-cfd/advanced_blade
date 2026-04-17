"""Local orthotropic panel buckling (section_properties), not GBT."""

import dataclasses

import numpy as np

from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import OrthotropicPly
from blade_precompute.section_properties.engine.mesh import build_line_mesh
from blade_precompute.section_properties.engine.elements import build_strip_fe_data
from blade_precompute.section_properties.engine.panel_buckling import (
    PanelBucklingSectionResult,
    _Nx_cr,
    _Nxy_cr,
    assess_panel_buckling_section,
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


def test_nx_cr_decreases_when_panel_longer() -> None:
    d11, d12, d22, d66 = 120.0, 8.0, 90.0, 35.0
    b = 0.08
    n_short, _ = _Nx_cr(d11, d12, d22, d66, a=0.4, b=b)
    n_long, _ = _Nx_cr(d11, d12, d22, d66, a=0.8, b=b)
    assert n_long < n_short
    assert 2.5 < n_short / n_long < 6.5


def test_nxy_cr_positive_finite() -> None:
    d11, d12, d22, d66 = 120.0, 8.0, 90.0, 35.0
    nxy = _Nxy_cr(d11, d12, d22, d66, a=0.5, b=0.06)
    assert np.isfinite(nxy) and nxy > 0.0


def test_assess_panel_section_aggregation() -> None:
    lam = LaminateDefinition(plies=[(_ud_ply(), 0.0)])
    pts = np.array([[0.0, 0.0], [0.15, 0.0]], dtype=np.float64)
    sub = SubcomponentGeometry(
        name="skin",
        midsurface_coords=pts,
        material=lam,
        thickness=lam.total_thickness(),
        strip_width_m=0.05,
    )
    sec = SectionDefinition(station_z=0.0, subcomponents=[sub])
    mesh = build_line_mesh(sec, 1e-6)
    fe = build_strip_fe_data(sec, mesh)
    bk0 = assess_panel_buckling_section(
        fe,
        [0],
        [lam],
        sigma_zz=np.array([0.0]),
        tau=np.array([0.0]),
        frame_spacing_m=0.4,
        sigma_yy=np.array([0.0]),
    )
    assert isinstance(bk0, PanelBucklingSectionResult)
    assert bk0.BI_max == 0.0 and bk0.n_buckled == 0

    bk1 = assess_panel_buckling_section(
        fe,
        [0],
        [lam],
        sigma_zz=np.array([500e6]),
        tau=np.array([120e6]),
        frame_spacing_m=0.15,
        sigma_yy=np.array([80e6]),
    )
    assert bk1.BI_max >= bk0.BI_max
    assert len(bk1.edge_results) == 1
    er = bk1.edge_results[0]
    d = dataclasses.asdict(er)
    assert "R_cy" in d and "N_y_applied" in d
