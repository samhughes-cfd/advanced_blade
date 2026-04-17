"""Warping solve: pinned Laplacian yields finite omega and positive K_ww."""
import numpy as np

from section_model.engine.geometry import SectionDefinition, SubcomponentGeometry
from section_model.engine.laminate import LaminateDefinition
from section_model.engine.materials import OrthotropicPly
from section_model.engine.solver import MidsurfaceSectionSolver


def _ud_ply():
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


def test_warping_open_section():
    lam = LaminateDefinition(plies=[(_ud_ply(), 0.0)])
    pts = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.05], [0.0, 0.05], [0.0, 0.0]])
    sub = SubcomponentGeometry(
        name="frame",
        midsurface_coords=pts,
        material=lam,
        thickness=lam.total_thickness(),
        strip_width_m=0.04,
    )
    sec = SectionDefinition(station_z=0.0, subcomponents=[sub])
    res = MidsurfaceSectionSolver().solve_one(sec)
    assert res.warping_function.shape[0] > 0
    assert np.all(np.isfinite(res.warping_function))
    assert res.K_ww >= 0.0
    assert res.K7.shape == (7, 7)
