from __future__ import annotations

import numpy as np

from section_model.engine.implicit_section_geometry import GeometryConstraintSpec, SDFSectionSolver, build_section_from_constraints
from section_model.engine.materials import IsotropicMaterial
from section_model.engine.solver import MidsurfaceSectionSolver


def _mat(name: str) -> IsotropicMaterial:
    return IsotropicMaterial(name=name, E=70e9, nu=0.33, rho=2700.0, sigma_allow=250e6)


def test_pipeline_builds_section_and_solves() -> None:
    outer = np.array([[-1.0, -0.6], [1.0, -0.6], [1.0, 0.6], [-1.0, 0.6]], dtype=np.float64)
    mats = {"skin": _mat("skin"), "web_left": _mat("wl"), "web_right": _mat("wr"), "spar_cap": _mat("cap")}
    spec = GeometryConstraintSpec(
        skin_outer_boundary_s=outer,
        skin_thickness=0.04,
        web_width=0.02,
        web_stations_s=(0.2, 0.8),
        spar_cap_width=0.35,
        spar_cap_thickness=0.03,
        twist_rad=0.2,
        station_z=0.0,
        materials=mats,
    )
    built = build_section_from_constraints(spec)
    assert len(built.section.subcomponents) == 4
    res = MidsurfaceSectionSolver().solve_one(built.section)
    assert np.all(np.isfinite(res.K6))
    assert res.K7.shape == (7, 7)


def test_sdf_solver_matches_midsurface_delegate() -> None:
    outer = np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]], dtype=np.float64)
    mats = {"skin": _mat("skin"), "web_left": _mat("wl"), "web_right": _mat("wr"), "spar_cap": _mat("cap")}
    spec = GeometryConstraintSpec(
        skin_outer_boundary_s=outer,
        skin_thickness=0.03,
        web_width=0.02,
        web_stations_s=(0.25, 0.75),
        spar_cap_width=0.3,
        spar_cap_thickness=0.02,
        twist_rad=0.1,
        station_z=0.0,
        materials=mats,
    )
    built = build_section_from_constraints(spec)
    a = MidsurfaceSectionSolver().solve_one(built.section)
    b = SDFSectionSolver().solve_one(spec)
    np.testing.assert_allclose(a.K7, b.K7, rtol=1e-9, atol=1e-9)

