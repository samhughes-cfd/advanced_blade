from __future__ import annotations

import numpy as np

from section_model.engine.implicit_section_geometry import GeometryConstraintSpec, build_constrained_geometry
from section_model.engine.materials import IsotropicMaterial


def _mat(name: str) -> IsotropicMaterial:
    return IsotropicMaterial(name=name, E=70e9, nu=0.33, rho=2700.0, sigma_allow=250e6)


def test_inner_boundary_moves_inward_with_skin_thickness() -> None:
    outer = np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]], dtype=np.float64)
    mats = {"skin": _mat("skin"), "web_left": _mat("wl"), "web_right": _mat("wr"), "spar_cap": _mat("cap")}
    s1 = GeometryConstraintSpec(
        skin_outer_boundary_s=outer,
        skin_thickness=0.05,
        web_width=0.02,
        web_stations_s=(0.2, 0.8),
        spar_cap_width=0.3,
        spar_cap_thickness=0.02,
        twist_rad=0.0,
        station_z=0.0,
        materials=mats,
    )
    s2 = GeometryConstraintSpec(**{**s1.__dict__, "skin_thickness": 0.1})
    g1 = build_constrained_geometry(s1)
    g2 = build_constrained_geometry(s2)
    assert float(np.mean(np.linalg.norm(g2.skin_outer_s - g2.skin_inner_s, axis=1))) > float(
        np.mean(np.linalg.norm(g1.skin_outer_s - g1.skin_inner_s, axis=1))
    )

