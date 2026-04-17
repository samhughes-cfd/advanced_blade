from __future__ import annotations

import numpy as np

from section_model.engine.implicit_section_geometry import GeometryConstraintSpec, StationFrame2D, build_constrained_geometry
from section_model.engine.materials import IsotropicMaterial


def _mat(name: str) -> IsotropicMaterial:
    return IsotropicMaterial(name=name, E=70e9, nu=0.33, rho=2700.0, sigma_allow=250e6)


def test_frame_round_trip_points() -> None:
    f = StationFrame2D(twist_rad=0.4)
    pts = np.array([[1.0, 0.0], [0.3, -0.7]], dtype=np.float64)
    back = f.points_b_to_s(f.points_s_to_b(pts))
    np.testing.assert_allclose(back, pts, atol=1e-12)


def test_webs_are_flapwise_parallel_in_b_frame() -> None:
    outer = np.array([[-1.0, -0.6], [1.0, -0.6], [1.0, 0.6], [-1.0, 0.6]], dtype=np.float64)
    mats = {"skin": _mat("skin"), "web_left": _mat("wl"), "web_right": _mat("wr"), "spar_cap": _mat("cap")}
    spec = GeometryConstraintSpec(
        skin_outer_boundary_s=outer,
        skin_thickness=0.05,
        web_width=0.02,
        web_stations_s=(0.2, 0.8),
        spar_cap_width=0.3,
        spar_cap_thickness=0.02,
        twist_rad=0.6,
        station_z=0.0,
        materials=mats,
    )
    g = build_constrained_geometry(spec)
    wl_b = g.frame.points_s_to_b(g.web_left_s)
    wr_b = g.frame.points_s_to_b(g.web_right_s)
    d0 = wl_b[1] - wl_b[0]
    d1 = wr_b[1] - wr_b[0]
    d0 = d0 / (np.linalg.norm(d0) + 1e-30)
    d1 = d1 / (np.linalg.norm(d1) + 1e-30)
    flap = np.array([0.0, 1.0], dtype=np.float64)
    assert abs(float(np.dot(d0, flap))) > 0.99
    assert abs(float(np.dot(d1, flap))) > 0.99

