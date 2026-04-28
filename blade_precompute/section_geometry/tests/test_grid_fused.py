"""Regression tests for fused SDFGrid section properties."""

import numpy as np

from blade_precompute.section_geometry.geometry.grid import SDFGrid


def _legacy_properties(grid: SDFGrid, phi):
    mask = (phi < 0.0).astype(float)
    A_cells = mask.sum()
    if A_cells == 0:
        return {
            "area": 0.0,
            "cx": float("nan"),
            "cy": float("nan"),
            "Ixx": float("nan"),
            "Iyy": float("nan"),
            "Ixy": float("nan"),
        }
    cx = float((mask * grid.X).sum() / A_cells)
    cy = float((mask * grid.Y).sum() / A_cells)
    dA = grid.dx * grid.dy
    xr = grid.X - cx
    yr = grid.Y - cy
    Ixx = float((mask * yr**2).sum() * dA)
    Iyy = float((mask * xr**2).sum() * dA)
    Ixy = float((mask * xr * yr).sum() * dA)
    return {
        "area": float(A_cells) * dA,
        "cx": cx,
        "cy": cy,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Ixy": Ixy,
    }


def test_fused_section_properties_matches_legacy_formula():
    grid = SDFGrid.from_bbox(-1.0, 1.0, -1.0, 1.0, nx=120, ny=100)
    phi = np.sqrt((grid.X - 0.2) ** 2 + (grid.Y + 0.1) ** 2) - 0.55
    fused = grid.section_properties_fused(phi)
    legacy = _legacy_properties(grid, phi)
    for key in ("area", "cx", "cy", "Ixx", "Iyy", "Ixy"):
        np.testing.assert_allclose(fused[key], legacy[key], rtol=0.0, atol=1e-12)

