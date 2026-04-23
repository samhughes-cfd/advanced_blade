from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import MedialAxisDiagnostics, MidlineExtractionResult, SDFField


@dataclass(frozen=True)
class GridSpec:
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    ny: int = 192
    nz: int = 192


def sample_grid(field: SDFField, grid: GridSpec) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y = np.linspace(grid.y_min, grid.y_max, grid.ny)
    z = np.linspace(grid.z_min, grid.z_max, grid.nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    pts = np.stack([yy.ravel(), zz.ravel()], axis=1)
    phi = field.eval(pts).reshape(grid.ny, grid.nz)
    return y, z, phi


def extract_zero_contour_polyline(field: SDFField, grid: GridSpec) -> NDArray[np.float64]:
    """Lightweight contour approximation via nearest-to-zero samples."""
    y, z, phi = sample_grid(field, grid)
    idx = np.argwhere(np.abs(phi) <= max((grid.y_max - grid.y_min) / grid.ny, 1e-5))
    if idx.shape[0] < 4:
        raise ValueError("Unable to extract contour from SDF field.")
    pts = np.stack([y[idx[:, 0]], z[idx[:, 1]]], axis=1)
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    poly = pts[order]
    return poly


def _arc_normals(poly: NDArray[np.float64]) -> NDArray[np.float64]:
    tang = np.roll(poly, -1, axis=0) - np.roll(poly, 1, axis=0)
    n = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return n / nn


def extract_midline_from_offset_boundaries(
    outer_boundary: NDArray[np.float64],
    inner_boundary: NDArray[np.float64],
    strip_width_m: float,
) -> MidlineExtractionResult:
    n = min(outer_boundary.shape[0], inner_boundary.shape[0])
    if n < 4:
        raise ValueError("Need at least 4 points on boundaries for midline extraction.")
    outer = outer_boundary[:n]
    inner = inner_boundary[:n]
    mid = 0.5 * (outer + inner)
    seg = np.linalg.norm(np.diff(mid, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-8])
    mid = mid[keep]
    if mid.shape[0] < 2:
        raise ValueError("Midline extraction collapsed to fewer than 2 points.")
    thick = np.linalg.norm(outer - inner, axis=1)
    rms = float(np.sqrt(np.mean((thick - np.mean(thick)) ** 2)))
    if not np.all(np.isfinite(mid)):
        raise ValueError("Midline extraction produced non-finite coordinates.")
    if float(np.min(thick)) <= 0.0:
        raise ValueError("Non-positive local thickness encountered in boundary pair.")
    diag = MedialAxisDiagnostics(
        branch_count=1,
        spur_count=0,
        disconnected_components=0,
        thickness_residual_rms=rms,
        self_intersections=0,
        notes=[],
    )
    return MidlineExtractionResult(midsurface_coords_s=mid, strip_width_m=float(max(strip_width_m, 1e-9)), diagnostics=diag)

