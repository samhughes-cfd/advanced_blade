from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import GeometryConstraintSpec, StationFrame2D


def _resample_closed(poly: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    p = np.asarray(poly, dtype=np.float64)
    if np.linalg.norm(p[0] - p[-1]) > 1e-12:
        p = np.vstack([p, p[0]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        raise ValueError("Degenerate boundary polyline.")
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    y = np.interp(targets, s, p[:, 0])
    z = np.interp(targets, s, p[:, 1])
    return np.stack([y, z], axis=1)


def _polygon_orientation(poly: NDArray[np.float64]) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def offset_inner_boundary(
    outer_boundary_s: NDArray[np.float64],
    thickness: float,
    n_samples: int = 256,
) -> NDArray[np.float64]:
    outer = _resample_closed(outer_boundary_s, n_samples)
    tang = np.roll(outer, -1, axis=0) - np.roll(outer, 1, axis=0)
    # Interior normal depends on orientation.
    if _polygon_orientation(outer) > 0.0:
        n = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    else:
        n = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    return outer + float(thickness) * n


def _line_polygon_intersections(
    p0: NDArray[np.float64],
    direction: NDArray[np.float64],
    poly: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    out: list[NDArray[np.float64]] = []
    d = direction
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        e = b - a
        mat = np.array([[d[0], -e[0]], [d[1], -e[1]]], dtype=np.float64)
        rhs = a - p0
        det = np.linalg.det(mat)
        if abs(det) < 1e-12:
            continue
        t, u = np.linalg.solve(mat, rhs)
        if 0.0 <= u <= 1.0:
            out.append(p0 + t * d)
    return out


@dataclass(frozen=True)
class ConstrainedGeometry:
    skin_outer_s: NDArray[np.float64]
    skin_inner_s: NDArray[np.float64]
    web_left_s: NDArray[np.float64]
    web_right_s: NDArray[np.float64]
    spar_cap_s: NDArray[np.float64]
    frame: StationFrame2D


def build_constrained_geometry(spec: GeometryConstraintSpec) -> ConstrainedGeometry:
    outer = _resample_closed(spec.skin_outer_boundary_s, int(spec.n_samples))
    inner_t = float(spec.skin_thickness)
    if spec.thickness_field is not None:
        inner_t = float(np.mean([spec.thickness_field(float(s)) for s in np.linspace(0.0, 1.0, 21)]))
    inner = offset_inner_boundary(outer, inner_t, n_samples=int(spec.n_samples))
    frame = StationFrame2D(twist_rad=float(spec.twist_rad))

    # B-frame flapwise axis is [0,1], mapped to S for geometric intersection.
    web_dir_s = frame.direction_b_to_s(np.array([0.0, 1.0], dtype=np.float64))
    web_dir_s = web_dir_s / (np.linalg.norm(web_dir_s) + 1e-30)

    n = outer.shape[0]
    i0 = int(np.clip(round(spec.web_stations_s[0] * (n - 1)), 0, n - 1))
    i1 = int(np.clip(round(spec.web_stations_s[1] * (n - 1)), 0, n - 1))
    a0 = outer[i0]
    a1 = outer[i1]

    ints0 = _line_polygon_intersections(a0, web_dir_s, inner)
    ints1 = _line_polygon_intersections(a1, web_dir_s, inner)
    if not ints0 or not ints1:
        raise ValueError("Could not intersect web guide lines with inner skin.")

    b0 = min(ints0, key=lambda p: float(np.linalg.norm(p - a0)))
    b1 = min(ints1, key=lambda p: float(np.linalg.norm(p - a1)))
    web_left = np.vstack([a0, b0])
    web_right = np.vstack([a1, b1])

    # Spar cap constrained between webs.
    c_mid = 0.5 * (a0 + a1)
    span_vec = a1 - a0
    span_norm = float(np.linalg.norm(span_vec))
    if span_norm <= 1e-12:
        raise ValueError("Web stations collapsed; cannot place spar cap.")
    span_hat = span_vec / span_norm
    cap_half = 0.5 * min(float(spec.spar_cap_width), 0.95 * span_norm)
    cap = np.vstack([c_mid - cap_half * span_hat, c_mid + cap_half * span_hat])
    return ConstrainedGeometry(
        skin_outer_s=outer,
        skin_inner_s=inner,
        web_left_s=web_left,
        web_right_s=web_right,
        spar_cap_s=cap,
        frame=frame,
    )

