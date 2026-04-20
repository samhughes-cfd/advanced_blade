"""
Sectorial coordinates and shear-centre from thin-wall sectorial static moments.

Shear centre (relative to modulus-weighted centroid) uses ω with pole at the
centroid (y_c, z_c):

  Iωy = ∫ ω (y - y_c) t ds ,   Iωz = ∫ ω (z - z_c) t ds

  e_y = (Iωz * Iyy - Iωy * Iyz) / D
  e_z = (Iωy * Izz - Iωz * Iyz) / D ,   D = Iyy Izz - Iyz²

  y_sc = y_c + e_y ,  z_sc = z_c + e_z

Straight segment increment (pole P):

  Δω = (y0 - y_p)(z1 - z0) - (z0 - z_p)(y1 - y0)

Closed outer contour: unique vertices (no repeated first point); all n edges
including the closing edge (v_{n-1} → v_0) enter the sectorial integrals.

ω has dimensions [L²]. Integrals Iωy are [L⁵] when t, ds are [L]; consistent with e [L].
"""

from __future__ import annotations

import numpy as np


def omega_increment(y0: float, z0: float, y1: float, z1: float, yp: float, zp: float) -> float:
    """Increment of sectorial coordinate along a straight segment A→B w.r.t. pole P."""
    return (y0 - yp) * (z1 - z0) - (z0 - zp) * (y1 - y0)


def closed_loop_from_airfoil(airfoil: np.ndarray) -> np.ndarray:
    """
    Closed outer polygon without duplicate first/last point.

    Expects ``airfoil = [upper; lower]`` with **both** halves ordered **LE→TE** at
    matching chord stations (``multi_cell_blade_section.naca_four_digit``). The
    lower half is reversed to TE→LE before concatenation so the loop runs along
    the outer boundary.
    """
    n = len(airfoil) // 2
    upper = airfoil[:n]
    lower = airfoil[n:][::-1]
    return np.vstack([upper, lower])


def open_outline_from_airfoil(airfoil: np.ndarray) -> np.ndarray:
    """
    Open midline for sectorial shear-centre (open thin-wall theory).

    Upper LE→TE, then the reversed lower (TE→LE) **without** repeating the TE
    vertex: ``upper ∪ lower_rev[1:]``.
    """
    n = len(airfoil) // 2
    upper = airfoil[:n]
    lower = airfoil[n:][::-1]
    return np.vstack([upper, lower[1:]])


def omega_vertices_open_chain(verts: np.ndarray, yp: float, zp: float) -> np.ndarray:
    """ω at vertices along path v0→v1→…→v_{n-1} with ω[0]=0 (pole P)."""
    n = len(verts)
    omega = np.zeros(n, dtype=float)
    for i in range(n - 1):
        omega[i + 1] = omega[i] + omega_increment(
            verts[i, 0], verts[i, 1], verts[i + 1, 0], verts[i + 1, 1], yp, zp
        )
    return omega


def sectorial_static_moments_open(
    verts: np.ndarray,
    omega_v: np.ndarray,
    y_c: float,
    z_c: float,
    t: float,
) -> tuple[float, float]:
    """Open polyline only: edges 0→1 … →(n-2)→(n-1)."""
    n = len(verts)
    Iy = 0.0
    Iz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        om_m = 0.5 * (omega_v[i] + omega_v[i + 1])
        ym = 0.5 * (y0 + y1) - y_c
        zm = 0.5 * (z0 + z1) - z_c
        ds = float(np.hypot(y1 - y0, z1 - z0))
        Iy += om_m * ym * t * ds
        Iz += om_m * zm * t * ds
    return Iy, Iz


def sectorial_static_moments_closed(
    verts: np.ndarray,
    omega_v: np.ndarray,
    y_c: float,
    z_c: float,
    yp: float,
    zp: float,
    t: float,
) -> tuple[float, float]:
    """
    Iωy = ∫ ω (y - y_c) t ds, Iωz = ∫ ω (z - z_c) t ds on all boundary edges,
    including the closing edge v_{n-1} → v_0. Mid-edge ω is used on each segment.
    """
    n = len(verts)
    Iy = 0.0
    Iz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        om_m = 0.5 * (omega_v[i] + omega_v[i + 1])
        ym = 0.5 * (y0 + y1) - y_c
        zm = 0.5 * (z0 + z1) - z_c
        ds = float(np.hypot(y1 - y0, z1 - z0))
        Iy += om_m * ym * t * ds
        Iz += om_m * zm * t * ds
    y0, z0 = verts[n - 1]
    y1, z1 = verts[0]
    dom = omega_increment(y0, z0, y1, z1, yp, zp)
    om_m = omega_v[n - 1] + 0.5 * dom
    ym = 0.5 * (y0 + y1) - y_c
    zm = 0.5 * (z0 + z1) - z_c
    ds = float(np.hypot(y1 - y0, z1 - z0))
    Iy += om_m * ym * t * ds
    Iz += om_m * zm * t * ds
    return Iy, Iz


def shear_center_from_sectorial(
    verts: np.ndarray,
    props_yz: tuple[float, float, float, float, float],
    t_skin: float,
    *,
    closed: bool = False,
) -> tuple[float, float, np.ndarray]:
    """
    Shear centre using supplied (y_c, z_c, Iyy, Izz, Iyz) matching ``verts``.

    For **open** sectorial theory (cut thin wall), pass ``closed=False`` and an open
    polyline. For a closed polygon use ``closed=True`` and the closing edge in ω.
    """
    y_c, z_c, Iyy, Izz, Iyz = props_yz
    omega_v = omega_vertices_open_chain(verts, y_c, z_c)
    if closed:
        Iwy, Iwz = sectorial_static_moments_closed(
            verts, omega_v, y_c, z_c, y_c, z_c, t_skin
        )
    else:
        Iwy, Iwz = sectorial_static_moments_open(verts, omega_v, y_c, z_c, t_skin)
    D = Iyy * Izz - Iyz**2
    if abs(D) < 1e-40:
        D = 1e-40
    e_y = (Iwz * Iyy - Iwy * Iyz) / D
    e_z = (Iwy * Izz - Iwz * Iyz) / D
    return float(y_c + e_y), float(z_c + e_z), omega_v


def modulus_weighted_open_line(
    verts: np.ndarray,
    t: float,
    e_n: float,
) -> tuple[float, float, float, float, float, float]:
    """Line properties along an **open** polyline (no closing edge)."""
    n = len(verts)
    EA = 0.0
    EAy = 0.0
    EAz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        w = e_n * t * ds
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        EA += w
        EAy += w * ym
        EAz += w * zm
    if EA < 1e-30:
        EA = 1e-30
    y_c = EAy / EA
    z_c = EAz / EA
    Iyy = 0.0
    Izz = 0.0
    Iyz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        w = e_n * t * ds
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        yn = ym - y_c
        zn = zm - z_c
        Iyy += w * zn**2
        Izz += w * yn**2
        Iyz += w * yn * zn
    return float(y_c), float(z_c), float(EA), float(Iyy), float(Izz), float(Iyz)


def shear_center_outer_skin_loop(
    verts_open: np.ndarray,
    t_skin: float,
    e_n: float,
) -> tuple[float, float, np.ndarray, tuple[float, float, float, float, float]]:
    """
    Shear centre from **open** sectorial theory on ``verts_open`` (e.g. from
    ``open_outline_from_airfoil``), with matching line inertia on the same path.
    """
    y_c, z_c, _ea, Iyy, Izz, Iyz = modulus_weighted_open_line(verts_open, t_skin, e_n)
    y_sc, z_sc, omega_v = shear_center_from_sectorial(
        verts_open, (y_c, z_c, Iyy, Izz, Iyz), t_skin, closed=False
    )
    return y_sc, z_sc, omega_v, (y_c, z_c, Iyy, Izz, Iyz)


def _edge_integrals_weighted_open(
    verts: np.ndarray,
    omega_v: np.ndarray,
    t: float,
    weight_fn,
) -> float:
    """∫ weight_fn(ω_mid, y_mid, z_mid) t ds on open polyline edges only."""
    n = len(verts)
    acc = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        om_m = 0.5 * (omega_v[i] + omega_v[i + 1])
        ds = float(np.hypot(y1 - y0, z1 - z0))
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        acc += weight_fn(om_m, ym, zm) * t * ds
    return float(acc)


def normalized_warping(
    verts: np.ndarray,
    y_sc: float,
    z_sc: float,
    _y_c: float,
    _z_c: float,
    t: float,
) -> tuple[np.ndarray, float]:
    """
    Sectorial ω with pole at shear centre; subtract thickness-weighted mean along edges.

    Returns ω_hat at vertices (open chain) and perimeter length ∫ ds.
    ``_y_c``, ``_z_c`` reserved for API compatibility (centroid of outline / section).
    """
    omega_v = omega_vertices_open_chain(verts, y_sc, z_sc)

    def kern(om, ym, zm):
        return om

    num = _edge_integrals_weighted_open(verts, omega_v, t, kern)
    den = 0.0
    n = len(verts)
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        den += t * float(np.hypot(y1 - y0, z1 - z0))
    if den < 1e-30:
        den = 1e-30
    om_mean = num / den
    omega_hat = omega_v - om_mean
    return omega_hat, den


def warping_constant_I_omega(
    verts: np.ndarray,
    omega_hat: np.ndarray,
    t: float,
    e_ref: float = 1.0,
) -> float:
    """I_ω = ∫ E_n ω_hat² t ds with E_n = 1; open polyline edges only."""
    n = len(verts)
    acc = 0.0
    for i in range(n - 1):
        om_m = 0.5 * (omega_hat[i] + omega_hat[i + 1])
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        acc += e_ref * (om_m**2) * t * ds
    return float(acc)


def modulus_weighted_outline_props(
    verts: np.ndarray,
    t: float,
    e_n: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Thin-wall line properties for a closed polygon (unique vertices, closing edge implied).

    Returns (y_c, z_c, EA_line, Iyy, Izz, Iyz) with I about centroid, using
    ∫ E_n t ds, ∫ E_n y z^n t ds on straight segments (trapezoidal rule on edges).
    """
    n = len(verts)
    EA = 0.0
    EAy = 0.0
    EAz = 0.0
    for i in range(n):
        y0, z0 = verts[i]
        y1, z1 = verts[(i + 1) % n]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        w = e_n * t * ds
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        EA += w
        EAy += w * ym
        EAz += w * zm
    if EA < 1e-30:
        EA = 1e-30
    y_c = EAy / EA
    z_c = EAz / EA
    Iyy = 0.0
    Izz = 0.0
    Iyz = 0.0
    for i in range(n):
        y0, z0 = verts[i]
        y1, z1 = verts[(i + 1) % n]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        w = e_n * t * ds
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        yn = ym - y_c
        zn = zm - z_c
        Iyy += w * zn**2
        Izz += w * yn**2
        Iyz += w * yn * zn
    return float(y_c), float(z_c), float(EA), float(Iyy), float(Izz), float(Iyz)


def polygon_area_signed(poly: np.ndarray) -> float:
    """Signed area (CCW positive) for closed polyline (first point may equal last)."""
    y = poly[:, 0]
    z = poly[:, 1]
    if np.allclose(y[0], y[-1]) and np.allclose(z[0], z[-1]):
        y, z = y[:-1], z[:-1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))
