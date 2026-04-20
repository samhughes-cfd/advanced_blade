"""
Secondary (warping) shear flow from longitudinal equilibrium with ∂σ_ω/∂x.

For σ_ω = B(x) ω̂(s) / I_ω, the wall equilibrium ∂q/∂s ≈ −t ∂σ_ω/∂x gives

    ∂q/∂s = −(t / I_ω) (dB/dx) ω̂(s)

on each straight segment (thin-wall, σ_ω constant through t). This module
integrates the **open-chain** particular solution starting at q=0 at the first
outline vertex; multi-cell **closure** is applied in ``multi_cell_blade_section``
via the same Bredt system used for primary shear flow.
"""

from __future__ import annotations

import numpy as np


def interp_scalar_open_polyline(
    verts: np.ndarray,
    scalar_vertex: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of scalar data on an open polyline (no wrap)."""
    n = len(verts)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    out = np.zeros(len(pts), dtype=float)
    for iq, q in enumerate(pts):
        dmin = 1e300
        best = 0.0
        for i in range(n - 1):
            p0 = verts[i]
            p1 = verts[i + 1]
            e = p1 - p0
            el2 = float(np.dot(e, e))
            if el2 < 1e-30:
                continue
            tpar = float(np.dot(q - p0, e) / el2)
            tpar = max(0.0, min(1.0, tpar))
            proj = p0 + tpar * e
            d = float(np.linalg.norm(q - proj))
            if d < dmin:
                dmin = d
                best = (1.0 - tpar) * scalar_vertex[i] + tpar * scalar_vertex[i + 1]
        out[iq] = best
    return out


def q_omega_secondary_open_vertices(
    verts: np.ndarray,
    omega_hat: np.ndarray,
    t: float,
    I_omega: float,
    dB_dx: float,
) -> np.ndarray:
    """
    Particular secondary shear flow at outline vertices with q[0] = 0 (open cut).

    Parameters
    ----------
    verts, omega_hat
        Same open chain as ``normalized_warping`` (aligned lengths).
    t
        Skin thickness [m] used in σ_ω / I_ω definitions (same as warping path).
    I_omega
        Warping constant ∫ E_n ω̂² t ds (code uses uniform E_n on outline).
    dB_dx
        Bimoment gradient [N·m] — if zero, returns zeros.
    """
    verts = np.asarray(verts, dtype=float)
    omega_hat = np.asarray(omega_hat, dtype=float).ravel()
    n = len(verts)
    q = np.zeros(n, dtype=float)
    if n < 2 or abs(dB_dx) < 1e-300:
        return q
    I_eff = float(I_omega) if abs(I_omega) >= 1e-40 else (1e-40 if I_omega >= 0 else -1e-40)
    factor = dB_dx / I_eff
    for i in range(n - 1):
        y0, z0 = verts[i, 0], verts[i, 1]
        y1, z1 = verts[i + 1, 0], verts[i + 1, 1]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        om_m = 0.5 * (omega_hat[i] + omega_hat[i + 1])
        dq = -factor * t * om_m * ds
        q[i + 1] = q[i] + dq
    return q


def q_omega_secondary_panels_particular(
    loop_open: np.ndarray,
    omega_hat_vert: np.ndarray,
    q_open_vert: np.ndarray,
    panels: list,
    dB_dx: float,
    I_omega: float,
) -> list[np.ndarray]:
    """
    Particular secondary shear q(s) on each panel: skins from outline interpolation;
    webs by integrating along the web with ω̂ linear between endpoint values.
    """
    if abs(dB_dx) < 1e-300:
        return [np.zeros(len(p.nodes), dtype=float) for p in panels]

    I_eff = float(I_omega) if abs(I_omega) >= 1e-40 else (1e-40 if I_omega >= 0 else -1e-40)
    factor = dB_dx / I_eff
    out: list[np.ndarray] = []

    for p in panels:
        nodes = np.asarray(p.nodes, dtype=float)
        npt = len(nodes)
        if npt < 2:
            out.append(np.zeros(npt, dtype=float))
            continue

        if "Web" in p.label:
            om_end = interp_scalar_open_polyline(
                loop_open, omega_hat_vert, np.vstack([nodes[0], nodes[-1]])
            )
            om0, om1 = float(om_end[0]), float(om_end[1])
            q_arr = np.zeros(npt, dtype=float)
            q_arr[0] = float(
                interp_scalar_open_polyline(loop_open, q_open_vert, nodes[0:1])[0]
            )
            t_w = float(p.lam.t)
            for j in range(npt - 1):
                a0 = j / max(npt - 1, 1)
                a1 = (j + 1) / max(npt - 1, 1)
                om_a = (1.0 - a0) * om0 + a0 * om1
                om_b = (1.0 - a1) * om0 + a1 * om1
                om_m = 0.5 * (om_a + om_b)
                dy = nodes[j + 1, 0] - nodes[j, 0]
                dz = nodes[j + 1, 1] - nodes[j, 1]
                ds = float(np.hypot(dy, dz))
                dq = -factor * t_w * om_m * ds
                q_arr[j + 1] = q_arr[j] + dq
            out.append(q_arr)
        else:
            q_arr = interp_scalar_open_polyline(loop_open, q_open_vert, nodes)
            out.append(q_arr)

    return out
