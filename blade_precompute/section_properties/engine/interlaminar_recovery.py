"""
Tier 3 equilibrium-based interlaminar stress recovery (optional, composite).

When ``n_s < 2``, spanwise gradients are undefined; returns zeros (documented).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interlaminar_stress_recovery(
    sigma_inplane: NDArray[np.float64],
    z_stations: NDArray[np.float64],
    z_ply: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Recover ``[τ13, τ23, σ33]`` at ply interfaces.

    In-plane stress gradients use :func:`numpy.gradient` along span (axis 0).
    Through-thickness integration uses uniform ply thickness inferred from
    ``z_ply`` span.

    Parameters
    ----------
    sigma_inplane
        ``(n_s, n_comp, n_ply, 3)`` — ``[σ11, σ22, τ12]``.
    z_stations
        ``(n_s,)`` spanwise positions [m].
    z_ply
        ``(n_comp, n_ply)`` ply mid-ordinates [m].

    Returns
    -------
    NDArray
        ``(n_s, n_comp, n_ply + 1, 3)`` interface values bottom → top.
    """
    n_s, n_comp, n_ply, _ = sigma_inplane.shape
    n_if = n_ply + 1
    out = np.zeros((n_s, n_comp, n_if, 3), dtype=np.float64)
    if n_s < 2 or n_ply == 0:
        return out

    d11 = np.gradient(sigma_inplane[..., 0], z_stations, axis=0)
    d22 = np.gradient(sigma_inplane[..., 1], z_stations, axis=0)
    d12 = np.gradient(sigma_inplane[..., 2], z_stations, axis=0)

    for c in range(n_comp):
        zp = z_ply[c, :n_ply]
        zmin, zmax = float(np.min(zp)), float(np.max(zp))
        dz = (zmax - zmin) / max(n_ply, 1)
        if dz <= 0:
            dz = 1e-6
        for s in range(n_s):
            acc13 = 0.0
            acc23 = 0.0
            out[s, c, 0, :] = 0.0
            for j in range(1, n_if):
                k = j - 1
                kp = min(k, n_ply - 1)
                g13 = -(d11[s, c, kp] + d12[s, c, kp])
                g23 = -(d12[s, c, kp] + d22[s, c, kp])
                acc13 += g13 * dz
                acc23 += g23 * dz
                out[s, c, j, 0] = acc13
                out[s, c, j, 1] = acc23
                out[s, c, j, 2] = 0.0
    return out


def delamination_fi(
    sigma_interlaminar: NDArray[np.float64],
    Zt: NDArray[np.float64],
    S13: NDArray[np.float64],
    S23: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Delamination failure index at interfaces.

    ``⟨σ33⟩ = max(σ33, 0)`` (Macaulay bracket — tensile only).
    Strength arrays ``(n_comp, n_ply)`` are averaged to interface columns.
    """
    tau13 = sigma_interlaminar[..., 0]
    tau23 = sigma_interlaminar[..., 1]
    s33 = np.maximum(sigma_interlaminar[..., 2], 0.0)
    n_if = sigma_interlaminar.shape[-2]
    n_p = Zt.shape[1]
    Zi = np.zeros((Zt.shape[0], n_if), dtype=np.float64)
    S13i = np.zeros_like(Zi)
    S23i = np.zeros_like(Zi)
    Zi[:, 0] = Zt[:, 0]
    Zi[:, -1] = Zt[:, min(n_p - 1, 0)]
    if n_if > 2 and n_p > 1:
        Zi[:, 1:-1] = 0.5 * (Zt[:, :-1] + Zt[:, 1:])
        S13i[:, 1:-1] = 0.5 * (S13[:, :-1] + S13[:, 1:])
        S23i[:, 1:-1] = 0.5 * (S23[:, :-1] + S23[:, 1:])
    S13i[:, 0] = S13[:, 0]
    S13i[:, -1] = S13[:, min(n_p - 1, 0)]
    S23i[:, 0] = S23[:, 0]
    S23i[:, -1] = S23[:, min(n_p - 1, 0)]
    Zi = np.maximum(Zi, 1e-18)
    S13i = np.maximum(S13i, 1e-18)
    S23i = np.maximum(S23i, 1e-18)
    return (s33 / Zi) ** 2 + (tau13 / S13i) ** 2 + (tau23 / S23i) ** 2
