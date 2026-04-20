"""
Vlasov thin-walled warping: bimoment-induced axial stress.

  sigma_omega(s) = B * omega_hat(s) / I_omega

where omega_hat is normalized sectorial coordinate (pole at shear centre, mean removed)
and I_omega = ∫ E_n omega_hat^2 t ds.

B is a prescribed bimoment [N·m²] at the section (spanwise distribution not solved here).
"""

from __future__ import annotations

import numpy as np


def sigma_from_bimoment(omega_hat: np.ndarray, B: float, I_omega: float) -> np.ndarray:
    """Warping normal stress along a polyline (vertex values), [Pa] if B, I_omega in SI."""
    if abs(I_omega) < 1e-40:
        I_omega = 1e-40
    return B * np.asarray(omega_hat, dtype=float) / I_omega


def axial_resultant_from_warping_stress(
    loop_yz: np.ndarray,
    sigma_omega: np.ndarray,
    t: float,
) -> float:
    """Sanity: ∫ sigma_omega * t ds should be ~ 0 for pure bimoment."""
    y = loop_yz[:, 0]
    z = loop_yz[:, 1]
    n = len(y)
    acc = 0.0
    for i in range(n - 1):
        sig = 0.5 * (sigma_omega[i] + sigma_omega[i + 1])
        ds = np.hypot(y[i + 1] - y[i], z[i + 1] - z[i])
        acc += sig * t * ds
    return float(acc)
