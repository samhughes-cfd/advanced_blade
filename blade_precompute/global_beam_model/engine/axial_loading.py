"""
Centrifugal + gravity axial line load along the beam (root-to-tip / beam ``x``).

``radial_r_m(z)`` is the hub-centred rotation radius (same as ``PrecomputeInputs.radial_r_m``)
[``m``].  ``R_tip`` in :class:`AxialLoadingConfig` is ``max(radial_r_m)`` and sets
``omega = U_inf * TSR / R_tip`` [rad/s].

The spanwise line load is applied in :func:`build_beam_loads_distributed` on
``distributed_q[:, span_axis]`` (default ``span_axis=2``, global **z**), where it
**adds** to the hydrodynamic ``q_z`` column on that axis.

.. math::

   q_x(z) = \\mu(z)\\left(\\omega^2 r_\\text{radial}(z) + g\\cos\\theta_\\text{ax}\\right)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.engine.distributed_load_integrator import (
    _trapz_piecewise_to_tip,
)


@dataclass(frozen=True)
class AxialLoadingConfig:
    """Operating point for axial (centrifugal + self-weight) loading."""

    u_inf_m_s: float
    tip_speed_ratio: float
    r_tip_m: float
    gravity_m_s2: float
    azimuth_deg: float
    """Blade-out-of-plane: 0 = blade down (gravity along +x), 90 = horizontal, 180 = up."""
    enabled: bool = True

    def omega_rad_s(self) -> float:
        """Turbine angular rate [rad/s] from TSR, free-stream, and ``r_tip_m``."""
        if not self.enabled or float(self.r_tip_m) <= 0.0:
            return 0.0
        return float(self.u_inf_m_s) * float(self.tip_speed_ratio) / float(self.r_tip_m)


def q_x_distributed(
    z: NDArray[np.float64],
    r_radial: NDArray[np.float64],
    mu: NDArray[np.float64],
    cfg: AxialLoadingConfig,
) -> NDArray[np.float64]:
    """Line load :math:`q_x` [N/m] (positive = tension) at each tabulated `z` station."""
    if not cfg.enabled:
        return np.zeros_like(z, dtype=np.float64)
    w = float(cfg.omega_rad_s())
    gcos = float(cfg.gravity_m_s2) * float(np.cos(np.deg2rad(float(cfg.azimuth_deg))))
    m = np.asarray(mu, dtype=np.float64).ravel()
    r = np.asarray(r_radial, dtype=np.float64).ravel()
    zz = np.asarray(z, dtype=np.float64).ravel()
    if not (m.shape[0] == r.shape[0] == zz.shape[0]):
        raise ValueError("z, r_radial, and mu must have the same length.")
    return m * (w * w * r + gcos)


def axial_force_distribution(
    z: NDArray[np.float64],
    q_x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Internal axial :math:`N(z)=\\int_z^{z_\\text{tip}}q_x(s)\\,\\mathrm{d}s` (cantilever-to-tip trapezoid)."""
    return _trapz_piecewise_to_tip(
        np.asarray(z, dtype=np.float64).ravel(),
        np.asarray(q_x, dtype=np.float64).ravel(),
    )


def manifest_dict(
    cfg: AxialLoadingConfig,
    *,
    mu_source: str = "section_properties",
) -> dict:
    """JSON-serialisable metadata for job provenance (``inputs.json`` / ``summary``)."""
    return {
        "enabled": bool(cfg.enabled),
        "u_inf_m_s": float(cfg.u_inf_m_s),
        "tip_speed_ratio": float(cfg.tip_speed_ratio),
        "r_tip_m": float(cfg.r_tip_m),
        "gravity_m_s2": float(cfg.gravity_m_s2),
        "azimuth_deg": float(cfg.azimuth_deg),
        "omega_rad_s": float(cfg.omega_rad_s()),
        "mu_source": str(mu_source),
    }
