"""
S–N curves for Miner damage (vectorised ``cycles_to_failure``).

Curve parameters use the same stress units as the time histories (typically Pa).
Factory curves documented with Pa-equivalent ``log_a`` where originals are in MPa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SNcurve:
    name: str
    m: float
    log_a: float
    sigma_uts: float | None = None
    stress_limit: float | None = None

    def cycles_to_failure(self, delta_sigma: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        ``N_f = 10**log_a / delta_sigma**m`` (vectorised).

        Returns ``inf`` where ``delta_sigma <= 0`` or ``delta_sigma < stress_limit``.
        """
        ds = np.asarray(delta_sigma, dtype=np.float64)
        out = np.full_like(ds, np.inf, dtype=np.float64)
        pos = ds > 0.0
        if self.stress_limit is not None:
            pos &= ds >= float(self.stress_limit)
        safe = np.where(pos, ds, 1.0)
        out = np.where(pos, (10.0**float(self.log_a)) / (safe**float(self.m)), np.inf)
        return out

    def apply_goodman(
        self,
        delta_sigma: NDArray[np.float64],
        sigma_mean: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Goodman: ``delta_sigma_eff = delta_sigma / (1 - sigma_mean / sigma_uts)``.

        If ``sigma_uts`` is unset, returns ``delta_sigma`` unchanged.

        Goodman is often conservative for metals; for composites a modified Goodman
        or Gerber correction may be more appropriate — document in design reviews.
        """
        if self.sigma_uts is None:
            return np.asarray(delta_sigma, dtype=np.float64)
        uts = float(self.sigma_uts)
        denom = 1.0 - np.asarray(sigma_mean, dtype=np.float64) / uts
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + (1e-12 if denom >= 0 else -1e-12), denom)
        return np.asarray(delta_sigma, dtype=np.float64) / denom

    @staticmethod
    def steel_dnv() -> "SNcurve":
        """DNV-RP-C203 D-curve style: ``m=3``, ``log_a`` adjusted for ``delta_sigma`` in Pa."""
        m = 3.0
        log_a_mpa = 12.164
        log_a_pa = log_a_mpa + 6.0 * m
        return SNcurve(name="DNV_D_Pa", m=m, log_a=log_a_pa, sigma_uts=400e6)

    @staticmethod
    def gfrp_blade() -> "SNcurve":
        """Representative GFRP blade trend (Pa); replace with project-certified GL/IEC data."""
        m = 10.0
        log_a_mpa = 25.0
        log_a_pa = log_a_mpa + 6.0 * m
        return SNcurve(name="GFRP_blade_Pa", m=m, log_a=log_a_pa)

    @staticmethod
    def cfrp_blade() -> "SNcurve":
        """Representative CFRP blade trend (Pa); replace with project-certified GL/IEC data."""
        m = 12.0
        log_a_mpa = 28.0
        log_a_pa = log_a_mpa + 6.0 * m
        return SNcurve(name="CFRP_blade_Pa", m=m, log_a=log_a_pa)
