"""Kreisselmeier–Steinhauser constraint aggregation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ks_aggregate(FI: NDArray[np.float64], rho: float) -> float:
    r"""
    ``KS(FI, ρ) = (1/ρ) ln(Σ_k exp(ρ (FI_k - 1))) + 1``.

    Constraint ``KS <= 1`` implies ``max_k FI_k \lesssim 1`` for large ``ρ``.
    """
    rho = max(float(rho), 1e-6)
    x = np.asarray(FI, dtype=np.float64).ravel()
    if x.size == 0:
        # Return 0.0 (not 1.0) so the constraint ks <= 1 is trivially satisfied
        # and the log correctly shows "no active failure indices" rather than a
        # spurious binding constraint.
        return 0.0
    y = rho * (x - 1.0)
    m = float(np.max(y))
    ex = np.exp(y - m)
    return float(m / rho + (1.0 / rho) * np.log(np.sum(ex)) + 1.0)
