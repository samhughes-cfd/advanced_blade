"""
Spanwise stiffness aggregate from section ``K7`` matrices.

The Tier-B sizing driver prescribes internal resultants from :class:`ExtremeLoads`
independently of ``K7`` (see :mod:`blade_precompute.section_optimisation.engine.beam_k7`).
This metric is therefore a **section constitutive** proxy (integrated ``trace(K7)`` along
the span), **not** strain energy or compliance under those loads.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def integrated_k7_trace(K7_stack: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    """
    Trapezoidal rule on ``z`` of per-station ``trace(K7)``, matching the spanwise
    integration style of :func:`mass_objective`.
    """
    K7 = np.asarray(K7_stack, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    if K7.ndim != 3 or K7.shape[-2:] != (7, 7):
        raise ValueError("K7_stack must have shape (n_stations, 7, 7).")
    if K7.shape[0] != z.shape[0]:
        raise ValueError("K7_stack first axis must match z length.")
    tr = np.trace(K7, axis1=-2, axis2=-1)
    if z.size == 1:
        return float(tr[0])
    dz = np.diff(z)
    return float(np.sum(0.5 * (tr[:-1] + tr[1:]) * dz))
