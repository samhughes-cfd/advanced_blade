"""Scalar stress drivers for fatigue (plane-stress Voigt, von Mises)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ply_stress_component(sigma: NDArray[np.float64], component: int) -> NDArray[np.float64]:
    """Extract Voigt component ``0=σ11, 1=σ22, 2=τ12`` from trailing axis."""
    return np.asarray(sigma[..., component], dtype=np.float64)


def von_mises_plane_stress(s11: NDArray[np.float64], s22: NDArray[np.float64], t12: NDArray[np.float64]) -> NDArray[np.float64]:
    """Plane-stress von Mises: ``sqrt(s11^2 - s11*s22 + s22^2 + 3*t12^2)``."""
    return np.sqrt(np.maximum(0.0, s11**2 - s11 * s22 + s22**2 + 3.0 * t12**2))
