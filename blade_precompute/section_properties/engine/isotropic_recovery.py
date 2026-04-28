"""
Tier 2 isotropic membrane stress and plane-stress von Mises failure index.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .failure_criteria import von_mises_plane_stress_fi


def isotropic_membrane_stresses(
    sub_resultants: NDArray[np.float64],
    thickness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Membrane stresses ``σ = N / t`` with ``N = [N11, N22, N12]`` [N/m].

    Parameters
    ----------
    sub_resultants
        ``(..., n_iso, 3)``.
    thickness
        ``(n_iso,)`` membrane thickness [m].

    Returns
    -------
    sigma
        ``(..., n_iso, 3)`` = ``[σ11, σ22, τ12]`` [Pa] in section frame.
    """
    t = np.maximum(np.asarray(thickness, dtype=np.float64), 1e-18)
    shape = (1,) * (sub_resultants.ndim - 2) + (t.shape[0], 1)
    return sub_resultants / t.reshape(shape)


def von_mises_plane_stress(
    sigma: NDArray[np.float64],
    sigma_allow: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compatibility wrapper over `failure_criteria.von_mises_plane_stress_fi`."""
    allow = np.maximum(np.asarray(sigma_allow, dtype=np.float64), 1e-18)
    return von_mises_plane_stress_fi(np.asarray(sigma, dtype=np.float64), allow)
