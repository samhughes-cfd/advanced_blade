"""Pure stress-frame transforms (no section topology)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def plane_stress_voigt_from_R(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Build ``(3, 3)`` matrix ``M`` with ``σ_voigt' = M @ σ_voigt`` where Voigt is
    ``[σ11, σ22, τ12]`` and ``σ`` is embedded as a symmetric ``(3, 3)`` tensor
    with zero out-of-plane rows/columns.

    Level-1 deformed-frame rotation: full ``3×3`` ``R`` acts on the symmetric
    ``3×3`` stress tensor; the returned Voigt triplet is the upper ``2×2`` block
    of ``R @ σ @ R.T`` in the same canonical embedding.
    """
    r = np.asarray(R, dtype=np.float64).reshape(3, 3)
    b1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    b2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    b3 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    m = np.zeros((3, 3), dtype=np.float64)
    for a, b in enumerate([b1, b2, b3]):
        sp = r @ b @ r.T
        m[0, a] = sp[0, 0]
        m[1, a] = sp[1, 1]
        m[2, a] = sp[0, 1]
    return m
