"""
Per-station checks on tabulated ``K6`` / ``K7`` before beam consumption.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray


def validate_k6_k7_stacks(
    K6: NDArray[np.float64],
    K7: NDArray[np.float64] | None = None,
    *,
    strict: bool = False,
) -> List[str]:
    """
    Return warning messages for non-finite / indefinite / non-positive-diagonal issues.

    When ``strict`` is True, raises ``ValueError`` with the joined messages instead.
    """
    K6a = np.asarray(K6, dtype=np.float64)
    if K6a.ndim == 2:
        K6a = K6a[None, ...]
    n = K6a.shape[0]
    msgs: List[str] = []
    K7a = None
    if K7 is not None:
        K7a = np.asarray(K7, dtype=np.float64)
        if K7a.ndim == 2:
            K7a = K7a[None, ...]
        if K7a.shape[0] != n:
            raise ValueError("K7 first dim must match K6 stations.")

    for i in range(n):
        k6 = 0.5 * (K6a[i] + K6a[i].T)
        if not np.all(np.isfinite(k6)):
            msgs.append(f"station {i}: K6 has non-finite entries")
            continue
        w = np.linalg.eigvalsh(k6)
        lam_max = max(float(np.max(np.abs(w))), 1.0)
        if float(w[0]) < -1e-6 * lam_max:
            msgs.append(f"station {i}: K6 indefinite (min eigenvalue {w[0]:.3e})")
        for j, name in enumerate(("EA", "EIy", "EIz", "GJ", "GAy", "GAz")):
            if float(k6[j, j]) <= 0.0:
                msgs.append(f"station {i}: K6 diagonal {name} = {k6[j, j]:.3e} <= 0")

        if K7a is not None:
            k7 = 0.5 * (K7a[i] + K7a[i].T)
            if not np.all(np.isfinite(k7)):
                msgs.append(f"station {i}: K7 has non-finite entries")
                continue
            w7 = np.linalg.eigvalsh(k7)
            lam7 = max(float(np.max(np.abs(w7))), 1.0)
            if float(w7[0]) < -1e-6 * lam7:
                msgs.append(f"station {i}: K7 indefinite (min eigenvalue {w7[0]:.3e})")

    if strict and msgs:
        raise ValueError("; ".join(msgs))
    return msgs
