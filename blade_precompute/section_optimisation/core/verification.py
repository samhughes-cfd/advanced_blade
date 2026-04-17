"""Reference-station verification helpers for section/beam completeness checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReferenceStation:
    """Reference station metadata and expected section matrices."""

    name: str
    z: float
    K6_ref: NDArray[np.float64]
    K7_ref: NDArray[np.float64]


@dataclass(frozen=True)
class VerificationMetrics:
    """Relative error metrics for fast regression gating."""

    rel_K6_fro: float
    rel_K7_fro: float
    rel_shear_block: float
    rel_torsion: float


def _rel_err(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    na = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / max(na, 1e-12))


def compute_station_metrics(K6_pred: NDArray[np.float64], K7_pred: NDArray[np.float64], ref: ReferenceStation) -> VerificationMetrics:
    """Compute compact errors for station-to-reference comparisons."""
    K6p = np.asarray(K6_pred, dtype=np.float64).reshape(6, 6)
    K7p = np.asarray(K7_pred, dtype=np.float64).reshape(7, 7)
    K6r = np.asarray(ref.K6_ref, dtype=np.float64).reshape(6, 6)
    K7r = np.asarray(ref.K7_ref, dtype=np.float64).reshape(7, 7)
    return VerificationMetrics(
        rel_K6_fro=_rel_err(K6r, K6p),
        rel_K7_fro=_rel_err(K7r, K7p),
        rel_shear_block=_rel_err(K6r[4:6, 4:6], K6p[4:6, 4:6]),
        rel_torsion=_rel_err(np.array([K6r[3, 3]]), np.array([K6p[3, 3]])),
    )
