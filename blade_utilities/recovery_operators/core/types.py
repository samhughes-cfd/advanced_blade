"""Storage and protocol types for runtime recovery-operator bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RecoveryOperatorBundle:
    """Runtime-only fused operators with no serialization contract."""

    H_eps: NDArray[np.float64]
    L_sec: NDArray[np.float64]
    M_voigt: NDArray[np.float64]
    D_z: NDArray[np.float64]
    G_if: Optional[NDArray[np.float64]]
    z_stations: NDArray[np.float64]
    composite_subcomp_idx: list[int]
    composite_subcomp_names: list[str]
    ply_count: NDArray[np.int32]


class RecoveryOperatorBundleProtocol(Protocol):
    """Structural typing for users that consume operator bundles."""

    H_eps: NDArray[np.float64]
    L_sec: NDArray[np.float64]
    M_voigt: NDArray[np.float64]
    D_z: NDArray[np.float64]
    G_if: Optional[NDArray[np.float64]]
    z_stations: NDArray[np.float64]
    composite_subcomp_idx: list[int]
    composite_subcomp_names: list[str]
    ply_count: NDArray[np.int32]
