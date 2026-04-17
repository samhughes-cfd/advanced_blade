"""
Time histories of beam resultants and recovered stresses for fatigue analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ResultantHistory:
    """Beam resultant history at spanwise stations (beam I/O order, seven-vector)."""

    z_stations: NDArray[np.float64]  # (n_station,)
    time: NDArray[np.float64]  # (n_t,)
    N: NDArray[np.float64]  # (n_t, n_station)
    Vy: NDArray[np.float64]
    Vz: NDArray[np.float64]
    My: NDArray[np.float64]
    Mz: NDArray[np.float64]
    T: NDArray[np.float64]
    B: NDArray[np.float64]  # (n_t, n_station) bimoment

    def to_array(self) -> NDArray[np.float64]:
        """Stack ``[N, Vy, Vz, My, Mz, T, B]`` → ``(n_t, n_station, 7)``."""
        return np.stack([self.N, self.Vy, self.Vz, self.My, self.Mz, self.T, self.B], axis=-1)


@dataclass
class StressHistory:
    """Recovered ply / isotropic membrane stress histories (after L_rec / L_iso)."""

    z_stations: NDArray[np.float64]  # (n_s,)
    time: NDArray[np.float64]  # (n_t,) or chunk-sized
    sigma_composite: NDArray[np.float64]  # (n_t, n_s, n_comp_sub, n_ply, 3) material frame
    sigma_isotropic: NDArray[np.float64]  # (n_t, n_s, n_iso_sub, 3) section frame Voigt
    composite_subcomp_names: list[str]
    isotropic_subcomp_names: list[str]

    def memory_mb(self) -> float:
        """
        Total footprint of stress arrays in MB (float64).

        composite: n_t × n_s × n_comp_sub × n_ply × 3 × 8 bytes
        isotropic: n_t × n_s × n_iso_sub × 3 × 8 bytes
        """
        n_bytes = self.sigma_composite.nbytes + self.sigma_isotropic.nbytes
        return float(n_bytes) / (1024.0 * 1024.0)
