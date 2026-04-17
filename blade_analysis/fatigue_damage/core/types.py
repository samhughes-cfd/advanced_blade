"""Fatigue pipeline output types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RainflowBins:
    """Binned rainflow results (fixed ``n_bin`` stress-range bins per plan)."""

    ranges_comp: NDArray[np.float64]  # (n_bin, n_s, n_comp_sub, n_ply)
    means_comp: NDArray[np.float64]
    counts_comp: NDArray[np.float64]

    ranges_iso: NDArray[np.float64]  # (n_bin, n_s, n_iso_sub)
    means_iso: NDArray[np.float64]
    counts_iso: NDArray[np.float64]


@dataclass
class FatigueResult:
    damage_composite: NDArray[np.float64]  # (n_s, n_comp_sub, n_ply)
    damage_isotropic: NDArray[np.float64]  # (n_s, n_iso_sub)
    damage_delam: NDArray[np.float64] | None

    life_composite: NDArray[np.float64]
    life_isotropic: NDArray[np.float64]

    max_damage_composite: float
    max_damage_isotropic: float
    worst_composite: tuple[int, str, int]
    worst_isotropic: tuple[int, str]
    fatigue_critical_material: str

    fi_static_tw: NDArray[np.float64]  # (n_s, n_comp_sub, n_ply)
    fi_static_vm: NDArray[np.float64]  # (n_s, n_iso_sub)

    stress_component_used: int
    goodman_applied: bool
    design_life_years: float
    memory_mode: str
    rainflow_bins: RainflowBins | None = None
