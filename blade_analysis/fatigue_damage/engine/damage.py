"""
Miner damage accumulation (vectorised over bins and trailing axes).

No FE / CLPT in this module — damage from binned rainflow + S–N only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .sn_curves import SNcurve


def miner_damage(
    ranges: NDArray[np.float64],
    means: NDArray[np.float64],
    counts: NDArray[np.float64],
    sn_curve: SNcurve,
    apply_goodman: bool = False,
) -> NDArray[np.float64]:
    """
    Miner's rule ``D = sum_i n_i / N_f,i`` summed over axis 0 (bin axis).

    Trailing axes (station, subcomponent, ply, …) are fully vectorised.
    No FE / CLPT — S–N lookup only.
    """
    if apply_goodman:
        eff_ranges = sn_curve.apply_goodman(ranges, means)
    else:
        eff_ranges = ranges
    N_f = sn_curve.cycles_to_failure(eff_ranges)
    damage_per_bin = np.divide(
        counts,
        N_f,
        out=np.zeros_like(counts, dtype=np.float64),
        where=np.isfinite(N_f) & (N_f > 0.0),
    )
    return np.sum(damage_per_bin, axis=0)


def life_from_damage(damage: NDArray[np.float64], design_life_years: float) -> NDArray[np.float64]:
    """``life = design_life_years / D``; ``inf`` where ``D == 0``."""
    d = np.asarray(damage, dtype=np.float64)
    out = np.full_like(d, np.inf, dtype=np.float64)
    pos = d > 0.0
    out = np.where(pos, float(design_life_years) / d, np.inf)
    return out
