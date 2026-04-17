"""Miner damage vectorisation."""

from __future__ import annotations

import numpy as np

from blade_analysis.fatigue_damage.engine.damage import miner_damage
from blade_analysis.fatigue_damage.engine.sn_curves import SNcurve


def test_miner_damage_matches_manual_sum():
    sn = SNcurve(name="test", m=3.0, log_a=12.0, sigma_uts=None)
    ranges = np.array([[100.0, 200.0], [100.0, 200.0]], dtype=np.float64)
    means = np.zeros_like(ranges)
    counts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    d = miner_damage(ranges, means, counts, sn, apply_goodman=False)
    Nf = sn.cycles_to_failure(ranges)
    manual = np.sum(counts / Nf, axis=0)
    np.testing.assert_allclose(d, manual, rtol=1e-12)
