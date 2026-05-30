"""Panel buckling helper: section-order resultant → K6 force vector mapping."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_optimisation.engine.evaluator import _beam7_to_reference_forces6


def test_beam7_to_reference_forces6_ordering() -> None:
    r7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    f6 = _beam7_to_reference_forces6(r7)
    np.testing.assert_array_equal(
        f6,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
    )
