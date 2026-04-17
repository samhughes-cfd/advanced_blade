from __future__ import annotations

import numpy as np

from section_model.engine.implicit_section_geometry import extract_midline_from_offset_boundaries


def test_midline_between_offset_boundaries() -> None:
    y = np.linspace(-1.0, 1.0, 40)
    outer = np.stack([y, np.zeros_like(y) + 0.2], axis=1)
    inner = np.stack([y, np.zeros_like(y) - 0.2], axis=1)
    out = extract_midline_from_offset_boundaries(outer, inner, strip_width_m=0.4)
    assert out.midsurface_coords_s.shape[1] == 2
    np.testing.assert_allclose(np.mean(out.midsurface_coords_s[:, 1]), 0.0, atol=1e-12)
    assert out.strip_width_m > 0.0

