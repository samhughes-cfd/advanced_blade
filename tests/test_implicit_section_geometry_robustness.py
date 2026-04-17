from __future__ import annotations

import numpy as np
import pytest

from section_model.engine.implicit_section_geometry import extract_midline_from_offset_boundaries


def test_midline_raises_for_invalid_thickness_pair() -> None:
    a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        extract_midline_from_offset_boundaries(a, a, strip_width_m=0.1)


def test_midline_filters_duplicate_points() -> None:
    outer = np.array([[0.0, 0.2], [0.5, 0.2], [0.5, 0.2], [1.0, 0.2]], dtype=np.float64)
    inner = np.array([[0.0, -0.2], [0.5, -0.2], [0.5, -0.2], [1.0, -0.2]], dtype=np.float64)
    out = extract_midline_from_offset_boundaries(outer, inner, strip_width_m=0.4)
    assert out.midsurface_coords_s.shape[0] >= 2

