from __future__ import annotations

import numpy as np

from section_model.engine.implicit_section_geometry import BoxSDF, CircleSDF, PolygonSDF, sdf_union


def test_circle_sdf_signs() -> None:
    sdf = CircleSDF(center=(0.0, 0.0), radius=1.0)
    p = np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float64)
    v = sdf.eval(p)
    assert v[0] < 0.0
    assert v[1] > 0.0


def test_polygon_sdf_signs() -> None:
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    sdf = PolygonSDF(square)
    v = sdf.eval(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64))
    assert v[0] < 0.0
    assert v[1] > 0.0


def test_sdf_union_keeps_inner_negative() -> None:
    a = BoxSDF(center=(-0.5, 0.0), half_size=(0.6, 0.4))
    b = BoxSDF(center=(0.5, 0.0), half_size=(0.6, 0.4))
    u = sdf_union([a, b])
    v = u.eval(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64))
    assert v[0] < 0.0
    assert v[1] > 0.0

