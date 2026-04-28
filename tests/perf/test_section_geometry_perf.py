"""Optional performance harness for section_geometry hot path."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from blade_precompute.section_geometry.engine.implicit_section_geometry import (
    AirfoilSDF,
    MultiCellSection,
    SDFGrid,
)
from blade_precompute.section_geometry.interface.export import SectionPropertiesReport


@pytest.mark.skipif(
    os.getenv("RUN_PERF_TESTS", "").lower() not in {"1", "true", "yes"},
    reason="Performance harness disabled by default; set RUN_PERF_TESTS=1",
)
def test_section_geometry_hot_path_5_stations_budget():
    """Time 5 representative stations and enforce a coarse runtime budget."""
    stations = np.linspace(0.0, 1.0, 5)
    t0 = time.perf_counter()
    for s in stations:
        chord = 0.8 + 0.4 * float(s)
        twist = np.deg2rad(20.0 * float(s))
        af = AirfoilSDF.from_naca("2412", n_points=200, chord=chord)
        section = MultiCellSection.twin_web(
            af,
            skin_thickness=0.003,
            web_thickness=0.004,
            cap_height=0.012,
            twist_angle=twist,
        )
        grid = SDFGrid.from_airfoil(af.rotate(twist) if abs(twist) > 1e-10 else af, nx=384, ny=180)
        report = SectionPropertiesReport(section, grid)
        props = report.to_dict()
        assert "outer_skin" in props
    elapsed = time.perf_counter() - t0
    # Coarse guardrail to catch severe regressions while remaining stable across machines.
    assert elapsed < 20.0, f"section_geometry 5-station runtime too high: {elapsed:.3f}s"

