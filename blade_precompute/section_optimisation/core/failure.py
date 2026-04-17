"""Backward-compatible re-exports; prefer :mod:`section_model.engine.failure_criteria`."""

from __future__ import annotations

from blade_precompute.section_properties.engine.failure_criteria import (
    tsai_wu_fi,
    tsai_wu_fi_plies,
    tsai_wu_strength_tensors,
    von_mises_plane_stress_fi,
)

__all__ = [
    "tsai_wu_strength_tensors",
    "tsai_wu_fi",
    "tsai_wu_fi_plies",
    "von_mises_plane_stress_fi",
]
