"""
blade_precompute
===============
Precompute utilities for blade structural models:

- `section_geometry`: implicit/SDF cross-section geometry (medial axis, plots)
- `section_properties`: midsurface strip section solver (K6/K7 + recovery)
- `beam_model`: geometrically exact 3D beam with warping (7 DOF/node)
- `design_optimisation`: sizing evaluation/optimisation pipeline
- `beam_model`: stable alias for :mod:`blade_precompute.global_beam_model`
- `orchestration`: system-type and material-map helpers for precompute drivers
"""

from __future__ import annotations

__all__ = [
    "beam_model",
    "design_optimisation",
    "orchestration",
    "section_geometry",
    "section_properties",
]

