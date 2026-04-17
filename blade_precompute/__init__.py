"""
blade_precompute
===============
Precompute utilities for blade structural models:

- `section_geometry`: implicit/SDF cross-section geometry (medial axis, plots)
- `section_properties`: midsurface strip section solver (K6/K7 + recovery)
- `global_beam_model`: geometrically exact 3D beam with warping (7 DOF/node)
- `section_optimisation`: blade structural sizing evaluation/optimisation (K7 pipeline)
- `orchestration`: system-type and material-map helpers for precompute drivers
"""

from __future__ import annotations

__all__ = [
    "global_beam_model",
    "section_optimisation",
    "orchestration",
    "section_geometry",
    "section_properties",
]
