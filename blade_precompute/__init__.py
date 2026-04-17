"""
blade_precompute
===============
Precompute utilities for blade structural models:

- `section_geometry`: implicit/SDF cross-section geometry (medial axis, plots)
- `section_properties`: midsurface strip section solver (K6/K7 + recovery)
- `global_beam_model`: geometrically exact 3D beam with warping (7 DOF/node)
- `section_optimisation`: blade structural sizing evaluation/optimisation (K7 pipeline)
- `section_beam_model`: cross-section GBT (strip mesh, modes, member buckling along buckling length)
- `section_buckling`: thin precompute façade over ``section_beam_model`` (loads → JSON/plots)
- `orchestration`: system-type and material-map helpers for precompute drivers
"""

from __future__ import annotations

__all__ = [
    "global_beam_model",
    "section_beam_model",
    "section_buckling",
    "section_optimisation",
    "orchestration",
    "section_geometry",
    "section_properties",
]
