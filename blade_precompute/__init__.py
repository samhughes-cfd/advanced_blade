"""
blade_precompute
===============
Precompute utilities for blade structural models:

- `section_geometry`: implicit/SDF cross-section geometry (medial axis, plots)
- `section_properties`: midsurface strip section solver (K6/K7 + recovery)
- `section_shell_model`: MITC4/CLPT shell handoff and section recovery (moved from examples)
- `global_beam_model`: geometrically exact 3D beam with warping (7 DOF/node)
- `section_optimisation`: blade structural sizing evaluation/optimisation (K7 pipeline)
- `orchestration`: system-type and material-map helpers for precompute drivers

GBT workflows live under ``examples/section_beam_model`` and ``examples/section_buckling``.
"""

from __future__ import annotations

__all__ = [
    "global_beam_model",
    "section_shell_model",
    "section_optimisation",
    "orchestration",
    "section_geometry",
    "section_properties",
]
