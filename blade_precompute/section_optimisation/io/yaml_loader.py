"""Backward-compatible YAML loader module.

Use :mod:`blade_precompute.section_optimisation.io.blade_geometry_loader` for
new code; this shim keeps existing imports working while ``load_mapping`` now
handles both JSON and YAML files.
"""

from __future__ import annotations

from .blade_geometry_loader import load_blade_geometry

__all__ = ["load_blade_geometry"]
