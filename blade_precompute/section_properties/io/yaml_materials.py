"""Backward-compatible material loader aliases."""

from __future__ import annotations

from .materials_loader import laminate_from_mapping_spec, orthotropic_ply_from_dict

laminate_from_yaml_spec = laminate_from_mapping_spec

__all__ = [
    "laminate_from_mapping_spec",
    "laminate_from_yaml_spec",
    "orthotropic_ply_from_dict",
]
