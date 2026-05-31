"""Backward-compatible YAML section loader module."""

from __future__ import annotations

from .section_loader import load_section_from_spec, load_section_from_yaml

__all__ = ["load_section_from_spec", "load_section_from_yaml"]
