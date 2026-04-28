"""MITC4 workflow entrypoints (compat wrapper)."""

from __future__ import annotations

from .recovery_adapter import run_section_both, run_section_with_mitc4_shell

__all__ = ["run_section_with_mitc4_shell", "run_section_both"]
