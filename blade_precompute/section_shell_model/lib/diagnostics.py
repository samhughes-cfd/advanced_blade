"""Section shell diagnostics entrypoints (compat wrapper)."""

from __future__ import annotations

from .recovery_adapter import (
    build_load_reaction_audit,
    check_cluster_equilibrium,
    check_panel_equilibrium,
)

__all__ = ["check_panel_equilibrium", "check_cluster_equilibrium", "build_load_reaction_audit"]
