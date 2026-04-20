"""
Section shell model: closed-cell section recovery adapter + local CLPT shell handoff.

MVP reuses :mod:`examples.section_stress_model` for physics; see ``docs/SHELL_MODEL.md``.
"""

from __future__ import annotations

from .lib.types import (
    FieldProvenance,
    ProvenanceKind,
    ShellPanelResultants,
    SectionShellRecoveryBundle,
)
from .lib.recovery_adapter import (
    run_section_with_shell_mapping,
    panel_station_shell_resultants,
)
from .lib.local_clpt_shell import (
    solve_station_clpt_shell,
    StationCLPTShellResult,
)

__all__ = [
    "FieldProvenance",
    "ProvenanceKind",
    "ShellPanelResultants",
    "SectionShellRecoveryBundle",
    "run_section_with_shell_mapping",
    "panel_station_shell_resultants",
    "solve_station_clpt_shell",
    "StationCLPTShellResult",
]
