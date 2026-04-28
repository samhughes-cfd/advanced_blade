"""
Section shell model: midline-to-MITC4 mesh, closed-cell recovery, and local CLPT handoff.

Public MITC4 strip meshing follows ``build_shell_mesh_inputs`` → ``build_section_v2`` →
``build_mitc4_mesh``; see ``docs/SHELL_MODEL.md``.  Legacy CLPT examples reuse
:mod:`examples.section_stress_model` for reference physics.
"""

from __future__ import annotations

from .lib.mitc4_mesh import (
    Mitc4Cluster,
    Mitc4PanelMesh,
    Mitc4SectionMesh,
    build_mitc4_mesh,
)
from .lib.shell_inputs_from_section import (
    ShellMeshInputs,
    build_shell_mesh_inputs,
)
from .lib.topology_v2 import build_section_v2
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
    "Mitc4Cluster",
    "Mitc4PanelMesh",
    "Mitc4SectionMesh",
    "ShellMeshInputs",
    "build_mitc4_mesh",
    "build_section_v2",
    "build_shell_mesh_inputs",
    "FieldProvenance",
    "ProvenanceKind",
    "ShellPanelResultants",
    "SectionShellRecoveryBundle",
    "run_section_with_shell_mapping",
    "panel_station_shell_resultants",
    "solve_station_clpt_shell",
    "StationCLPTShellResult",
]
