"""Regression tests for MITC4 v2 precompute wiring and package exports."""

from __future__ import annotations

from blade_precompute.orchestration.precompute import GridConfig, LinspaceSpec, grid_resolution_manifest


def test_grid_resolution_manifest_includes_section_shell_use_mitc4_v2() -> None:
    cfg = GridConfig(
        geometry=LinspaceSpec(z_min=0.0, z_max=1.0, n=2),
        structural=LinspaceSpec(z_min=0.0, z_max=1.0, n=2),
        section_plot_station_spec="all",
        n_beam_nodes=3,
        section_shell_use_mitc4_v2=True,
    )
    m = grid_resolution_manifest(cfg)
    assert m["section_shell_use_mitc4_v2"] is True
    cfg_off = GridConfig(
        geometry=LinspaceSpec(z_min=0.0, z_max=1.0, n=2),
        structural=LinspaceSpec(z_min=0.0, z_max=1.0, n=2),
        section_plot_station_spec="all",
        n_beam_nodes=3,
        section_shell_use_mitc4_v2=False,
    )
    assert grid_resolution_manifest(cfg_off)["section_shell_use_mitc4_v2"] is False


def test_section_shell_model_public_mitc4_exports() -> None:
    from blade_precompute.section_shell_model import (
        Mitc4Cluster,
        Mitc4PanelMesh,
        Mitc4SectionMesh,
        ShellMeshInputs,
        build_mitc4_mesh,
        build_section_v2,
        build_shell_mesh_inputs,
    )

    assert callable(build_mitc4_mesh)
    assert callable(build_shell_mesh_inputs)
    assert callable(build_section_v2)
    assert Mitc4SectionMesh.__name__ == "Mitc4SectionMesh"
    assert Mitc4PanelMesh.__name__ == "Mitc4PanelMesh"
    assert Mitc4Cluster.__name__ == "Mitc4Cluster"
    assert ShellMeshInputs.__name__ == "ShellMeshInputs"
