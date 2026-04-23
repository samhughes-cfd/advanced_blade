"""
section_buckling
================
Example layer: bridge ``SectionDefinition`` / extreme loads to GBT workflows
and emit station JSON and plots.

Core GBT physics lives in :mod:`section_beam_model` (``examples/section_beam_model``).
"""

from .interface import (
    analyze_station_buckling,
    cross_section_for_subcomponent_indices,
    line_mesh_meta,
    plot_buckling_member_overview_grid,
    plot_cross_section_mode_wireframes,
    plot_member_coupled_section_wireframe_approx,
    safe_subcomponent_filename_label,
    section_definition_to_gbt_cross_section,
    wall_definitions_from_line_mesh,
)

__all__ = [
    "analyze_station_buckling",
    "cross_section_for_subcomponent_indices",
    "line_mesh_meta",
    "plot_buckling_member_overview_grid",
    "plot_cross_section_mode_wireframes",
    "plot_member_coupled_section_wireframe_approx",
    "safe_subcomponent_filename_label",
    "section_definition_to_gbt_cross_section",
    "wall_definitions_from_line_mesh",
]
