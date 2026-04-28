"""Optional design optimisation visualisation."""

from __future__ import annotations

from .plot import (
    plot_beam_nr_residual_tail,
    plot_design_vector_vs_span,
    plot_fi_reserve_vs_span,
    plot_fi_span_heatmap,
    plot_fi_vs_span_per_iteration,
    plot_governing_subcomp_hashin_vs_span,
    plot_k7_condition_summary,
    plot_max_fi_vs_span,
    plot_mitc4_vs_hashin_span,
    plot_optimisation_history,
    plot_optimisation_objective_dual_axis,
    plot_optimisation_slack_stiffness_history,
    plot_panel_buckling_fi_vs_span,
    plot_resultants_with_max_fi,
    plot_thickness_delta_vs_span,
    plot_thickness_share_vs_span,
)

__all__ = [
    "plot_beam_nr_residual_tail",
    "plot_design_vector_vs_span",
    "plot_fi_reserve_vs_span",
    "plot_fi_span_heatmap",
    "plot_fi_vs_span_per_iteration",
    "plot_governing_subcomp_hashin_vs_span",
    "plot_k7_condition_summary",
    "plot_max_fi_vs_span",
    "plot_mitc4_vs_hashin_span",
    "plot_optimisation_history",
    "plot_optimisation_objective_dual_axis",
    "plot_optimisation_slack_stiffness_history",
    "plot_panel_buckling_fi_vs_span",
    "plot_resultants_with_max_fi",
    "plot_thickness_delta_vs_span",
    "plot_thickness_share_vs_span",
]
