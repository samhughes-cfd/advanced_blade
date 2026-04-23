"""Thin-wall spar chord fractions from system layout (shared by shell + beam enrichment)."""

from __future__ import annotations

from blade_precompute.orchestration.system_layout import SystemLayoutSpec


def section_shell_spars_from_layout(layout: SystemLayoutSpec) -> list[float]:
    """
    Chord fractions (0–1) for thin-wall ``multi_cell_blade_section.build_section``.

    Uses ``web_chord_fracs`` from the active system layout; empty when there are no webs
    or geometry is not multicell (outer skin only).
    """
    if layout.n_webs == 0 or layout.geometry_mode != "multicell":
        return []
    fracs = layout.web_chord_fracs
    if len(fracs) != layout.n_webs:
        raise ValueError(
            f"web_chord_fracs length ({len(fracs)}) must equal n_webs ({layout.n_webs}) for multicell layout."
        )
    return sorted(float(x) for x in fracs)
