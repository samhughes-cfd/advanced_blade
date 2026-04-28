"""
topology_v2
===========
Per-subcomponent topology generator for the section_shell_model stage.

Consumes a :class:`ShellMeshInputs` payload (B-frame midlines built directly
from ``section_geometry``'s :class:`MultiCellSection`) and produces one
:class:`Panel` per subcomponent midline. Each ``OuterSkin``, ``ShearWeb``
and ``SparCap`` / ``ContinuousSparCap`` is meshed as an independent shell
strip carrying its own laminate.

This is a deliberately thin layer: all geometry decisions
(midline location, web alignment, cap offset and slicing) live on the
``section_geometry`` subcomponent classes themselves; this module just
attaches a laminate and emits one ``Panel`` per midline.

Returned tuple ``(panels, booms, webs_geom, n_cells)`` keeps the same
shape as :func:`examples.section_stress_model.multi_cell_blade_section.build_section`
for backwards compatibility, but ``booms`` is empty and ``n_cells`` is
reported as ``n_webs + 1`` for diagnostics only — the panels themselves
are not split at web junctions, so callers that depend on per-cell
Bredt closure must mesh each panel into FE shell strips downstream.

**Panel emission order: skin → caps → webs.**
``panels[i]`` does NOT correspond to ``shell_inputs.midlines[i]``; the
midlines list preserves dict-insertion order (skin + webs from
``_components_unrotated``, then caps from ``_spar_cap_components_unrotated``).
Always reconstruct the ordered sequence as::

    ordered = [m for m in midlines if m.kind == "skin"] \\
            + [m for m in midlines if m.kind == "cap"]  \\
            + [m for m in midlines if m.kind == "web"]

to pair ``panels[i]`` with the correct strip.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _ensure_legacy_solver_on_path() -> None:
    """Add ``examples/section_stress_model`` to ``sys.path`` so we can reuse
    the existing :class:`Panel` / :class:`BoomNode` / :class:`Laminate` types
    and default laminates without copy-pasting them.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    legacy = repo_root / "examples" / "section_stress_model"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))


_ensure_legacy_solver_on_path()

from multi_cell_blade_section import (  # noqa: E402  (post-sys.path manipulation)
    CAP_LAM,
    Laminate,
    Panel,
    SKIN_LAM,
    WEB_LAM,
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyV2Diagnostics:
    """Lightweight diagnostics returned alongside the topology so callers
    can log a per-panel summary without re-traversing the panel objects.
    """

    n_cells: int
    panel_summary: list[dict[str, Any]]


def _polyline_arc_length(pts: NDArray[np.float64]) -> float:
    if pts.shape[0] < 2:
        return 0.0
    return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())


def _laminate_for(kind: str, sl: Laminate, wl: Laminate, cl: Laminate) -> Laminate:
    if kind == "skin":
        return sl
    if kind == "web":
        return wl
    if kind == "cap":
        return cl
    raise ValueError(f"Unknown subcomponent kind {kind!r}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_section_v2(
    shell_inputs: Any,
    *,
    skin_lam: Laminate | None = None,
    web_lam: Laminate | None = None,
    cap_lam: Laminate | None = None,
    return_diagnostics: bool = False,
):
    """Build the Panel topology directly from per-subcomponent midlines.

    Parameters
    ----------
    shell_inputs
        :class:`blade_precompute.section_shell_model.lib.shell_inputs_from_section.ShellMeshInputs`
        instance produced by :func:`build_shell_mesh_inputs`. Must expose
        ``midlines``: a list of :class:`SubcomponentMidline` records.
    skin_lam, web_lam, cap_lam
        Optional overrides for the skin / web / cap laminates. Defaults
        come from :mod:`multi_cell_blade_section` (``SKIN_LAM`` / ``WEB_LAM``
        / ``CAP_LAM``).
    return_diagnostics
        If ``True``, returns ``(panels, booms, webs_geom, n_cells, diagnostics)``;
        otherwise returns ``(panels, booms, webs_geom, n_cells)``.

    Returns
    -------
    (panels, booms, webs_geom, n_cells[, diagnostics])
        ``panels`` contains one :class:`Panel` per subcomponent midline,
        in the order ``skin → caps → webs``. ``booms`` is always empty
        (each midline is integrated as one continuous panel). ``webs_geom``
        is the legacy-shaped list of ``(top_b, bot_b)`` endpoint pairs for
        each web midline (used by the existing plotting helpers).
        ``n_cells = n_webs + 1`` for diagnostics only.
    """
    sl = skin_lam if skin_lam is not None else SKIN_LAM
    wl = web_lam if web_lam is not None else WEB_LAM
    cl = cap_lam if cap_lam is not None else CAP_LAM

    midlines = list(getattr(shell_inputs, "midlines", []))
    if not midlines:
        raise ValueError(
            "shell_inputs.midlines is empty; nothing to mesh. Did the "
            "MultiCellSection finish building before calling the adapter?"
        )

    skin_midlines = [m for m in midlines if m.kind == "skin"]
    cap_midlines = [m for m in midlines if m.kind == "cap"]
    web_midlines = [m for m in midlines if m.kind == "web"]
    ordered = skin_midlines + cap_midlines + web_midlines

    panels: list[Panel] = []
    panel_summary: list[dict[str, Any]] = []
    webs_geom: list[tuple[NDArray, NDArray]] = []

    for cell_id, m in enumerate(ordered):
        lam = _laminate_for(m.kind, sl, wl, cl)
        nodes = np.asarray(m.midline_b, dtype=float)
        if nodes.shape[0] < 2:
            raise ValueError(
                f"Subcomponent {m.label!r} midline has fewer than 2 vertices."
            )
        label = f"{m.kind}:{m.label}"
        panels.append(Panel(nodes, lam, cell_id, None, label))
        panel_summary.append({
            "kind": m.kind,
            "label": m.label,
            "panel_label": label,
            "cell_id": cell_id,
            "thickness_m": float(m.thickness_m),
            "n_pts": int(nodes.shape[0]),
            "arc_length_m": _polyline_arc_length(nodes),
            "x_b_range": [float(nodes[:, 0].min()), float(nodes[:, 0].max())],
            "y_b_range": [float(nodes[:, 1].min()), float(nodes[:, 1].max())],
            "lam_E_pa": float(lam.E),
            "lam_t_m": float(lam.t),
            "closed": bool(m.closed),
            "surface": m.surface,
            "alignment": m.alignment,
        })
        if m.kind == "web":
            webs_geom.append((nodes[0].copy(), nodes[-1].copy()))

    n_cells = len(web_midlines) + 1

    if return_diagnostics:
        diagnostics = TopologyV2Diagnostics(
            n_cells=n_cells,
            panel_summary=panel_summary,
        )
        return panels, [], webs_geom, n_cells, diagnostics

    return panels, [], webs_geom, n_cells


__all__ = ["build_section_v2", "TopologyV2Diagnostics"]
