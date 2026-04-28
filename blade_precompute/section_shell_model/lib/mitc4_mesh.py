"""
mitc4_mesh
==========
Build an explicit MITC4 shell-element mesh from the per-subcomponent midlines
produced by :func:`build_shell_mesh_inputs`.

The MITC4 assembly (:func:`solve_global_coupled_mitc4`) already discretises
each :class:`Panel` into ``n_elements`` quad elements as one spanwise strip
(``L_x = 1`` m). This module is a thin convenience layer that:

* Interpolates the midline polyline onto the same ``s_nodes`` the solver
  uses (including polyline-knot merge for cap/web and sparse skin strips),
  so callers can plot or export the **actual** MITC4 element node
  positions in the cross-section ``(y, z)`` plane.
* Surfaces the endpoint-clustering topology
  (which junctions get coincident DOFs / MPCs) without re-running the
  full solve, so it can be inspected and visualised before any loads
  are applied.

Returned :class:`Mitc4SectionMesh` is laid out so it can be fed straight
into :func:`solve_global_coupled_mitc4` — ``mesh.panels`` and
``mesh.n_elements_per_panel`` are exactly the call arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .global_mitc4_assembly import (
    NElements,
    _cluster_points,
    _effective_n_elements_spec,
    _panel_endpoints,
    _panel_mesh,
    _resolve_n_elements,
)
from .shell_inputs_from_section import ShellMeshInputs
from .topology_v2 import build_section_v2


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mitc4PanelMesh:
    """Explicit MITC4 mesh for a single :class:`Panel`.

    The MITC4 element layout used by ``solve_global_coupled_mitc4``
    discretises each panel into ``n_elements`` quads spanning one
    spanwise unit (``L_x = 1`` m). Every quad has 4 nodes laid out in
    two layers of ``n_elements + 1`` arc-length nodes:

        layer 0 (bottom, ``X = 0``)     : indices [0 .. n]
        layer 1 (top,    ``X = L_x=1``) : indices [n .. 2n - 1]

    ``s_nodes`` are the arc-length parameters; ``yz_nodes`` are the
    midline-interpolated cross-section coordinates of those same nodes
    (identical for the bottom and top layers because the strip is
    extruded along the spanwise X axis).
    """

    panel_index: int
    panel_label: str
    kind: str
    n_elements: int
    s_nodes: NDArray[np.float64]
    yz_nodes: NDArray[np.float64]
    elements: list[list[int]]
    thickness_m: float
    arc_length_m: float
    midline_kind_closed: bool


@dataclass(frozen=True)
class Mitc4Cluster:
    """One endpoint-cluster (a junction in the section topology).

    Members are ``(panel_index, end, point_yz)`` triples sharing a
    junction, exactly as classified by
    ``solve_global_coupled_mitc4``'s endpoint-clustering pass.
    """

    cluster_id: int
    point_yz: tuple[float, float]
    members: list[tuple[int, str]]
    panel_labels: list[str]


@dataclass(frozen=True)
class Mitc4SectionMesh:
    """Full per-section MITC4 mesh built from per-subcomponent midlines."""

    panels: list[Any]
    n_elements_per_panel: list[int]
    panel_meshes: list[Mitc4PanelMesh]
    clusters: list[Mitc4Cluster]
    n_total_nodes: int
    n_total_elements: int
    layout_key: str | None
    twist_rad: float
    chord_m: float

    def summary(self) -> dict[str, Any]:
        """Compact dict-of-stats suitable for NDJSON logging or a CLI table."""
        per_kind: dict[str, dict[str, Any]] = {}
        for pm in self.panel_meshes:
            d = per_kind.setdefault(
                pm.kind, {"n_panels": 0, "n_elements": 0, "n_nodes": 0,
                          "arc_length_m": 0.0, "thickness_m_avg": 0.0}
            )
            d["n_panels"] += 1
            d["n_elements"] += pm.n_elements
            d["n_nodes"] += int(pm.yz_nodes.shape[0]) * 2
            d["arc_length_m"] += pm.arc_length_m
            d["thickness_m_avg"] += pm.thickness_m
        for kind, d in per_kind.items():
            n = max(int(d["n_panels"]), 1)
            d["thickness_m_avg"] = float(d["thickness_m_avg"]) / n
        return {
            "layout_key": self.layout_key,
            "twist_rad": float(self.twist_rad),
            "chord_m": float(self.chord_m),
            "n_panels": int(len(self.panels)),
            "n_total_nodes": int(self.n_total_nodes),
            "n_total_elements": int(self.n_total_elements),
            "n_clusters": int(len(self.clusters)),
            "junctions_by_size": {
                k: sum(1 for c in self.clusters if len(c.members) == k)
                for k in sorted({len(c.members) for c in self.clusters})
            },
            "per_kind": per_kind,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interpolate_yz_at_s(p_nodes: NDArray[np.float64],
                         p_s: NDArray[np.float64],
                         s_query: NDArray[np.float64]) -> NDArray[np.float64]:
    """Component-wise linear interpolation of midline coordinates onto ``s_query``."""
    y = np.interp(s_query, p_s, p_nodes[:, 0])
    z = np.interp(s_query, p_s, p_nodes[:, 1])
    return np.column_stack([y, z])


def _kind_from_label(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else "panel"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_mitc4_mesh(
    shell_inputs: ShellMeshInputs,
    *,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    endpoint_tol: float = 1e-6,
    skin_lam: Any | None = None,
    web_lam: Any | None = None,
    cap_lam: Any | None = None,
) -> Mitc4SectionMesh:
    """Build the MITC4 shell mesh for one cross-section.

    Each :class:`SubcomponentMidline` in ``shell_inputs.midlines`` is
    promoted to a :class:`Panel` (via :func:`build_section_v2`) and then
    discretised along its arc length into ``n`` MITC4 quads using the
    same ``_panel_mesh`` helper the solver uses internally. The result
    is a fully-instrumented mesh that can be passed straight to
    :func:`solve_global_coupled_mitc4` or rendered for inspection.

    Parameters
    ----------
    shell_inputs
        Per-subcomponent midline payload from
        :func:`build_shell_mesh_inputs`.
    n_elements_per_panel
        Element-count specification accepted by the solver: an ``int``
        (uniform), ``Sequence[int]`` (per panel, falls back to ``10``),
        or ``Mapping[int, int]`` (panel-index → count, falls back to
        ``10``).
    target_element_length_m
        Optional target physical element length (metres). When set to a
        positive finite value together with the default uniform count
        ``n_elements_per_panel=10``, each panel's element count is
        ``max(1, round(arc_length_m / target))``.  A uniform integer other
        than ``10``, or any per-panel ``Mapping`` / ``Sequence``, overrides
        this rule (same semantics as :func:`solve_global_coupled_mitc4`).
    endpoint_tol
        Tolerance (metres) for the geometric clustering of panel
        endpoints into junctions. Must match the value passed to the
        solver later, otherwise the cluster topology shown here will
        not match the assembled DOF map.
    skin_lam, web_lam, cap_lam
        Optional laminate overrides forwarded to
        :func:`build_section_v2`.
    """
    panels, _booms, _webs_geom, _n_cells = build_section_v2(
        shell_inputs,
        skin_lam=skin_lam,
        web_lam=web_lam,
        cap_lam=cap_lam,
    )
    n_elem_spec = _effective_n_elements_spec(
        panels, n_elements_per_panel, target_element_length_m
    )

    panel_meshes: list[Mitc4PanelMesh] = []
    n_elements_resolved: list[int] = []
    n_total_nodes = 0
    n_total_elements = 0
    endpoints: list[tuple[int, str, NDArray[np.float64]]] = []

    label_to_kind = {f"{m.kind}:{m.label}": m.kind for m in shell_inputs.midlines}
    label_to_closed = {f"{m.kind}:{m.label}": bool(m.closed) for m in shell_inputs.midlines}
    label_to_thickness = {f"{m.kind}:{m.label}": float(m.thickness_m) for m in shell_inputs.midlines}

    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        nodes_yz = np.asarray(p.nodes, dtype=float)
        n_elem = _resolve_n_elements(n_elem_spec, pi)
        if s_panel.size < 2:
            n_elements_resolved.append(0)
            panel_meshes.append(Mitc4PanelMesh(
                panel_index=pi, panel_label=str(p.label), kind=_kind_from_label(p.label),
                n_elements=0, s_nodes=np.empty(0), yz_nodes=np.empty((0, 2)),
                elements=[], thickness_m=label_to_thickness.get(p.label, float(p.lam.t)),
                arc_length_m=0.0,
                midline_kind_closed=label_to_closed.get(p.label, False),
            ))
            continue
        s_nodes, elems = _panel_mesh(
            s_panel,
            int(n_elem),
            panel_label=str(getattr(p, "label", "") or ""),
            nodes_yz=nodes_yz,
        )
        yz_nodes = _interpolate_yz_at_s(nodes_yz, s_panel, s_nodes)
        n_elem_actual = int(len(elems))
        n_elements_resolved.append(n_elem_actual)

        n_local_nodes = 2 * int(s_nodes.shape[0])
        n_total_nodes += n_local_nodes
        n_total_elements += len(elems)

        ep = _panel_endpoints(nodes_yz)
        endpoints.append((pi, "start", ep["start"]))
        endpoints.append((pi, "end", ep["end"]))

        panel_meshes.append(Mitc4PanelMesh(
            panel_index=pi,
            panel_label=str(p.label),
            kind=label_to_kind.get(p.label, _kind_from_label(p.label)),
            n_elements=n_elem_actual,
            s_nodes=s_nodes,
            yz_nodes=yz_nodes,
            elements=[list(map(int, e)) for e in elems],
            thickness_m=label_to_thickness.get(p.label, float(p.lam.t)),
            arc_length_m=float(s_panel[-1] - s_panel[0]),
            midline_kind_closed=label_to_closed.get(p.label, False),
        ))

    raw_clusters = _cluster_points(endpoints, endpoint_tol)
    clusters: list[Mitc4Cluster] = []
    for cid, c in enumerate(raw_clusters):
        members = [(int(pi), str(end)) for pi, end, _pt in c]
        labels = [str(panels[pi].label) for pi, _e in members]
        pt = c[0][2]
        clusters.append(Mitc4Cluster(
            cluster_id=int(cid),
            point_yz=(float(pt[0]), float(pt[1])),
            members=members,
            panel_labels=labels,
        ))

    return Mitc4SectionMesh(
        panels=panels,
        n_elements_per_panel=n_elements_resolved,
        panel_meshes=panel_meshes,
        clusters=clusters,
        n_total_nodes=int(n_total_nodes),
        n_total_elements=int(n_total_elements),
        layout_key=shell_inputs.layout_key,
        twist_rad=float(shell_inputs.twist_rad),
        chord_m=float(shell_inputs.chord_m),
    )


__all__ = [
    "Mitc4PanelMesh",
    "Mitc4Cluster",
    "Mitc4SectionMesh",
    "build_mitc4_mesh",
]
