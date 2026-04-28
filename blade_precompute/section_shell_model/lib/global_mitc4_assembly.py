from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mitc4_element import (
    mitc4_edge_resultants,
    mitc4_edge_shear_traction_integrated,
    mitc4_resultants,
    mitc4_stiffness,
)
from .types import FieldProvenance, ProvenanceKind, ShellPanelResultants

_NDOF_NODE = 5
_U_X = 0
_U_S = 1
_W = 2
_BETA_S = 3
_BETA_X = 4

# Cap-only: minimum polyline deflection (rad) at an interior vertex to count as
# a "kink" and force an MITC4 node at the corresponding ``s`` value.
_PANEL_MESH_KINK_DEFLECTION_RAD: float = float(np.deg2rad(0.35))

# Cap-only: if ``s_panel[j]`` is farther than this fraction of one nominal
# linspace step from the nearest ``s_lin`` sample, add it to the discretisation
# (picks up smooth junctions that are not kinks).
_PANEL_MESH_S_FAR_ALPHA: float = 0.3


def _panel_mesh_kind(panel_label: str) -> str:
    if ":" in panel_label:
        return str(panel_label.split(":", 1)[0])
    return ""


def _merge_sorted_s_unique_eps(s_sorted: np.ndarray, *, eps: float) -> np.ndarray:
    """Sort, then drop values within ``eps`` of the previous kept sample."""
    s_sorted = np.asarray(s_sorted, dtype=float).ravel()
    if s_sorted.size == 0:
        return s_sorted
    s_sorted = np.sort(s_sorted)
    out: list[float] = [float(s_sorted[0])]
    for x in s_sorted[1:]:
        xf = float(x)
        if xf - out[-1] > eps:
            out.append(xf)
    return np.asarray(out, dtype=float)


def _cap_s_extra_kink_and_far(
    s_panel: np.ndarray,
    s_lin: np.ndarray,
    nodes_yz: np.ndarray,
    *,
    span: float,
    n_el: int,
    deflection_min_rad: float = _PANEL_MESH_KINK_DEFLECTION_RAD,
    far_alpha: float = _PANEL_MESH_S_FAR_ALPHA,
) -> list[float]:
    """Pick extra ``s`` abscissae for cap strips: kinks + far-from-linspace samples."""
    s_panel = np.asarray(s_panel, dtype=float).ravel()
    nodes = np.asarray(nodes_yz, dtype=float)
    n = int(s_panel.size)
    extra: set[float] = set()
    nominal = span / max(float(n_el), 1.0)
    if nominal <= 0.0:
        nominal = float(span) if float(span) > 0.0 else 1.0
    far_tau = float(far_alpha) * nominal

    if nodes.shape[0] >= 3 and n == nodes.shape[0]:
        for j in range(1, n - 1):
            t1 = nodes[j] - nodes[j - 1]
            t2 = nodes[j + 1] - nodes[j]
            n1 = float(np.linalg.norm(t1))
            n2 = float(np.linalg.norm(t2))
            if n1 < 1e-30 or n2 < 1e-30:
                continue
            c = float(np.dot(t1, t2) / (n1 * n2))
            c = float(np.clip(c, -1.0, 1.0))
            turn = float(np.arccos(c))
            deflection = float(np.pi) - turn
            if deflection > float(deflection_min_rad):
                extra.add(float(s_panel[j]))

    for j in range(n):
        sj = float(s_panel[j])
        dmin = float(np.min(np.abs(s_lin - sj)))
        if dmin > far_tau:
            extra.add(sj)

    return [float(s) for s in extra]


def _panel_mesh(
    s_panel: np.ndarray,
    n_elements: int,
    *,
    panel_label: str = "",
    nodes_yz: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """Equi-arc (uniform in cumulative ``s``) MITC4 nodes, with selective cap-only knots.

    ``s`` is arc length along the piecewise linear ``Panel`` midline, so
    ``linspace(s_min, s_max)`` is equi-spaced in physical length. **Web** and
    **skin** use that grid only. **Cap** panels include all cap polyline knots
    so Class-A/B/C cap node locations (skin-coincident ends, web intersections,
    and skin-ray-resampled interiors) are represented in the MITC4 mesh.
    """
    s_panel = np.asarray(s_panel, dtype=float).ravel()
    s_min, s_max = float(s_panel.min()), float(s_panel.max())
    n_el = max(1, int(n_elements))
    s_lin = np.linspace(s_min, s_max, n_el + 1)
    span = s_max - s_min
    eps = max(1e-15 * (span if span > 0.0 else 1.0), 1e-18)
    kind = _panel_mesh_kind(panel_label)

    if kind in ("web", "skin", ""):
        s_nodes = s_lin
    elif kind == "cap":
        # Cap nodes must represent Class A/B/C shell-knot locations exactly.
        s_nodes = _merge_sorted_s_unique_eps(s_panel, eps=eps)
    else:
        s_nodes = s_lin

    n = len(s_nodes)
    elems: list[list[int]] = []
    for i in range(n - 1):
        i0 = i
        i1 = i + 1
        i2 = i + 1 + n
        i3 = i + n
        elems.append([i0, i1, i2, i3])
    return s_nodes, elems


def _panel_endpoints(nodes_yz: np.ndarray) -> dict[str, np.ndarray]:
    return {"start": np.asarray(nodes_yz[0], dtype=float), "end": np.asarray(nodes_yz[-1], dtype=float)}


def _panel_end_tangent(nodes_yz: np.ndarray, which: str) -> np.ndarray:
    if len(nodes_yz) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if which == "start":
        t = np.asarray(nodes_yz[1] - nodes_yz[0], dtype=float)
    else:
        t = np.asarray(nodes_yz[-1] - nodes_yz[-2], dtype=float)
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return t / n


def _cluster_points(points: list[tuple[int, str, np.ndarray]], tol: float) -> list[list[tuple[int, str, np.ndarray]]]:
    clusters: list[list[tuple[int, str, np.ndarray]]] = []
    for p in points:
        placed = False
        for c in clusters:
            if float(np.linalg.norm(p[2] - c[0][2])) <= tol:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters


def _dof(node_idx: int, local: int) -> int:
    return node_idx * _NDOF_NODE + local


def _normal_2d(t: np.ndarray) -> np.ndarray:
    """90° CCW rotation of a 2-D tangent unit vector → panel outward normal in (Y,Z)."""
    return np.array([-float(t[1]), float(t[0])], dtype=float)


def _cluster_is_collinear(
    cluster: list[tuple[int, str, Any]],
    panels: list[Any],
    tol_deg: float = 1.0,
) -> bool:
    """Return True if all panel tangents in the cluster are within tol_deg of each other."""
    tangents: list[np.ndarray] = []
    for pi, which, _ in cluster:
        nodes = np.asarray(getattr(panels[pi], "nodes"), dtype=float)
        tangents.append(_panel_end_tangent(nodes, which))
    if len(tangents) < 2:
        return True
    tol_rad = tol_deg * np.pi / 180.0
    t0 = tangents[0]
    for t in tangents[1:]:
        cos_a = float(np.clip(np.dot(t0, t), -1.0, 1.0))
        angle = float(np.arccos(abs(cos_a)))
        if angle > tol_rad:
            return False
    return True


def _any_noncollinear_cluster(
    clusters: list[list[tuple[int, str, Any]]],
    panels: list[Any],
    tol_deg: float = 1.0,
) -> bool:
    """Return True if any cluster in the list is non-collinear."""
    return any(
        not _cluster_is_collinear(c, panels, tol_deg)
        for c in clusters
        if len(c) >= 2
    )


# ---------------------------------------------------------------------------
# Per-panel mesh-count resolution.
# ---------------------------------------------------------------------------

NElements = Union[int, Mapping[int, int], Sequence[int]]
"""
Flexible element-count specification for :func:`solve_global_coupled_mitc4`.

- ``int``: same count for every panel.
- ``Mapping[int, int]``: panel-index → element count; falls back to *default=10*.
- ``Sequence[int]``: element count per panel index; falls back to *default=10*
  if the index is out of range.
"""


def _resolve_n_elements(spec: NElements, pi: int, default: int = 10) -> int:
    """Return the element count for panel ``pi`` given a flexible spec."""
    if isinstance(spec, numbers.Integral):
        return int(spec)
    if isinstance(spec, Mapping):
        return int(spec.get(pi, default))
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return int(spec[pi]) if pi < len(spec) else default
    raise TypeError(f"n_elements_per_panel must be int, Mapping, or Sequence; got {type(spec)}")


def _effective_n_elements_spec(
    panels: list[Any],
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
) -> NElements:
    """Combine uniform / per-panel counts with optional distance-based sizing.

    When ``target_element_length_m`` is a positive finite value, per-panel
    counts are ``max(1, round(arc_length_m / target))`` using each panel's
    cumulative ``p.s`` span, **unless** an explicit per-panel spec is already
    provided (non-``int`` :class:`NElements`, or an ``int`` other than the
    default ``10``), in which case that specification wins.
    """
    if (
        target_element_length_m is None
        or not np.isfinite(target_element_length_m)
        or float(target_element_length_m) <= 0.0
    ):
        return n_elements_per_panel
    if isinstance(n_elements_per_panel, numbers.Integral) and int(n_elements_per_panel) != 10:
        return n_elements_per_panel
    if isinstance(n_elements_per_panel, Mapping):
        return n_elements_per_panel
    if isinstance(n_elements_per_panel, Sequence) and not isinstance(
        n_elements_per_panel, (str, bytes)
    ):
        return n_elements_per_panel

    tgt = float(target_element_length_m)
    mapping: dict[int, int] = {}
    for pi, p in enumerate(panels):
        s_p = np.asarray(getattr(p, "s", None), dtype=float)
        if s_p.size < 2:
            mapping[pi] = 1
        else:
            arc = float(s_p[-1] - s_p[0])
            mapping[pi] = max(1, int(round(arc / tgt)))
    return mapping


# Type alias for the unified multi-master constraint format.
# Each slave DOF maps to a list of (master_dof, coefficient) pairs so that
# q_slave = sum_k coeff_k * q_master_k.
_Constraints = dict[int, list[tuple[int, float]]]


def _build_transform_constraints(
    panel_maps: list[_PanelGlobalMap],
    panels: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
) -> _Constraints:
    """
    Build scalar-ratio MPC constraints (legacy scalar path, used by ``"transformed"`` mode).

    Each slave DOF is a scalar multiple of one master DOF.  This is retained for
    backward compatibility; the physically correct block constraints are in
    ``_build_basis_transform_constraints``.
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 2:
            continue
        pi_m, end_m, _ = cluster[0]
        pm_m = panel_maps[pi_m]
        n_m = len(pm_m.s_nodes)
        row_m = 0 if end_m == "start" else n_m - 1
        t_m = _panel_end_tangent(np.asarray(getattr(panels[pi_m], "nodes"), dtype=float), end_m)
        t_ref = t_m
        c_m = float(np.dot(t_m, t_ref))
        c_m = 1.0 if abs(c_m) < 1e-12 else c_m
        for pi_s, end_s, _ in cluster[1:]:
            pm_s = panel_maps[pi_s]
            n_s = len(pm_s.s_nodes)
            row_s = 0 if end_s == "start" else n_s - 1
            t_s = _panel_end_tangent(np.asarray(getattr(panels[pi_s], "nodes"), dtype=float), end_s)
            c_s = float(np.dot(t_s, t_ref))
            c_s = 1.0 if abs(c_s) < 1e-12 else c_s
            coeff_s = c_m / c_s
            for off_m, off_s in ((row_m, row_s), (row_m + n_m, row_s + n_s)):
                gn_m = pm_m.global_nodes[off_m]
                gn_s = pm_s.global_nodes[off_s]
                constraints[_dof(gn_s, _U_X)] = [(_dof(gn_m, _U_X), 1.0)]
                constraints[_dof(gn_s, _U_S)] = [(_dof(gn_m, _U_S), coeff_s)]
                constraints[_dof(gn_s, _W)] = [(_dof(gn_m, _W), 1.0)]
                constraints[_dof(gn_s, _BETA_S)] = [(_dof(gn_m, _BETA_S), coeff_s)]
                constraints[_dof(gn_s, _BETA_X)] = [(_dof(gn_m, _BETA_X), 1.0)]
    return constraints


def _build_basis_transform_constraints(
    panel_maps: list[_PanelGlobalMap],
    panels: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
) -> _Constraints:
    """
    Build physically correct 2-D kinematic compatibility MPCs.

    For each junction cluster the master panel (first in cluster) defines a local
    (u_s, w) basis.  All slave panels' endpoint DOFs are expressed in terms of the
    master's DOFs via the 2-D rotation:

      u_s_s = (ŝ_s·ŝ_m) u_s_m + (ŝ_s·n̂_m) w_m
      w_s   = (n̂_s·ŝ_m) u_s_m + (n̂_s·n̂_m) w_m

    where n̂ = CCW_90(ŝ) is the panel's out-of-plane direction in the (Y,Z) plane.

    Spanwise (u_x) and drilling (β_x) DOFs are shared 1:1 since X is common to all
    panels.  The tangential rotation β_s is projected by (ŝ_s·ŝ_m).
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 2:
            continue
        pi_m, end_m, _ = cluster[0]
        pm_m = panel_maps[pi_m]
        n_m = len(pm_m.s_nodes)
        row_m = 0 if end_m == "start" else n_m - 1
        nodes_m = np.asarray(getattr(panels[pi_m], "nodes"), dtype=float)
        t_m = _panel_end_tangent(nodes_m, end_m)
        n_m_hat = _normal_2d(t_m)

        for pi_s, end_s, _ in cluster[1:]:
            pm_s = panel_maps[pi_s]
            n_s = len(pm_s.s_nodes)
            row_s = 0 if end_s == "start" else n_s - 1
            nodes_s = np.asarray(getattr(panels[pi_s], "nodes"), dtype=float)
            t_s = _panel_end_tangent(nodes_s, end_s)
            n_s_hat = _normal_2d(t_s)

            # 2×2 rotation matrix entries: R = [[ts_tm, ts_nm],[ns_tm, ns_nm]]
            # q_slave = R * q_master   for (u_s, w) block.
            ts_tm = float(np.dot(t_s, t_m))
            ts_nm = float(np.dot(t_s, n_m_hat))
            ns_tm = float(np.dot(n_s_hat, t_m))
            ns_nm = float(np.dot(n_s_hat, n_m_hat))

            for off_m, off_s in ((row_m, row_s), (row_m + n_m, row_s + n_s)):
                gn_m = pm_m.global_nodes[off_m]
                gn_s = pm_s.global_nodes[off_s]

                # Spanwise: u_x is common to all panels (no rotation of X-axis).
                constraints[_dof(gn_s, _U_X)] = [(_dof(gn_m, _U_X), 1.0)]

                # 2-D in-plane displacement: u_s_s = ts_tm*u_s_m + ts_nm*w_m
                us_terms = [(_dof(gn_m, _U_S), ts_tm)]
                if abs(ts_nm) > 1e-12:
                    us_terms.append((_dof(gn_m, _W), ts_nm))
                constraints[_dof(gn_s, _U_S)] = us_terms

                # 2-D out-of-plane: w_s = ns_tm*u_s_m + ns_nm*w_m
                w_terms = [(_dof(gn_m, _U_S), ns_tm)]
                if abs(ns_nm) > 1e-12:
                    w_terms.append((_dof(gn_m, _W), ns_nm))
                constraints[_dof(gn_s, _W)] = w_terms

                # Rotation about span: β_x is common.
                constraints[_dof(gn_s, _BETA_X)] = [(_dof(gn_m, _BETA_X), 1.0)]

                # Tangential rotation: project β_s by the tangent dot product.
                # β_s represents rotation about the panel tangent; its (Y,Z) component
                # along the slave tangent is (ŝ_s·ŝ_m)*β_s_m.
                constraints[_dof(gn_s, _BETA_S)] = [(_dof(gn_m, _BETA_S), ts_tm)]

    return constraints


def _build_cluster_basis_constraints(
    panel_maps: list["_PanelGlobalMap"],
    panels: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
    endpoint_cluster_node: dict[tuple[int, str, str], int],
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int],
) -> _Constraints:
    """
    Build transformed-basis constraints against cluster reference DOFs.

    This avoids panel-to-panel master/slave chains and makes BC propagation robust:
    all panel endpoint DOFs are slaves, cluster reference DOFs are masters.
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 1:
            continue
        # Reference tangent from principal direction of endpoint tangents.
        t_mats: list[np.ndarray] = []
        for pi, which, _ in cluster:
            nodes = np.asarray(getattr(panels[pi], "nodes"), dtype=float)
            t = _panel_end_tangent(nodes, which)
            t_mats.append(np.outer(t, t))
        if t_mats:
            M = np.sum(np.stack(t_mats, axis=0), axis=0)
            evals, evecs = np.linalg.eigh(M)
            t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
            if float(np.linalg.norm(t_ref)) < 1e-12:
                t_ref = np.array([1.0, 0.0], dtype=float)
        else:
            t_ref = np.array([1.0, 0.0], dtype=float)
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        t_ref = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)
        n_ref = _normal_2d(t_ref)

        for pi, end, _ in cluster:
            pm = panel_maps[pi]
            n = len(pm.s_nodes)
            if n == 0:
                continue
            row = 0 if end == "start" else n - 1
            nodes = np.asarray(getattr(panels[pi], "nodes"), dtype=float)
            t_p = _panel_end_tangent(nodes, end)
            n_p = _normal_2d(t_p)
            ts = float(np.dot(t_p, t_ref))
            tn = float(np.dot(t_p, n_ref))
            ns = float(np.dot(n_p, t_ref))
            nn = float(np.dot(n_p, n_ref))

            for off, layer in ((row, "bottom"), (row + n, "top")):
                gn_p = pm.global_nodes[off]
                gn_c_main = endpoint_cluster_node[(pi, end, layer)]
                gn_c_rot = endpoint_cluster_rot_node[(pi, end, layer)]
                constraints[_dof(gn_p, _U_X)] = [(_dof(gn_c_main, _U_X), 1.0)]
                us_terms = [(_dof(gn_c_main, _U_S), ts)]
                if abs(tn) > 1e-12:
                    us_terms.append((_dof(gn_c_main, _W), tn))
                constraints[_dof(gn_p, _U_S)] = us_terms
                w_terms = [(_dof(gn_c_main, _U_S), ns)]
                if abs(nn) > 1e-12:
                    w_terms.append((_dof(gn_c_main, _W), nn))
                constraints[_dof(gn_p, _W)] = w_terms
                constraints[_dof(gn_p, _BETA_X)] = [(_dof(gn_c_main, _BETA_X), 1.0)]
                beta_terms = [(_dof(gn_c_main, _BETA_S), ts)]
                if abs(tn) > 1e-12:
                    beta_terms.append((_dof(gn_c_rot, _BETA_S), tn))
                constraints[_dof(gn_p, _BETA_S)] = beta_terms
    return constraints


@dataclass
class _PanelGlobalMap:
    s_nodes: np.ndarray
    elements: list[list[int]]
    global_nodes: list[int]  # local node -> global node id
    panel_label: str
    panel_index: int


@dataclass
class GlobalNodeMeta:
    """Per-node geometric metadata for K7 cross-section stiffness extraction.

    All arrays are indexed by global node ID (0 … n_nodes-1).  Nodes that are
    MPC virtual cluster reference nodes (``transformed_basis`` mode) have NaN
    in ``yz``, ``tangent_yz``, and ``arc_length_s``.
    """
    n_nodes: int                  # total global node count
    yz: np.ndarray                # (n_nodes, 2) — (y, z) cross-section coords [m]
    is_top: np.ndarray            # (n_nodes,) bool — top layer (x=L_x) vs bottom (x=0)
    tangent_yz: np.ndarray        # (n_nodes, 2) — (ty, tz) unit tangent along contour
    arc_length_s: np.ndarray      # (n_nodes,) — contour arc-length s [m]


def _collect_global_node_meta(
    panel_maps: list[_PanelGlobalMap],
    panels: list[Any],
    n_total_nodes: int,
) -> GlobalNodeMeta:
    """Build per-node geometric metadata from the assembled panel maps."""
    yz = np.full((n_total_nodes, 2), np.nan)
    is_top = np.zeros(n_total_nodes, dtype=bool)
    tangent_yz = np.full((n_total_nodes, 2), np.nan)
    arc_s = np.full(n_total_nodes, np.nan)

    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p = panels[pi]
        nodes_yz_p = np.asarray(getattr(p, "nodes"), dtype=float)
        s_panel = np.asarray(p.s, dtype=float)
        s_nodes = pm.s_nodes
        n = len(s_nodes)

        y_m = np.interp(s_nodes, s_panel, nodes_yz_p[:, 0])
        z_m = np.interp(s_nodes, s_panel, nodes_yz_p[:, 1])

        if len(nodes_yz_p) >= 2:
            ty_p = np.gradient(nodes_yz_p[:, 0], s_panel)
            tz_p = np.gradient(nodes_yz_p[:, 1], s_panel)
            ty_m = np.interp(s_nodes, s_panel, ty_p)
            tz_m = np.interp(s_nodes, s_panel, tz_p)
            nrm = np.maximum(np.hypot(ty_m, tz_m), 1e-30)
            ty_m = ty_m / nrm
            tz_m = tz_m / nrm
        else:
            ty_m = np.zeros(n)
            tz_m = np.ones(n)

        for li in range(2 * n):
            si = li if li < n else li - n
            gn = pm.global_nodes[li]
            if gn < 0:
                continue
            if np.isnan(yz[gn, 0]):
                yz[gn] = [y_m[si], z_m[si]]
                tangent_yz[gn] = [ty_m[si], tz_m[si]]
                arc_s[gn] = s_nodes[si]
            is_top[gn] = (li >= n)

    return GlobalNodeMeta(n_nodes=n_total_nodes, yz=yz, is_top=is_top, tangent_yz=tangent_yz, arc_length_s=arc_s)


def solve_global_coupled_mitc4(
    panels: list[Any],
    Nx_panels: list[np.ndarray],
    Nxy_panels: list[np.ndarray],
    *,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    endpoint_tol: float = 1e-6,
    bc_mode: str = "full_clamp",
    interface_constraint_mode: str = "shared",
    enforce_traction_balance_at_cusp: bool = False,
    traction_penalty_alpha: float = 1e-2,
    return_assembly_data: bool = False,
) -> tuple:
    """
    Assemble and solve all panel strips in one global system with shared endpoint nodes.

    Parameters
    ----------
    target_element_length_m
        When set to a positive finite value (and ``n_elements_per_panel`` is the
        default uniform ``10``), element counts are derived from each panel's
        polyline arc length in ``p.s``.  Explicit per-panel ``Mapping`` /
        ``Sequence`` counts, or a uniform integer other than ``10``, override
        this rule.
    """
    panel_maps: list[_PanelGlobalMap] = []
    endpoints: list[tuple[int, str, np.ndarray]] = []
    n_elem_spec = _effective_n_elements_spec(
        panels, n_elements_per_panel, target_element_length_m
    )

    # Build local strip meshes per panel.
    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        if len(s_panel) < 2:
            panel_maps.append(_PanelGlobalMap(np.array([]), [], [], str(getattr(p, "label", f"panel_{pi}")), pi))
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_nodes, elems = _panel_mesh(
            s_panel,
            _resolve_n_elements(n_elem_spec, pi),
            panel_label=str(getattr(p, "label", "") or ""),
            nodes_yz=nodes_yz,
        )
        n_local_nodes = 2 * len(s_nodes)
        global_nodes = [-1] * n_local_nodes
        panel_maps.append(_PanelGlobalMap(s_nodes, elems, global_nodes, str(getattr(p, "label", f"panel_{pi}")), pi))
        ep = _panel_endpoints(nodes_yz)
        endpoints.append((pi, "start", ep["start"]))
        endpoints.append((pi, "end", ep["end"]))

    # Interface topology from endpoint clusters.
    clusters = _cluster_points(endpoints, endpoint_tol)
    endpoint_cluster_index: dict[tuple[int, str], int] = {}
    for cluster_id, c in enumerate(clusters):
        for pi, which, _ in c:
            endpoint_cluster_index[(pi, which)] = cluster_id

    # Collinearity flags per cluster (used for diagnostics and optional penalty).
    cluster_is_collinear_flags: dict[int, bool] = {
        cid: _cluster_is_collinear(c, panels, tol_deg=5.0)
        for cid, c in enumerate(clusters)
    }

    # effective_mode reflects what is actually used internally. "shared_rotated"
    # is an alias for "transformed_basis" topology + MPCs; when explicitly chosen,
    # it is stored as "shared_rotated" in diagnostics to distinguish from
    # "transformed_basis" (same internals, same label for now).
    effective_mode = interface_constraint_mode

    next_global_node = 0
    endpoint_cluster_node: dict[tuple[int, str, str], int] = {}
    endpoint_cluster_rot_node: dict[tuple[int, str, str], int] = {}
    cluster_layer_node: dict[tuple[int, str], int] = {}
    cluster_layer_rot_node: dict[tuple[int, str], int] = {}
    if interface_constraint_mode == "shared":
        endpoint_global_node: dict[tuple[int, str, str], int] = {}
        for c in clusters:
            g_bottom = next_global_node
            g_top = next_global_node + 1
            next_global_node += 2
            for pi, which, _ in c:
                endpoint_global_node[(pi, which, "bottom")] = g_bottom
                endpoint_global_node[(pi, which, "top")] = g_top
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            n = len(pm.s_nodes)
            pm.global_nodes[0] = endpoint_global_node[(pm.panel_index, "start", "bottom")]
            pm.global_nodes[n - 1] = endpoint_global_node[(pm.panel_index, "end", "bottom")]
            pm.global_nodes[n] = endpoint_global_node[(pm.panel_index, "start", "top")]
            pm.global_nodes[2 * n - 1] = endpoint_global_node[(pm.panel_index, "end", "top")]
            for li in range(1, n - 1):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1
            for li in range(n + 1, 2 * n - 1):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1
    elif interface_constraint_mode in ("transformed_basis", "shared_rotated") or effective_mode == "shared_rotated":
        # Allocate two cluster reference nodes per layer (bottom/top):
        # - main node carries {U_X, U_S, W, BETA_X, BETA_S(t_ref)}
        # - rot node reuses BETA_S slot as BETA_S(n_ref) = beta_n_ref
        # This block handles: transformed_basis, shared_rotated (explicit),
        # and shared auto-promoted to shared_rotated (effective_mode).
        for cid, c in enumerate(clusters):
            for layer in ("bottom", "top"):
                cluster_layer_node[(cid, layer)] = next_global_node
                next_global_node += 1
                cluster_layer_rot_node[(cid, layer)] = next_global_node
                next_global_node += 1
            for pi, which, _ in c:
                endpoint_cluster_node[(pi, which, "bottom")] = cluster_layer_node[(cid, "bottom")]
                endpoint_cluster_node[(pi, which, "top")] = cluster_layer_node[(cid, "top")]
                endpoint_cluster_rot_node[(pi, which, "bottom")] = cluster_layer_rot_node[(cid, "bottom")]
                endpoint_cluster_rot_node[(pi, which, "top")] = cluster_layer_rot_node[(cid, "top")]
        # Panel nodes remain unique; endpoint compatibility enforced by MPC.
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            for li in range(2 * len(pm.s_nodes)):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1
    else:
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            for li in range(2 * len(pm.s_nodes)):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1

    n_gdof = next_global_node * _NDOF_NODE
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    f_global = np.zeros(n_gdof)
    panel_load_totals: list[dict] = []

    # Assemble K and f.
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            panel_load_totals.append({})
            continue
        p = panels[pi]
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        # Build ABD per panel from laminate
        from examples.section_stress_model.lib.laminate_clpt import (  # type: ignore[import-untyped]
            abd_stack,
        )
        A_mat, B_mat, D_mat = abd_stack(p.lam.build_plies())
        ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])
        thickness = float(p.lam.t)
        nu = float(p.lam.nu)
        E_p = float(p.lam.E)
        G_eff = E_p / (2.0 * (1.0 + nu))

        s_nodes = pm.s_nodes
        Nx_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nx_panels[pi], dtype=float))
        Nxy_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nxy_panels[pi], dtype=float))
        fx_total = 0.0
        fs_total = 0.0

        for e in pm.elements:
            i0, i1, i2, i3 = e
            L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
            if L_s < 1e-30:
                continue
            Ke = mitc4_stiffness(L_s, 1.0, ABD, thickness, G_eff=G_eff)
            local_nodes = [i0, i1, i2, i3]
            g_dofs: list[int] = []
            for ln in local_nodes:
                gn = pm.global_nodes[ln]
                for d in range(_NDOF_NODE):
                    g_dofs.append(_dof(gn, d))
            for a, ga in enumerate(g_dofs):
                for b, gb in enumerate(g_dofs):
                    rows.append(ga)
                    cols.append(gb)
                    vals.append(float(Ke[a, b]))

            Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
            Nxy_avg = 0.5 * (Nxy_nodes[i0] + Nxy_nodes[i1])
            f_node_x = Nx_avg * L_s / 2.0
            f_node_s = Nxy_avg * L_s / 2.0
            for ln in (i2, i3):
                gn = pm.global_nodes[ln]
                f_global[_dof(gn, _U_X)] += f_node_x
                f_global[_dof(gn, _U_S)] += f_node_s
            fx_total += 2.0 * f_node_x
            fs_total += 2.0 * f_node_s

        panel_load_totals.append({"Fx_total": fx_total, "Fs_total": fs_total})

        # Donnell curvature correction: q_n(s) = Nx(s) · κ(s) [N/m²]
        # Mirror the per-panel _assemble_curvature_loads + _panel_curvature logic so
        # the global solve sees the same constitutive load path as solve_panel_mitc4.
        if len(nodes_yz) >= 3:
            # Curvature κ at each s-panel node (tangent-angle derivative).
            y_p, z_p = nodes_yz[:, 0], nodes_yz[:, 1]
            s_panel_full = np.asarray(p.s, dtype=float)
            dy_p, dz_p = np.diff(y_p), np.diff(z_p)
            ds_p = np.maximum(np.hypot(dy_p, dz_p), 1e-30)
            theta_p = np.arctan2(dz_p, dy_p)
            kappa_p = np.zeros(len(nodes_yz))
            for ii in range(1, len(nodes_yz) - 1):
                dth = theta_p[ii] - theta_p[ii - 1]
                dth = (dth + np.pi) % (2 * np.pi) - np.pi
                arc = 0.5 * (ds_p[ii - 1] + ds_p[ii])
                kappa_p[ii] = dth / arc if arc > 1e-30 else 0.0
            if len(kappa_p) > 1:
                kappa_p[0] = kappa_p[1]
                kappa_p[-1] = kappa_p[-2]
            # Interpolate κ onto the solver s_nodes.
            kappa_nodes = np.interp(s_nodes, s_panel_full, kappa_p)
            # Consistent nodal load for each element.
            for e in pm.elements:
                i0, i1, i2, i3 = e
                L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
                if L_s < 1e-30:
                    continue
                Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
                kappa_avg = 0.5 * (kappa_nodes[i0] + kappa_nodes[i1])
                q_n = Nx_avg * kappa_avg          # [N/m²]
                f_node_w = q_n * L_s * 1.0 / 4.0  # L_x = 1 for unit slice
                for ln in (i0, i1, i2, i3):
                    gn = pm.global_nodes[ln]
                    f_global[_dof(gn, _W)] += f_node_w

    K_global = sp.coo_matrix((vals, (rows, cols)), shape=(n_gdof, n_gdof)).tocsr()

    # Build compatibility constraints based on effective mode.
    # "shared_rotated" (whether explicit or auto-promoted from "shared") uses the
    # same 6-DOF cluster-basis MPCs as "transformed_basis".
    use_cluster_basis = effective_mode in ("transformed_basis", "shared_rotated") or \
                        interface_constraint_mode in ("transformed_basis", "shared_rotated")
    if not use_cluster_basis and interface_constraint_mode == "shared":
        constraints: _Constraints = {}
    elif use_cluster_basis:
        constraints = _build_cluster_basis_constraints(
            panel_maps,
            panels,
            clusters,
            endpoint_cluster_node,
            endpoint_cluster_rot_node,
        )
        # Rot nodes only carry beta_n_ref in their BETA_S slot; tie all other
        # translational/spanwise-rotation DOFs to the corresponding main node so
        # they do not introduce free rigid mechanisms.
        for key, gn_rot in cluster_layer_rot_node.items():
            gn_main = cluster_layer_node[key]
            constraints[_dof(gn_rot, _U_X)] = [(_dof(gn_main, _U_X), 1.0)]
            constraints[_dof(gn_rot, _U_S)] = [(_dof(gn_main, _U_S), 1.0)]
            constraints[_dof(gn_rot, _W)] = [(_dof(gn_main, _W), 1.0)]
            constraints[_dof(gn_rot, _BETA_X)] = [(_dof(gn_main, _BETA_X), 1.0)]
    else:
        # "transformed" mode: legacy scalar-ratio constraints.
        constraints = _build_transform_constraints(panel_maps, panels, clusters)

    # Global supports: minimal RBM + optional legacy endpoint support.
    fixed: set[int] = set()
    if next_global_node > 1:
        fixed.add(_dof(0, _U_X))
        fixed.add(_dof(0, _U_S))
        fixed.add(_dof(1, _U_X))
    if use_cluster_basis:
        fixed_cluster_layers: set[tuple[int, str]] = set()
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            n = len(pm.s_nodes)
            if bc_mode == "full_clamp":
                support_nodes = (0, n - 1, n, 2 * n - 1)
            else:
                support_nodes = (0, n - 1)
            for ln in support_nodes:
                if ln == 0:
                    which, layer = "start", "bottom"
                elif ln == n - 1:
                    which, layer = "end", "bottom"
                elif ln == n:
                    which, layer = "start", "top"
                elif ln == 2 * n - 1:
                    which, layer = "end", "top"
                else:
                    continue
                cid = endpoint_cluster_index.get((pm.panel_index, which), -1)
                if cid < 0:
                    continue
                key = (cid, layer)
                if key in fixed_cluster_layers:
                    continue
                fixed_cluster_layers.add(key)
                gn_main = cluster_layer_node[key]
                gn_rot = cluster_layer_rot_node[key]
                for dof_type in (_W, _BETA_S, _BETA_X):
                    fixed.add(_dof(gn_main, dof_type))
                fixed.add(_dof(gn_rot, _BETA_S))
    else:
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            n = len(pm.s_nodes)
            if bc_mode == "full_clamp":
                support_nodes = (0, n - 1, n, 2 * n - 1)
            else:
                support_nodes = (0, n - 1)
            for ln in support_nodes:
                gn = pm.global_nodes[ln]
                for dof_type in (_W, _BETA_S, _BETA_X):
                    fixed.add(_dof(gn, dof_type))

    # Apply panel-level static consistency correction to match integrated targets.
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p = panels[pi]
        s_nodes = pm.s_nodes
        Nx_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nx_panels[pi], dtype=float))
        Nxy_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nxy_panels[pi], dtype=float))
        target_fx = float(np.trapezoid(Nx_nodes, s_nodes))
        target_fs = float(np.trapezoid(Nxy_nodes, s_nodes))
        panel_load_totals[pi]["Fx_target"] = target_fx
        panel_load_totals[pi]["Fs_target"] = target_fs
        d_fx = target_fx - float(panel_load_totals[pi].get("Fx_total", 0.0))
        d_fs = target_fs - float(panel_load_totals[pi].get("Fs_total", 0.0))
        if abs(d_fx) > 0.0 or abs(d_fs) > 0.0:
            n = len(s_nodes)
            w = np.zeros(n)
            for i in range(n):
                if i == 0:
                    w[i] = 0.5 * abs(s_nodes[1] - s_nodes[0]) if n > 1 else 1.0
                elif i == n - 1:
                    w[i] = 0.5 * abs(s_nodes[-1] - s_nodes[-2])
                else:
                    w[i] = 0.5 * abs(s_nodes[i] - s_nodes[i - 1]) + 0.5 * abs(s_nodes[i + 1] - s_nodes[i])
            w_sum = float(np.sum(w))
            w = np.ones(n) if w_sum < 1e-30 else w
            w_sum = float(np.sum(w))
            for i in range(n):
                frac = float(w[i] / w_sum)
                gn = pm.global_nodes[i + n]
                f_global[_dof(gn, _U_X)] += d_fx * frac
                f_global[_dof(gn, _U_S)] += d_fs * frac
            panel_load_totals[pi]["Fx_total"] = target_fx
            panel_load_totals[pi]["Fs_total"] = target_fs
            panel_load_totals[pi]["Fx_correction"] = d_fx
            panel_load_totals[pi]["Fs_correction"] = d_fs

    # Propagate fixity through MPCs: if all masters of a slave are fixed, the slave
    # is also effectively fixed to zero.  Without propagation, such slaves appear as
    # unconstrained free DOFs whose residuals are never attributed to any support
    # reaction, breaking the global force-balance audit.
    _propagated = True
    while _propagated:
        _propagated = False
        for _slave, _terms in constraints.items():
            if _slave in fixed:
                continue
            if all(_master in fixed for _master, _ in _terms):
                fixed.add(_slave)
                _propagated = True

    fixed_sorted = sorted(fixed)
    constrained_slaves = set(constraints.keys())
    # Each constraint value is a list of (master_dof, coeff) pairs.
    constrained_masters = {m for terms in constraints.values() for m, _ in terms}
    all_idx = np.arange(n_gdof)
    independent = np.array([i for i in all_idx if i not in fixed and i not in constrained_slaves], dtype=int)
    red_pos = {dof: i for i, dof in enumerate(independent.tolist())}
    t_rows: list[int] = []
    t_cols: list[int] = []
    t_vals: list[float] = []
    for dof in independent.tolist():
        t_rows.append(dof)
        t_cols.append(red_pos[dof])
        t_vals.append(1.0)
    for slave, terms in constraints.items():
        if slave in fixed:
            continue
        for master, coeff in terms:
            if master in red_pos:
                t_rows.append(slave)
                t_cols.append(red_pos[master])
                t_vals.append(float(coeff))
    T = sp.coo_matrix((t_vals, (t_rows, t_cols)), shape=(n_gdof, len(independent))).tocsr()
    d_global = np.zeros(n_gdof)
    if len(independent) > 0:
        K_rr = (T.T @ K_global @ T).tocsc()
        f_r = np.asarray(T.T @ f_global, dtype=float)

        # Optional soft traction-balance penalty for 2-way non-collinear clusters.
        # Adds (k/2)*(Tx_i + Tx_j)^2 to the energy functional, enforcing Newton III
        # at sharp cusp junctions (e.g., LE) where the MPC alone cannot guarantee it.
        if enforce_traction_balance_at_cusp and use_cluster_basis:
            from examples.section_stress_model.lib.laminate_clpt import (  # type: ignore[import-untyped]
                abd_stack,
            )
            k_diag_max = float(K_rr.diagonal().max()) if K_rr.shape[0] > 0 else 1.0
            k_pen = traction_penalty_alpha * k_diag_max
            for cid, clu in enumerate(clusters):
                if len(clu) != 2 or cluster_is_collinear_flags.get(cid, True):
                    continue
                c_vecs: list[np.ndarray] = []
                for pi_c, which_c, _ in clu:
                    pm_c = panel_maps[pi_c]
                    if not pm_c.elements:
                        break
                    elem_local = pm_c.elements[0] if which_c == "start" else pm_c.elements[-1]
                    edge_side = which_c
                    i0_e, i1_e, i2_e, i3_e = elem_local
                    L_s_e = float(abs(pm_c.s_nodes[i1_e] - pm_c.s_nodes[i0_e]))
                    p_c = panels[pi_c]
                    A_c, B_c, D_c = abd_stack(p_c.lam.build_plies())
                    ABD_c = np.block([[A_c, B_c], [B_c, D_c]])
                    # c_elem[k] = ∂Tx_int/∂d_elem[k]: linear map from element DOFs
                    # to boundary traction (evaluated analytically via unit vectors).
                    c_elem = np.zeros(20)
                    for k_dof in range(20):
                        e_k = np.zeros(20)
                        e_k[k_dof] = 1.0
                        c_elem[k_dof] = float(
                            mitc4_edge_shear_traction_integrated(
                                e_k, L_s_e, 1.0, ABD_c, edge=edge_side, gauss_n=4
                            )["Tx_edge_int"]
                        )
                    c_global = np.zeros(n_gdof)
                    for li_idx, ln in enumerate([i0_e, i1_e, i2_e, i3_e]):
                        gn_e = pm_c.global_nodes[ln]
                        for d_idx in range(_NDOF_NODE):
                            c_global[_dof(gn_e, d_idx)] += c_elem[li_idx * _NDOF_NODE + d_idx]
                    c_vecs.append(c_global)
                if len(c_vecs) == 2:
                    c_sum_r = np.asarray(T.T @ (c_vecs[0] + c_vecs[1]), dtype=float)
                    K_rr = (K_rr + sp.csr_matrix(k_pen * np.outer(c_sum_r, c_sum_r))).tocsc()

        q = spla.spsolve(K_rr, f_r)
        d_global = np.asarray(T @ q, dtype=float)
    r_full = np.asarray(K_global @ d_global - f_global, dtype=float)

    # Global force balance at fixed DOFs: authoritative equilibrium metric.
    # At free DOFs, r_full ≈ 0 (solved). At fixed DOFs, r_full = support reaction.
    # sum_rx_fixed_UX + sum(f_global[UX_dofs]) ≈ 0 for a correct solve.
    sum_rx_fixed_UX = float(sum(r_full[dof] for dof in fixed_sorted if dof % _NDOF_NODE == _U_X))

    # Recover per-panel results + diagnostics.
    all_panel_results: list[list[ShellPanelResultants]] = []
    all_panel_diag: list[dict] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            all_panel_results.append([])
            all_panel_diag.append({})
            continue
        p = panels[pi]
        from examples.section_stress_model.lib.laminate_clpt import (  # type: ignore[import-untyped]
            abd_stack,
        )
        A_mat, B_mat, D_mat = abd_stack(p.lam.build_plies())
        ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])
        panel_label = pm.panel_label
        results: list[ShellPanelResultants] = []
        edge_start_acc: list[float] = []
        edge_end_acc: list[float] = []
        edge_start_s_acc: list[float] = []
        edge_end_s_acc: list[float] = []
        edge_start_tx_acc: list[float] = []
        edge_end_tx_acc: list[float] = []
        edge_start_ts_acc: list[float] = []
        edge_end_ts_acc: list[float] = []
        prov = {
            "Nx": FieldProvenance(ProvenanceKind.MITC4, "MITC4 membrane x-resultant"),
            "Ny": FieldProvenance(ProvenanceKind.MITC4, "MITC4 membrane s-resultant"),
            "Nxy": FieldProvenance(ProvenanceKind.MITC4, "MITC4 in-plane shear resultant"),
            "Mx": FieldProvenance(ProvenanceKind.MITC4, "MITC4 bending moment (x-axis)"),
            "My": FieldProvenance(ProvenanceKind.MITC4, "MITC4 bending moment (s-axis)"),
            "Mxy": FieldProvenance(ProvenanceKind.MITC4, "MITC4 twisting moment"),
            "Qx": FieldProvenance(ProvenanceKind.RESERVED, "FSDT / future"),
            "Qy": FieldProvenance(ProvenanceKind.RESERVED, "FSDT / future"),
        }
        for station_idx, e in enumerate(pm.elements):
            i0, i1, i2, i3 = e
            L_s = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
            d_elem = np.zeros(20)
            for local_i, ln in enumerate([i0, i1, i2, i3]):
                gn = pm.global_nodes[ln]
                for dd in range(_NDOF_NODE):
                    d_elem[local_i * _NDOF_NODE + dd] = d_global[_dof(gn, dd)]
            res = mitc4_resultants(d_elem, L_s, 1.0, ABD)
            edge = mitc4_edge_resultants(d_elem, L_s, 1.0, ABD)
            edge_start_acc.append(float(edge["start"]["Nx"]))
            edge_end_acc.append(float(edge["end"]["Nx"]))
            edge_start_s_acc.append(float(edge["start"]["Nxy"]))
            edge_end_s_acc.append(float(edge["end"]["Nxy"]))
            edge_int_start = mitc4_edge_shear_traction_integrated(
                d_elem, L_s, 1.0, ABD, edge="start", gauss_n=4
            )
            edge_int_end = mitc4_edge_shear_traction_integrated(
                d_elem, L_s, 1.0, ABD, edge="end", gauss_n=4
            )
            edge_start_tx_acc.append(float(edge_int_start["Tx_edge_int"]))
            edge_start_ts_acc.append(float(edge_int_start["Ts_edge_int"]))
            edge_end_tx_acc.append(float(edge_int_end["Tx_edge_int"]))
            edge_end_ts_acc.append(float(edge_int_end["Ts_edge_int"]))
            results.append(
                ShellPanelResultants(
                    Nx=res["Nx"],
                    Ny=res["Ny"],
                    Nxy=res["Nxy"],
                    Mx=res["Mx"],
                    My=res["My"],
                    Mxy=res["Mxy"],
                    Qx=None,
                    Qy=None,
                    provenance=dict(prov),
                    sigma_xx_pa=res["Nx"] / max(ABD[0, 0], 1e-30),
                    tau_xy_pa=res["Nxy"],
                    q_n_per_m=res["Nxy"],
                    thickness_m=0.0,
                    panel_label=panel_label,
                    panel_index=pi,
                    station_index=station_idx,
                )
            )
        all_panel_results.append(results)

        n = len(pm.s_nodes)
        if use_cluster_basis:
            # In cluster-basis modes (transformed_basis, shared_rotated) panel endpoint
            # nodes are unique, so r_full at the cluster master node doesn't represent
            # the per-panel contribution. Use outward-normal-signed edge tractions:
            #   Fx = Tx_int = Nxy * normal_sign  (spanwise, X-direction)
            #   Fs = Tx_int = Nxy * normal_sign  (shear-flow continuity; same data,
            #        checked with orientation mapping in check_panel_equilibrium)
            # This is the same quantity used by the cluster-sum check so that Newton III
            # reads Fx_i + Fx_j = 0 at a 2-way junction.
            start_fx = float(edge_start_tx_acc[0]) if edge_start_tx_acc else 0.0
            start_fs = float(edge_start_tx_acc[0]) if edge_start_tx_acc else 0.0
            end_fx = float(edge_end_tx_acc[-1]) if edge_end_tx_acc else 0.0
            end_fs = float(edge_end_tx_acc[-1]) if edge_end_tx_acc else 0.0
        else:
            start_nodes = [pm.global_nodes[0], pm.global_nodes[n]]
            end_nodes = [pm.global_nodes[n - 1], pm.global_nodes[2 * n - 1]]
            start_fx = float(sum(r_full[_dof(gn, _U_X)] for gn in start_nodes))
            start_fs = float(sum(r_full[_dof(gn, _U_S)] for gn in start_nodes))
            end_fx = float(sum(r_full[_dof(gn, _U_X)] for gn in end_nodes))
            end_fs = float(sum(r_full[_dof(gn, _U_S)] for gn in end_nodes))
        edge_start_nx = float(edge_start_acc[0]) if edge_start_acc else 0.0
        edge_end_nx = float(edge_end_acc[-1]) if edge_end_acc else 0.0
        edge_start_nxy = float(edge_start_s_acc[0]) if edge_start_s_acc else 0.0
        edge_end_nxy = float(edge_end_s_acc[-1]) if edge_end_s_acc else 0.0
        edge_start_nxy_int = float(edge_start_tx_acc[0]) if edge_start_tx_acc else 0.0
        edge_end_nxy_int = float(edge_end_tx_acc[-1]) if edge_end_tx_acc else 0.0
        edge_start_ts_int = float(edge_start_ts_acc[0]) if edge_start_ts_acc else 0.0
        edge_end_ts_int = float(edge_end_ts_acc[-1]) if edge_end_ts_acc else 0.0
        # Near-boundary traction samples for R4 shear-gradient analysis.
        _k = min(3, len(edge_start_tx_acc))
        edge_start_tx_near = edge_start_tx_acc[:_k]
        edge_end_tx_near = edge_end_tx_acc[-_k:] if _k > 0 else []
        # Blend edge-resultant recovery with nodal interface-force recovery to
        # stabilize secondary continuity diagnostics at multi-way junctions.
        # In cluster-basis modes start_fx/end_fx carry Tx_int (Nxy-derived), not Nx,
        # so the Nx field diagnostic uses the pure element-edge Nx directly.
        alpha_force = 0.75
        if use_cluster_basis:
            field_start_nx = float(edge_start_nx)
            field_end_nx = float(edge_end_nx)
        else:
            field_start_nx = float(alpha_force * start_fx + (1.0 - alpha_force) * edge_start_nx)
            field_end_nx = float(alpha_force * end_fx + (1.0 - alpha_force) * edge_end_nx)
        # Nxy secondary field is now taken from edge line-integration directly.
        field_start_nxy = edge_start_nxy_int
        field_end_nxy = edge_end_nxy_int
        _cid_start = endpoint_cluster_index.get((pi, "start"), -1)
        _cid_end = endpoint_cluster_index.get((pi, "end"), -1)
        all_panel_diag.append(
            {
                "endpoint_cluster_index": {
                    "start": int(_cid_start),
                    "end": int(_cid_end),
                },
                "endpoint_cluster_collinear": {
                    "start": bool(cluster_is_collinear_flags.get(_cid_start, True)),
                    "end": bool(cluster_is_collinear_flags.get(_cid_end, True)),
                },
                "panel_end_tangent": {
                    "start": _panel_end_tangent(nodes_yz, "start").tolist(),
                    "end": _panel_end_tangent(nodes_yz, "end").tolist(),
                },
                "boundary_reaction_set": {
                    "start": {"Fx": start_fx, "Fs": start_fs},
                    "end": {"Fx": end_fx, "Fs": end_fs},
                },
                "interface_field_set": {
                    "start": {"Nx": field_start_nx, "Nxy": field_start_nxy},
                    "end": {"Nx": field_end_nx, "Nxy": field_end_nxy},
                },
                "interface_edge_set": {
                    "start": {
                        "Nx": edge_start_nx,
                        "Nxy": edge_start_nxy,
                        "Nxy_int": edge_start_nxy_int,
                        "Tx_int": edge_start_nxy_int,
                        "Ts_int": edge_start_ts_int,
                        "Tx_near": edge_start_tx_near,
                    },
                    "end": {
                        "Nx": edge_end_nx,
                        "Nxy": edge_end_nxy,
                        "Nxy_int": edge_end_nxy_int,
                        "Tx_int": edge_end_nxy_int,
                        "Ts_int": edge_end_ts_int,
                        "Tx_near": edge_end_tx_near,
                    },
                },
                "load_totals": panel_load_totals[pi],
                "residual": {
                    "free_res_rel": float(np.linalg.norm(r_full[independent]) / max(np.linalg.norm(f_global[independent]), 1e-12)),
                    "global_force_mismatch_rel": float(np.linalg.norm(r_full) / max(np.linalg.norm(f_global), 1e-12)),
                },
                "constraint_stats": {
                    "n_reduced_dofs": int(len(independent)),
                    "n_slave_dofs": int(len(constrained_slaves)),
                    "n_master_dofs": int(len(constrained_masters)),
                    "n_fixed_dofs": int(len(fixed_sorted)),
                },
                "endpoint_cluster_rot_node": (
                    {
                        "start": {
                            "bottom": int(endpoint_cluster_rot_node[(pi, "start", "bottom")]),
                            "top": int(endpoint_cluster_rot_node[(pi, "start", "top")]),
                        },
                        "end": {
                            "bottom": int(endpoint_cluster_rot_node[(pi, "end", "bottom")]),
                            "top": int(endpoint_cluster_rot_node[(pi, "end", "top")]),
                        },
                    }
                    if use_cluster_basis
                    else {}
                ),
                "interface_constraint_mode": str(effective_mode),
                # Authoritative global UX force balance at fixed (RBM-support) DOFs.
                # Should equal -sum(Fx_target) for a correct solve; see build_load_reaction_audit.
                "global_reaction_at_fixed_UX": float(sum_rx_fixed_UX),
                "solver": "global_coupled",
            }
        )
    if return_assembly_data:
        node_meta = _collect_global_node_meta(panel_maps, panels, next_global_node)
        return all_panel_results, all_panel_diag, K_global, T, node_meta
    return all_panel_results, all_panel_diag
