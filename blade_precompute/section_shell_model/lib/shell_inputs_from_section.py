"""
shell_inputs_from_section
=========================
Adapter that converts a :class:`MultiCellSection` from ``section_geometry``
into a :class:`ShellMeshInputs` payload consumed by the shell-stage
topology generator.

**Pipeline boundary**

This module is the handoff point between geometry and meshing.  Once
:func:`build_shell_mesh_inputs` returns, no further SDF evaluation
(``section[label](x, y)`` / component ``__call__``) should occur on the
underlying section — all geometry is materialised as B-frame polylines inside
:class:`SubcomponentMidline` records (which are
:class:`~blade_precompute.contract.shell_midline_strip.ShellMidlineStrip`
instances).

**Naming note:** "caching" a midline here means *materialising the mid-surface
locus as a polyline* for the mesh handoff.  This is distinct from any SDF
*evaluation cache* held internally by the ``section_geometry`` subcomponent
objects.

**Architecture**

Every SDF subcomponent built by ``section_geometry`` (``OuterSkin``,
``ShearWeb``, ``SparCap`` / ``ContinuousSparCap``) exposes a
``midline_polyline()`` method that returns its midline in the **chord frame**.
:func:`build_shell_midline_strips` (in ``section_geometry.interface``) iterates
those, rotates each polyline into the **B-frame**, and returns them as
:class:`~blade_precompute.contract.shell_midline_strip.ShellMidlineStrip`
records.  This adapter wraps that call and appends LE/TE context to produce
:class:`ShellMeshInputs`.

After skin split, **web** endpoints snap to the skin polyline at their
projected arc-lengths; **spar cap** endpoints then prefer the **nearest**
snapped **shear-web foot** (within a fixed tolerance) so junction clustering
reuses web–skin nodes. Cap **interiors** may then be replaced by intersections
of **inward** rays from master skin vertices with each cap midline (see
:func:`_resample_cap_interiors_from_skin_rays`); cap endpoints are not moved by
that step. Finally, :func:`_insert_cap_web_interior_junctions` adds vertices at
strict interior cap–web crossings and near web feet (within a proximity
tolerance) so caps carry mesh nodes at skin–web T-junctions along the strip.

**Taxonomy (structural family Y):** **Y = C** (discrete caps per web) yields
cap–web interior junctions at every web by construction. **Y = B** (single cap
band) may have no strict interior cap–web crossing at low web counts, but at
larger **X** (more webs) the band can cross interior webs, so the same insertion
logic applies when geometry demands it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.contract.shell_midline_strip import ShellMidlineStrip
from blade_precompute.section_geometry.interface.shell_midline_export import (
    build_shell_midline_strips,
    rotate_chord_to_blade,
)

# Near-zero segment-length threshold (metres). Segments shorter than this are
# considered degenerate and will raise ValueError during strip validation.
_MIN_SEGMENT_LENGTH_M: float = 1e-12

# Merge tolerance for split arc-length parameters on the skin midline (metres).
_MERGE_S_TOL_M: float = 1e-4

# Merge cap interior junction hits that fall within this arc-length (metres) on the cap.
_CAP_WEB_JUNCTION_MERGE_ARC_M: float = 1e-6

# Endpoint-on-web tolerance (metres): if a cap endpoint is within this distance
# of any web segment, treat it as a web-coupled endpoint (Class B).
_CAP_ENDPOINT_ON_WEB_TOL_M: float = 5e-4

# Bbox inflation when selecting skin vertices for cap interior ray casting (metres).
_SKIN_RAY_CAP_BBOX_MARGIN_M: float = 5e-3

# Minimum ray parameter (metres) to ignore grazing hits at the ray origin.
_SKIN_RAY_T_MIN_M: float = 1e-9

# Backward-compatible alias: existing imports of SubcomponentMidline continue to work.
SubcomponentMidline = ShellMidlineStrip

_DEBUG_LOG_PATH = Path(__file__).resolve().parents[3] / "debug-caf7ba.log"


def _agent_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    payload = {
        "sessionId": "caf7ba",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass
    # endregion agent log


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellMeshInputs:
    """Per-subcomponent midline payload for the shell-stage topology generator.

    All polylines are expressed in the rotated **B-frame** so the downstream
    mesh can be plotted directly on top of ``section_geometry`` outputs
    without further rotation.

    ``leading_edge_b`` / ``trailing_edge_b`` are the airfoil LE/TE points
    rotated into the B-frame and provided as scalar context for diagnostics
    or any consumer that needs to orient the section without recomputing
    the airfoil contour.
    """

    chord_m: float
    twist_rad: float
    layout_key: str | None
    midlines: list[SubcomponentMidline] = field(default_factory=list)
    leading_edge_b: tuple[float, float] = (0.0, 0.0)
    trailing_edge_b: tuple[float, float] = (1.0, 0.0)


# ---------------------------------------------------------------------------
# Strip validation (private)
# ---------------------------------------------------------------------------


def _validate_strips(strips: tuple[ShellMidlineStrip, ...]) -> None:
    """Raise ``ValueError`` if any strip fails the geometric sanity checks.

    B1: midline_b must have shape ``(N, 2)`` with ``N >= 2``.
    B2: No near-zero consecutive segment lengths (< ``_MIN_SEGMENT_LENGTH_M``).
    """
    for s in strips:
        pts = np.asarray(s.midline_b)

        # B1 — shape and minimum point count.
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"Strip {s.label!r} ({s.kind}): midline_b must have shape (N, 2), "
                f"got {pts.shape}."
            )
        if pts.shape[0] < 2:
            raise ValueError(
                f"Strip {s.label!r} ({s.kind}): midline_b must have at least 2 "
                f"vertices, got {pts.shape[0]}."
            )

        # B2 — near-zero segment lengths.
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        bad = np.where(seg_lengths < _MIN_SEGMENT_LENGTH_M)[0]
        if bad.size:
            raise ValueError(
                f"Strip {s.label!r} ({s.kind}): {bad.size} degenerate segment(s) "
                f"with length < {_MIN_SEGMENT_LENGTH_M:.2e} m at vertex indices "
                f"{bad.tolist()}. Check midline_polyline() implementation for "
                f"this subcomponent."
            )


# ---------------------------------------------------------------------------
# Skin split at web/cap junctions (shared nodes on skin midline)
# ---------------------------------------------------------------------------


def _closed_ring_edge_data(
    verts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Per-edge length, arc offset at each vertex start, and perimeter ``L``."""
    n = int(verts.shape[0])
    edge_len = np.zeros(n, dtype=float)
    for i in range(n):
        j = (i + 1) % n
        edge_len[i] = float(np.linalg.norm(verts[j] - verts[i]))
    L = float(np.sum(edge_len))
    edge_start = np.zeros(n, dtype=float)
    for i in range(1, n):
        edge_start[i] = edge_start[i - 1] + edge_len[i - 1]
    return edge_len, edge_start, L


def _merge_sorted_s_values(values: list[float], tol_m: float) -> list[float]:
    if not values:
        return []
    vals = sorted(float(v) for v in values)
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > tol_m:
            out.append(v)
    return out


def _nearest_on_closed_polyline(
    pt: NDArray[np.float64],
    verts: NDArray[np.float64],
    edge_len: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    L: float,
) -> tuple[float, NDArray[np.float64]]:
    """Closest point on a closed polygonal ring; returns ``(s, yz)`` with ``s`` in ``[0, L)``."""
    n = int(verts.shape[0])
    p = np.asarray(pt, dtype=float).reshape(2)
    best_d2 = float("inf")
    best_s = 0.0
    best_yz = verts[0].astype(float).copy()
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        e = b - a
        elen2 = float(np.dot(e, e))
        if elen2 < 1e-30:
            t = 0.0
        else:
            t = float(np.clip(np.dot(p - a, e) / elen2, 0.0, 1.0))
        q = a + t * e
        d2 = float(np.sum((p - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_yz = q
            best_s = float(edge_start[i] + t * edge_len[i])
    if L > 1e-15:
        best_s = float(best_s % L)
    return best_s, best_yz


def _yz_at_s_on_closed_ring(
    s: float,
    verts: NDArray[np.float64],
    edge_len: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    L: float,
) -> NDArray[np.float64]:
    if L <= 1e-15:
        return verts[0].astype(float).copy()
    s_mod = float(s % L)
    n = int(verts.shape[0])
    for i in range(n):
        lo = float(edge_start[i])
        hi = lo + float(edge_len[i])
        if lo <= s_mod <= hi + 1e-12 or (i == n - 1 and s_mod >= lo - 1e-12):
            el = float(edge_len[i])
            t = 0.0 if el < 1e-30 else float(np.clip((s_mod - lo) / el, 0.0, 1.0))
            j = (i + 1) % n
            return verts[i] + t * (verts[j] - verts[i])
    return verts[0].astype(float).copy()


def _arc_polyline_no_wrap(
    s_lo: float,
    s_hi: float,
    verts: NDArray[np.float64],
    edge_len: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    L: float,
) -> NDArray[np.float64]:
    """Polyline samples from ``s_lo`` to ``s_hi`` along the ring (``s_lo < s_hi``)."""
    n = int(verts.shape[0])
    out_pts: list[NDArray[np.float64]] = []
    out_pts.append(_yz_at_s_on_closed_ring(s_lo, verts, edge_len, edge_start, L))
    for j in range(1, n):
        sj = float(edge_start[j])
        if s_lo + 1e-9 < sj < s_hi - 1e-9:
            out_pts.append(np.asarray(verts[j], dtype=float).copy())
    out_pts.append(_yz_at_s_on_closed_ring(s_hi, verts, edge_len, edge_start, L))
    return np.stack(out_pts, axis=0)


def _arc_polyline_wrap(
    s_lo: float,
    s_hi: float,
    verts: NDArray[np.float64],
    edge_len: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    L: float,
) -> NDArray[np.float64]:
    """Arc from ``s_lo`` forward through ``L`` and ``0`` to ``s_hi`` (``s_lo > s_hi`` on the circle)."""
    a = _arc_polyline_no_wrap(s_lo, L, verts, edge_len, edge_start, L)
    b = _arc_polyline_no_wrap(0.0, s_hi, verts, edge_len, edge_start, L)
    if len(a) and len(b) and np.allclose(a[-1], b[0], atol=1e-9, rtol=0.0):
        return np.vstack([a[:-1], b])
    return np.vstack([a, b])


def _split_skin_at_junctions(
    strips: tuple[ShellMidlineStrip, ...],
    *,
    merge_s_tol_m: float = _MERGE_S_TOL_M,
) -> tuple[ShellMidlineStrip, ...]:
    """Split the outer skin at web/cap projections and snap strip endpoints for mesh junctions.

    The skin ring is split using arc-length parameters from **both** web and
    cap endpoint projections onto the **original** closed skin midline
    (unchanged ordering). **Web** endpoints are then moved to those skin
    projections. **Cap** strips are left unchanged here; cap/web shared nodes
    are handled downstream by :func:`_insert_cap_web_interior_junctions`.

    Ensures web strip endpoints lie on skin polyline vertices so downstream
    endpoint clustering joins web–skin strips at one junction without duplicate
    colocated vertices.
    """
    skin_indices = [i for i, s in enumerate(strips) if s.kind == "skin"]
    if not skin_indices:
        return strips

    si = skin_indices[0]
    skin = strips[si]
    if not skin.closed:
        return strips

    verts = np.asarray(skin.midline_b, dtype=float)
    if verts.shape[0] < 3:
        return strips

    edge_len, edge_start, L = _closed_ring_edge_data(verts)
    if L < 1e-9:
        return strips

    projections: list[tuple[int, bool, float, NDArray[np.float64]]] = []
    split_s: list[float] = []

    for j, st in enumerate(strips):
        if st.kind not in ("web", "cap"):
            continue
        pts = np.asarray(st.midline_b, dtype=float)
        if pts.shape[0] < 2:
            continue
        for is_first in (True, False):
            row = 0 if is_first else -1
            pt = np.asarray(pts[row], dtype=float)
            s_val, yz = _nearest_on_closed_polyline(pt, verts, edge_len, edge_start, L)
            split_s.append(s_val)
            projections.append((j, is_first, s_val, yz))

    split_merged = _merge_sorted_s_values(split_s, merge_s_tol_m)
    k = len(split_merged)

    out = list(strips)

    # Phase web: snap web strip ends to skin projections (same geometry as the
    # first-pass skin split parameters).
    for j, st in enumerate(strips):
        if st.kind != "web":
            continue
        arr = np.asarray(st.midline_b, dtype=float).copy()
        for jj, is_first, _s, yz in projections:
            if jj != j:
                continue
            if is_first:
                arr[0] = yz
            else:
                arr[-1] = yz
        out[j] = replace(st, midline_b=arr)

    # Phase cap Class-A: for cap endpoints *away from webs*, snap to the
    # projected skin node at cap-end position. If the endpoint is on/near a web
    # (Class-B), leave unchanged here and let cap-web junction insertion own it.
    web_polys: list[NDArray[np.float64]] = [
        np.asarray(st.midline_b, dtype=float)
        for st in out
        if st.kind == "web" and np.asarray(st.midline_b, dtype=float).shape[0] >= 2
    ]
    for j, st in enumerate(strips):
        if st.kind != "cap":
            continue
        orig = np.asarray(st.midline_b, dtype=float).copy()
        arr = orig.copy()
        for jj, is_first, _s, yz_skin in projections:
            if jj != j:
                continue
            row = 0 if is_first else -1
            p_end = np.asarray(orig[row], dtype=float).reshape(2)
            on_web = False
            for w in web_polys:
                d_web, _q_web = _point_to_open_polyline_nearest(p_end, w)
                if d_web <= _CAP_ENDPOINT_ON_WEB_TOL_M:
                    on_web = True
                    break
            if not on_web:
                arr[row] = np.asarray(yz_skin, dtype=float).reshape(2)
        out[j] = replace(st, midline_b=arr)

    if k < 2:
        return tuple(out)

    new_skin_strips: list[ShellMidlineStrip] = []
    for seg_i in range(k):
        s_lo = float(split_merged[seg_i])
        s_hi = float(split_merged[(seg_i + 1) % k])
        if seg_i < k - 1:
            if s_hi <= s_lo + 1e-12:
                continue
            seg_pts = _arc_polyline_no_wrap(s_lo, s_hi, verts, edge_len, edge_start, L)
        else:
            # Close the ring: last split → perimeter end → vertex 0 → first split.
            seg_pts = _arc_polyline_wrap(s_lo, s_hi, verts, edge_len, edge_start, L)
        if seg_pts.shape[0] < 2:
            continue
        new_skin_strips.append(
            ShellMidlineStrip(
                label=f"{skin.label}:seg{seg_i}",
                kind="skin",
                midline_b=seg_pts,
                thickness_m=float(skin.thickness_m),
                closed=False,
                surface=skin.surface,
                alignment=skin.alignment,
            )
        )

    if len(new_skin_strips) < 2:
        return tuple(out)

    return tuple(out[:si] + new_skin_strips + out[si + 1 :])


# ---------------------------------------------------------------------------
# Cap interior: resample via inward rays from skin master (B-frame)
# ---------------------------------------------------------------------------


def _normal_2d_ccw(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit normal: 90° CCW rotation of a 2-D tangent (same convention as MITC4 assembly)."""
    tx, ty = float(t[0]), float(t[1])
    n = np.array([-ty, tx], dtype=float)
    ln = float(np.linalg.norm(n))
    if ln < 1e-30:
        return np.array([0.0, 1.0], dtype=float)
    return n / ln


def _concat_skin_ring_vertices(strips: tuple[ShellMidlineStrip, ...]) -> NDArray[np.float64]:
    """Concatenate all skin ``midline_b`` strips in list order; dedupe shared joints."""
    chunks: list[NDArray[np.float64]] = []
    for st in strips:
        if st.kind != "skin":
            continue
        arr = np.asarray(st.midline_b, dtype=float)
        if arr.shape[0] < 2:
            continue
        if chunks and np.allclose(chunks[-1][-1], arr[0], atol=1e-9, rtol=0.0):
            arr = arr[1:]
        if arr.shape[0] == 0:
            continue
        chunks.append(arr)
    if not chunks:
        return np.empty((0, 2), dtype=float)
    return np.vstack(chunks)


def _skin_ring_vertex_tangents(ring: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit tangent at each ring vertex; ring treated closed (last → first edge)."""
    m = int(ring.shape[0])
    if m < 2:
        return np.zeros((0, 2), dtype=float)
    out = np.zeros((m, 2), dtype=float)
    for i in range(m):
        prev_v = ring[i - 1]
        next_v = ring[(i + 1) % m]
        d = next_v - prev_v
        ln = float(np.linalg.norm(d))
        if ln < 1e-30:
            out[i] = np.array([1.0, 0.0], dtype=float)
        else:
            out[i] = d / ln
    return out


def _open_polyline_prefix_lengths(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative distance along open polyline ``pts``; ``prefix[i]`` = arc to vertex ``i``."""
    n = int(pts.shape[0])
    if n < 2:
        return np.zeros(max(n, 0), dtype=float)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    prefix = np.zeros(n, dtype=float)
    prefix[1:] = np.cumsum(ds)
    return prefix


def _intersect_open_segments_2d(
    a0: NDArray[np.float64],
    a1: NDArray[np.float64],
    b0: NDArray[np.float64],
    b1: NDArray[np.float64],
    *,
    eps: float = 1e-9,
) -> tuple[float, float, NDArray[np.float64]] | None:
    """Strict-interior intersection of segment ``a0a1`` with ``b0b1``; ``ta,tb`` in ``(0,1)``."""
    r = (a1.astype(float) - a0).reshape(2)
    svec = (b1.astype(float) - b0).reshape(2)
    rxs = float(r[0] * svec[1] - r[1] * svec[0])
    denom_tol = 1e-15 * max(
        1.0,
        float(np.linalg.norm(r)) * float(np.linalg.norm(svec)),
    )
    if abs(rxs) < denom_tol:
        return None
    qp = (b0.astype(float) - a0.astype(float)).reshape(2)
    ta = float((qp[0] * svec[1] - qp[1] * svec[0]) / rxs)
    tb = float((qp[0] * r[1] - qp[1] * r[0]) / rxs)
    if ta <= eps or ta >= 1.0 - eps or tb <= eps or tb >= 1.0 - eps:
        return None
    pt = (a0.astype(float) + ta * r).astype(float)
    return ta, tb, pt


def _open_polyline_min_segment_length(pts: NDArray[np.float64]) -> float:
    if pts.shape[0] < 2:
        return float("inf")
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).min())


def _point_to_open_polyline_nearest(
    p: NDArray[np.float64], w: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Nearest distance and point from 2-D point ``p`` to open polyline ``w``."""
    q = np.asarray(p, dtype=float).reshape(2)
    pts = np.asarray(w, dtype=float)
    if pts.shape[0] < 2:
        return float("inf"), q.copy()
    best_d = float("inf")
    best_pt = q.copy()
    for j in range(pts.shape[0] - 1):
        a = pts[j]
        b = pts[j + 1]
        ab = b - a
        l2 = float(np.dot(ab, ab)) + 1e-30
        t = float(np.dot(q - a, ab) / l2)
        t = float(np.clip(t, 0.0, 1.0))
        c = a + t * ab
        d = float(np.linalg.norm(q - c))
        if d < best_d:
            best_d = d
            best_pt = np.asarray(c, dtype=float).reshape(2)
    return best_d, best_pt


def _insert_cap_web_interior_junctions(
    strips: tuple[ShellMidlineStrip, ...],
    *,
    vertex_coincide_tol_m: float = 1e-6,
    merge_arc_m: float | None = None,
    min_seg_m: float | None = None,
    foot_proximity_tol_m: float = 0.0,
) -> tuple[ShellMidlineStrip, ...]:
    """Insert cap midline vertices for web-related T-junctions.

    (1) Strict interior **segment intersections** of the cap with each web polyline.
    (1b) Cap endpoint-on-web cases: if a cap endpoint is within
    ``_CAP_ENDPOINT_ON_WEB_TOL_M`` of a web polyline, insert the nearest
    web-midline point so cap/web share one web-centric node.
    (2) Optional **proximity** to each shear-web **foot** (endpoint on skin): when
    enabled (``foot_proximity_tol_m > 0``) and the foot is within tolerance of the
    cap polyline, insert the foot ``yz`` as a new vertex.

    Layout-wise, **Y = C** systems hit every web this way by construction; **Y = B**
    systems often need it only once **X** is large enough that the cap band
    crosses webs between bay ends.

    Intended after :func:`_split_skin_at_junctions` and (when enabled) after
    :func:`_resample_cap_interiors_from_skin_rays`, so cap endpoints are final.
    """
    web_pts: list[NDArray[np.float64]] = []
    web_feet: list[NDArray[np.float64]] = []
    for st in strips:
        if st.kind != "web":
            continue
        w = np.asarray(st.midline_b, dtype=float)
        if w.shape[0] >= 2:
            web_pts.append(w)
            web_feet.append(np.asarray(w[0], dtype=float).reshape(2))
            web_feet.append(np.asarray(w[-1], dtype=float).reshape(2))

    if not web_pts:
        return strips

    min_seg = float(min_seg_m) if min_seg_m is not None else max(_MIN_SEGMENT_LENGTH_M * 100.0, 1e-9)
    foot_tol = float(foot_proximity_tol_m)
    use_foot_proximity = foot_tol > 0.0

    out: list[ShellMidlineStrip] = []
    for st in strips:
        if st.kind != "cap":
            out.append(st)
            continue
        pts = np.asarray(st.midline_b, dtype=float)
        if pts.shape[0] < 2:
            out.append(st)
            continue
        prefix = _open_polyline_prefix_lengths(pts)
        arc_total = float(prefix[-1])
        if arc_total < min_seg * 3.0:
            out.append(st)
            continue
        merge_arc = (
            float(merge_arc_m)
            if merge_arc_m is not None
            else max(1e-7 * arc_total, _CAP_WEB_JUNCTION_MERGE_ARC_M)
        )
        eps_s = max(1e-12 * arc_total, 1e-15)

        hits: list[tuple[float, NDArray[np.float64]]] = []
        n = int(pts.shape[0])
        for i in range(n - 1):
            a0, a1 = pts[i], pts[i + 1]
            seg_len = float(np.linalg.norm(a1 - a0))
            if seg_len < min_seg:
                continue
            ab = (a1 - a0).astype(float)
            l2 = float(np.dot(ab, ab)) + 1e-30

            for w in web_pts:
                nw = int(w.shape[0])
                for j in range(nw - 1):
                    inter = _intersect_open_segments_2d(a0, a1, w[j], w[j + 1])
                    if inter is None:
                        continue
                    ta, _tb, p_hit = inter
                    s_hit = float(prefix[i] + ta * seg_len)
                    if s_hit <= merge_arc or s_hit >= arc_total - merge_arc:
                        continue
                    hits.append((s_hit, np.asarray(p_hit, dtype=float).reshape(2)))

            if use_foot_proximity:
                eps_line = max(1e-12, min(1e-7, 1e-6 * seg_len / max(seg_len, 1e-15)))
                for qf in web_feet:
                    q = np.asarray(qf, dtype=float).reshape(2)
                    t_raw = float(np.dot(q - a0, ab) / l2)
                    if t_raw <= 0.0:
                        da = float(np.linalg.norm(q - a0))
                        if da > foot_tol:
                            continue
                        t_ins = eps_line
                    elif t_raw >= 1.0:
                        db = float(np.linalg.norm(q - a1))
                        if db > foot_tol:
                            continue
                        t_ins = 1.0 - eps_line
                    else:
                        c = a0 + ab * t_raw
                        d = float(np.linalg.norm(q - c))
                        if d > foot_tol:
                            continue
                        t_ins = float(t_raw)
                    s_hit = float(prefix[i] + t_ins * seg_len)
                    if s_hit <= merge_arc or s_hit >= arc_total - merge_arc:
                        continue
                    if float(np.linalg.norm(pts - q, axis=1).min()) <= vertex_coincide_tol_m:
                        continue
                    hits.append((s_hit, q.copy()))

        # Endpoint-on-web handling (Class-B):
        # inject a web-midline point for cap endpoints that sit on/near a web,
        # unless already represented by an existing cap vertex.
        for row in (0, -1):
            p_end = np.asarray(pts[row], dtype=float).reshape(2)
            best_d = float("inf")
            best_q: NDArray[np.float64] | None = None
            best_s_hit: float | None = None
            for w in web_pts:
                d_web, q_web = _point_to_open_polyline_nearest(p_end, w)
                if d_web >= best_d:
                    continue
                # Place slightly interior on first/last segment so insertion
                # can happen inside open-polyline merge logic.
                seg_idx = 0 if row == 0 else n - 2
                seg_idx = int(np.clip(seg_idx, 0, n - 2))
                seg_len = float(np.linalg.norm(pts[seg_idx + 1] - pts[seg_idx]))
                if seg_len < min_seg:
                    continue
                eps_line = max(1e-12, min(1e-7, 1e-6 * seg_len / max(seg_len, 1e-15)))
                t_ins = eps_line if row == 0 else 1.0 - eps_line
                s_hit = float(prefix[seg_idx] + t_ins * seg_len)
                best_d = d_web
                best_q = np.asarray(q_web, dtype=float).reshape(2)
                best_s_hit = s_hit
            if best_q is None or best_s_hit is None:
                continue
            if best_d > _CAP_ENDPOINT_ON_WEB_TOL_M:
                continue
            if best_s_hit <= merge_arc or best_s_hit >= arc_total - merge_arc:
                continue
            if float(np.linalg.norm(pts - best_q, axis=1).min()) <= vertex_coincide_tol_m:
                continue
            hits.append((best_s_hit, best_q.copy()))

        if not hits:
            out.append(st)
            continue

        hits.sort(key=lambda h: h[0])
        merged: list[tuple[float, NDArray[np.float64]]] = []
        for s_h, p_h in hits:
            if merged and abs(s_h - merged[-1][0]) <= merge_arc:
                continue
            if float(np.linalg.norm(pts - p_h, axis=1).min()) <= vertex_coincide_tol_m:
                continue
            merged.append((s_h, p_h))

        if not merged:
            out.append(st)
            continue

        new_rows: list[NDArray[np.float64]] = []
        mi = 0
        for i in range(n - 1):
            new_rows.append(pts[i].copy())
            s_lo = float(prefix[i])
            s_hi = float(prefix[i + 1])
            while mi < len(merged) and merged[mi][0] < s_hi - eps_s:
                s_h, p_h = merged[mi]
                if s_h > s_lo + eps_s and float(np.linalg.norm(p_h - new_rows[-1])) >= min_seg:
                    new_rows.append(p_h.copy())
                mi += 1
        new_rows.append(pts[-1].copy())

        slim: list[NDArray[np.float64]] = [new_rows[0]]
        for r in new_rows[1:]:
            if float(np.linalg.norm(r - slim[-1])) < min_seg:
                slim[-1] = r.copy()
            else:
                slim.append(r.copy())

        new_pts = np.stack(slim, axis=0)
        if new_pts.shape[0] < 2 or _open_polyline_min_segment_length(new_pts) < _MIN_SEGMENT_LENGTH_M:
            out.append(st)
            continue
        out.append(replace(st, midline_b=new_pts))

    return tuple(out)


def _ray_segment_hit_first(
    p: NDArray[np.float64],
    v: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    t_min: float = _SKIN_RAY_T_MIN_M,
) -> tuple[float, NDArray[np.float64]] | None:
    """First hit of ray ``p + t v`` (``t >= t_min``) on segment ``ab``; ``v`` unit."""
    w = b - a
    mat = np.column_stack([v, -w])
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-18:
        return None
    rhs = a - p
    tu = np.linalg.solve(mat, rhs)
    t = float(tu[0])
    u = float(tu[1])
    if t < t_min or u < -1e-10 or u > 1.0 + 1e-10:
        return None
    uu = float(np.clip(u, 0.0, 1.0))
    hit = a + uu * w
    return t, hit


def _ray_hits_open_polyline(
    p: NDArray[np.float64],
    v: NDArray[np.float64],
    cap_pts: NDArray[np.float64],
) -> list[tuple[float, float, NDArray[np.float64]]]:
    """All ray hits on open ``cap_pts``; each entry ``(t_ray, s_cap, hit)`` with smallest ``t`` first."""
    n = int(cap_pts.shape[0])
    hits: list[tuple[float, float, NDArray[np.float64]]] = []
    prefix = _open_polyline_prefix_lengths(cap_pts)
    for j in range(n - 1):
        hs = _ray_segment_hit_first(p, v, cap_pts[j], cap_pts[j + 1])
        if hs is None:
            continue
        t_ray, hit = hs
        u_len = float(np.linalg.norm(cap_pts[j + 1] - cap_pts[j]))
        uu = 0.0 if u_len < 1e-30 else float(np.linalg.norm(hit - cap_pts[j]) / u_len)
        uu = float(np.clip(uu, 0.0, 1.0))
        s_cap = float(prefix[j] + uu * u_len)
        hits.append((t_ray, s_cap, hit.astype(float)))
    hits.sort(key=lambda h: h[0])
    return hits


def _pick_inward_ray_direction(
    p_skin: NDArray[np.float64],
    n_ccw: NDArray[np.float64],
    cap_centroid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Choose ``+/- n_ccw`` so the ray points generally toward the cap interior."""
    n1 = np.asarray(n_ccw, dtype=float)
    n2 = -n1
    if float(np.dot(n1, cap_centroid - p_skin)) >= float(np.dot(n2, cap_centroid - p_skin)):
        return n1
    return n2


def _resample_one_cap_interior_from_skin_rays(
    cap_strip: ShellMidlineStrip,
    skin_ring: NDArray[np.float64],
    skin_tangents: NDArray[np.float64],
    *,
    bbox_margin_m: float = _SKIN_RAY_CAP_BBOX_MARGIN_M,
) -> ShellMidlineStrip | None:
    """Return replaced cap strip, or ``None`` to keep original."""
    pts = np.asarray(cap_strip.midline_b, dtype=float)
    if pts.shape[0] < 3:
        return None

    p0 = pts[0].copy()
    p1 = pts[-1].copy()
    c_mid = np.asarray(np.mean(pts, axis=0), dtype=float).reshape(2)

    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    mrg = float(bbox_margin_m)

    prefix_full = _open_polyline_prefix_lengths(pts)
    arc_total = float(prefix_full[-1])
    if arc_total < 1e-9:
        return None
    s_eps = max(1e-6 * arc_total, 1e-7)
    # Target interior spacing follows the incoming cap polyline resolution so
    # skin-ray hits do not over-cluster and create pathological short segments.
    target_step = arc_total / max(float(pts.shape[0] - 1), 1.0)

    collected: list[tuple[float, NDArray[np.float64]]] = []

    m = int(skin_ring.shape[0])
    for i in range(m):
        q = skin_ring[i]
        if (
            q[0] < xmin - mrg
            or q[0] > xmax + mrg
            or q[1] < ymin - mrg
            or q[1] > ymax + mrg
        ):
            continue
        ti = skin_tangents[i]
        n_ccw = _normal_2d_ccw(ti)
        v = _pick_inward_ray_direction(q, n_ccw, c_mid)
        hits = _ray_hits_open_polyline(q, v, pts)
        for t_ray, s_cap, hit in hits:
            if s_cap <= s_eps or s_cap >= arc_total - s_eps:
                continue
            collected.append((s_cap, hit))
            break

    if len(collected) < 1:
        return None

    collected.sort(key=lambda x: x[0])
    merged: list[tuple[float, NDArray[np.float64]]] = []
    merge_tol = max(0.35 * target_step, 1e-7)
    for s_cap, hit in collected:
        if merged and abs(s_cap - merged[-1][0]) <= merge_tol:
            continue
        merged.append((s_cap, hit))

    seq: list[NDArray[np.float64]] = [p0.copy()]
    min_keep = max(0.5 * target_step, _MIN_SEGMENT_LENGTH_M)
    for _s_cap, hit in merged:
        if float(np.linalg.norm(hit - seq[-1])) >= min_keep:
            seq.append(hit.copy())
    if float(np.linalg.norm(p1 - seq[-1])) >= min_keep:
        seq.append(p1.copy())
    else:
        seq[-1] = p1.copy()

    new_pts = np.stack(seq, axis=0)
    if new_pts.shape[0] < 2:
        return None
    if not np.allclose(new_pts[0], p0, atol=1e-12, rtol=0.0) or not np.allclose(
        new_pts[-1], p1, atol=1e-12, rtol=0.0
    ):
        return None
    # region agent log
    _agent_debug_log(
        hypothesis_id="B",
        location="shell_inputs_from_section.py:_resample_one_cap_interior_from_skin_rays",
        message="cap interior replaced via skin rays",
        data={
            "label": str(cap_strip.label),
            "n_before": int(pts.shape[0]),
            "n_after": int(new_pts.shape[0]),
            "n_merged_hits": int(len(merged)),
            "n_collected": int(len(collected)),
        },
    )
    # endregion agent log
    return replace(cap_strip, midline_b=new_pts)


def _resample_cap_interiors_from_skin_rays(
    strips: tuple[ShellMidlineStrip, ...],
    *,
    bbox_margin_m: float = _SKIN_RAY_CAP_BBOX_MARGIN_M,
) -> tuple[ShellMidlineStrip, ...]:
    """Replace interior vertices of each cap using inward rays from skin master ring.

    Cap **endpoints** are left unchanged (skin–web junction snap is applied earlier
    in :func:`_split_skin_at_junctions`). If resampling yields too few hits for a
    cap, that strip is left unchanged.
    """
    skin_ring = _concat_skin_ring_vertices(strips)
    if skin_ring.shape[0] < 3:
        return strips

    skin_tangents = _skin_ring_vertex_tangents(skin_ring)
    if skin_tangents.shape[0] != skin_ring.shape[0]:
        return strips

    out: list[ShellMidlineStrip] = []
    for st in strips:
        if st.kind != "cap":
            out.append(st)
            continue
        new_st = _resample_one_cap_interior_from_skin_rays(
            st,
            skin_ring,
            skin_tangents,
            bbox_margin_m=bbox_margin_m,
        )
        out.append(new_st if new_st is not None else st)
    return tuple(out)


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


def _raise_missing_components_error(section: Any, layout_key: str | None) -> None:
    """Raise a descriptive ValueError when ``_components_unrotated`` is absent.

    Distinguishes two cases:
    * ``airfoil_sdf_only`` layout (0A / 0B) — geometry_mode flag detected via
      the layout_key registry or duck-typed from the section class name.
    * Broken multicell object — likely a programming error.
    """
    is_sdf_only = False

    # Try layout registry first (fastest signal when layout_key is provided).
    if layout_key is not None:
        try:
            from blade_precompute.orchestration.system_layout import resolve_system_type

            spec = resolve_system_type(layout_key)
            is_sdf_only = spec.geometry_mode == "airfoil_sdf_only"
        except (KeyError, Exception):
            pass

    # Fallback: duck-type by checking the section class name.
    if not is_sdf_only:
        cls_name = type(section).__name__
        if "OuterSkinOnly" in cls_name or "SdfOnly" in cls_name:
            is_sdf_only = True

    if is_sdf_only:
        raise ValueError(
            f"build_shell_mesh_inputs: section for layout {layout_key!r} uses "
            f"geometry_mode='airfoil_sdf_only' and does not expose "
            f"'_components_unrotated'. Subcomponent midline strips (skin, webs, "
            f"caps) are not available for this layout type. Use a multicell layout "
            f"(e.g. '2D-F', '1A-CN') or call build_shell_midline_strips only on "
            f"MultiCellSection objects."
        )
    raise ValueError(
        f"build_shell_mesh_inputs: section object of type {type(section)!r} does "
        f"not expose '_components_unrotated'. Expected a "
        f"MultiCellSection-compatible object produced by build_section_view with "
        f"a multicell layout. If you intended an airfoil_sdf_only layout (0A/0B), "
        f"pass the correct layout_key so the error message can guide you."
    )


def build_shell_mesh_inputs(
    section: Any,
    *,
    twist_rad: float,
    layout_key: str | None = None,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
    cap_resample_from_skin_rays: bool = True,
    skin_ray_cap_bbox_margin_m: float = _SKIN_RAY_CAP_BBOX_MARGIN_M,
) -> ShellMeshInputs:
    """Convert a ``MultiCellSection`` into a :class:`ShellMeshInputs` payload.

    Delegates midline extraction and B-frame rotation to
    :func:`~blade_precompute.section_geometry.interface.shell_midline_export.build_shell_midline_strips`,
    then appends airfoil LE/TE context.

    Parameters
    ----------
    section
        :class:`blade_precompute.section_geometry.sections.multicell.MultiCellSection`
        instance. Must expose:

        * ``airfoil`` — the underlying ``AirfoilSDF`` (used for
          chord length and LE/TE diagnostics);
        * ``_components_unrotated`` — chord-frame component objects
          (skin, webs, ...);
        * ``_spar_cap_components_unrotated`` — chord-frame spar cap
          objects.
    twist_rad
        Section twist in radians.
    layout_key
        Optional ``SystemTypeXY-Z`` key passed through for
        provenance / diagnostics.
    n_web_samples, n_cap_samples
        Number of points used to sample each web / cap midline.
    cap_resample_from_skin_rays
        When ``True`` (default), after :func:`_split_skin_at_junctions`, cap
        strip **interior** vertices are rebuilt from inward skin-normal ray
        hits; cap endpoints stay snapped at skin–web junctions. After that (or
        after split only), :func:`_insert_cap_web_interior_junctions` adds any
        missing cap–web T-junction vertices along continuous caps.
    skin_ray_cap_bbox_margin_m
        Axis-aligned bbox inflation (metres) when selecting skin vertices
        opposite each cap for ray casting.
    """
    airfoil_sdf = getattr(section, "airfoil", None)
    if airfoil_sdf is None:
        raise ValueError("section must expose an 'airfoil' attribute (AirfoilSDF).")
    chord_m = float(getattr(airfoil_sdf, "chord", 1.0))

    # B3 — detect airfoil_sdf_only layouts early and raise a targeted error,
    # rather than letting build_shell_midline_strips emit a generic message.
    if getattr(section, "_components_unrotated", None) is None:
        _raise_missing_components_error(section, layout_key)

    strips = build_shell_midline_strips(
        section,
        twist_rad=twist_rad,
        n_web_samples=n_web_samples,
        n_cap_samples=n_cap_samples,
    )
    _validate_strips(strips)
    # region agent log
    _caps_pre = {
        str(s.label): int(np.asarray(s.midline_b).shape[0])
        for s in strips
        if s.kind == "cap"
    }
    _agent_debug_log(
        hypothesis_id="C",
        location="shell_inputs_from_section.py:build_shell_mesh_inputs",
        message="cap vertex counts after build_shell_midline_strips",
        data={
            "layout_key": layout_key,
            "n_web_samples": n_web_samples,
            "n_cap_samples": n_cap_samples,
            "caps": _caps_pre,
        },
    )
    # endregion agent log
    strips = _split_skin_at_junctions(strips)
    _validate_strips(strips)
    if cap_resample_from_skin_rays:
        strips = _resample_cap_interiors_from_skin_rays(
            strips,
            bbox_margin_m=skin_ray_cap_bbox_margin_m,
        )
        _validate_strips(strips)
        # region agent log
        _caps_post = {
            str(s.label): int(np.asarray(s.midline_b).shape[0])
            for s in strips
            if s.kind == "cap"
        }
        _agent_debug_log(
            hypothesis_id="C",
            location="shell_inputs_from_section.py:build_shell_mesh_inputs",
            message="cap vertex counts after skin-ray resample",
            data={"layout_key": layout_key, "caps": _caps_post},
        )
        # endregion agent log

    strips = _insert_cap_web_interior_junctions(strips)
    _validate_strips(strips)

    le_S = np.asarray(airfoil_sdf.leading_edge, dtype=float).reshape(1, 2)
    te_S = np.asarray(airfoil_sdf.trailing_edge, dtype=float).reshape(1, 2)
    le_b = tuple(float(v) for v in rotate_chord_to_blade(le_S, twist_rad)[0])
    te_b = tuple(float(v) for v in rotate_chord_to_blade(te_S, twist_rad)[0])

    return ShellMeshInputs(
        chord_m=chord_m,
        twist_rad=float(twist_rad),
        layout_key=layout_key,
        midlines=list(strips),
        leading_edge_b=le_b,  # type: ignore[arg-type]
        trailing_edge_b=te_b,  # type: ignore[arg-type]
    )


__all__ = [
    "ShellMeshInputs",
    "ShellMidlineStrip",
    "SubcomponentMidline",
    "build_shell_mesh_inputs",
]
