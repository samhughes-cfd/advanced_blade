"""
Midsurface polyline mesh: merge subcomponents into a global node/edge graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import SectionDefinition, SubcomponentGeometry


def _find_or_add_node(
    nodes: list[tuple[float, float]],
    index_map: dict[tuple[int, int], int],
    y: float,
    z: float,
    tol: float,
) -> int:
    """Quantised merge: snap to integer grid of tol."""
    key = (int(round(y / tol)), int(round(z / tol)))
    if key in index_map:
        return index_map[key]
    idx = len(nodes)
    nodes.append((y, z))
    index_map[key] = idx
    return idx


@dataclass
class LineMesh:
    """Global midsurface graph."""

    nodes: NDArray[np.float64]  # (n_n, 2)
    edges: NDArray[np.int32]  # (n_e, 2) node indices
    edge_subcomp: NDArray[np.int32]  # (n_e,) index into section.subcomponents
    edge_lengths: NDArray[np.float64]  # (n_e,)
    merge_tolerance: float


def build_line_mesh(section: SectionDefinition, merge_tolerance: float = 1e-6) -> LineMesh:
    """Concatenate all subcomponent polylines into edges with shared nodes at endpoints."""
    nodes: list[tuple[float, float]] = []
    index_map: dict[tuple[int, int], int] = {}
    edges_list: list[tuple[int, int, int]] = []

    for si, sub in enumerate(section.subcomponents):
        pts = sub.midsurface_coords
        if pts.shape[0] < 2:
            continue
        for k in range(pts.shape[0] - 1):
            y0, z0 = float(pts[k, 0]), float(pts[k, 1])
            y1, z1 = float(pts[k + 1, 0]), float(pts[k + 1, 1])
            i0 = _find_or_add_node(nodes, index_map, y0, z0, merge_tolerance)
            i1 = _find_or_add_node(nodes, index_map, y1, z1, merge_tolerance)
            if i0 == i1:
                continue
            edges_list.append((i0, i1, si))

    if not nodes:
        return LineMesh(
            nodes=np.zeros((0, 2), dtype=np.float64),
            edges=np.zeros((0, 2), dtype=np.int32),
            edge_subcomp=np.zeros((0,), dtype=np.int32),
            edge_lengths=np.zeros((0,), dtype=np.float64),
            merge_tolerance=merge_tolerance,
        )

    n_arr = np.array(nodes, dtype=np.float64)
    e_arr = np.array([[e[0], e[1]] for e in edges_list], dtype=np.int32)
    sub = np.array([e[2] for e in edges_list], dtype=np.int32)
    p0 = n_arr[e_arr[:, 0]]
    p1 = n_arr[e_arr[:, 1]]
    L = np.linalg.norm(p1 - p0, axis=1)
    return LineMesh(
        nodes=n_arr,
        edges=e_arr,
        edge_subcomp=sub,
        edge_lengths=L,
        merge_tolerance=merge_tolerance,
    )


def subcomponents_by_type(
    section: SectionDefinition,
) -> tuple[list[int], list[int]]:
    """Return indices of composite and isotropic subcomponents."""
    comp, iso = [], []
    for i, s in enumerate(section.subcomponents):
        if s.is_composite:
            comp.append(i)
        else:
            iso.append(i)
    return comp, iso
