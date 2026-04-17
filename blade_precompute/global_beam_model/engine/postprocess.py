"""
beam_model/postprocess.py
=========================
Spanwise sampling of strains/resultants and reaction extraction.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .assembly import assemble_gradient, dof_base, external_load_vector
from .element import element_gp_cache
from .interp import sample_field_at_z
from ..core.types import BeamLoads, BeamModel, BoundaryCondition, NodeState, SectionStation


def collect_station_data(
    model: BeamModel,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Flatten Gauss-point samples, sort by span ``z``, shape
    ``(n_sample, 7)`` for strains and resultants.
    """
    z_list: List[float] = []
    e_list: List[NDArray[np.float64]] = []
    r_list: List[NDArray[np.float64]] = []
    for el in model.elements:
        zs, eev, rrv = element_gp_cache(model, el, nodes, stations, n_gauss, fd_h)
        for g in range(n_gauss):
            z_list.append(float(zs[g]))
            e_list.append(eev[g].copy())
            r_list.append(rrv[g].copy())
    z_flat = np.array(z_list, dtype=np.float64)
    e_flat = np.stack(e_list, axis=0)
    r_flat = np.stack(r_list, axis=0)
    order = np.argsort(z_flat)
    return z_flat[order], e_flat[order], r_flat[order]


def sample_resultants_at_z(
    z_query: NDArray[np.float64],
    z_src: NDArray[np.float64],
    resultants: NDArray[np.float64],
) -> NDArray[np.float64]:
    z_flat = np.asarray(z_src, dtype=np.float64).reshape(-1)
    r_flat = np.asarray(resultants, dtype=np.float64).reshape(-1, resultants.shape[-1])
    order = np.argsort(z_flat)
    return sample_field_at_z(z_query, z_flat[order], r_flat[order])


def compute_reactions(
    model: BeamModel,
    nodes: List[NodeState],
    loads: BeamLoads,
    bcs: List[BoundaryCondition],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
) -> Dict[Tuple[int, int], float]:
    F = external_load_vector(model, loads, n_gauss)
    g = assemble_gradient(model, nodes, stations, n_gauss, fd_h)
    r = F - g
    out: Dict[Tuple[int, int], float] = {}
    for bc in bcs:
        base = dof_base(bc.node_id)
        for local in bc.fixed_dofs:
            dof = base + int(local)
            out[(bc.node_id, int(local))] = float(r[dof])
    return out
