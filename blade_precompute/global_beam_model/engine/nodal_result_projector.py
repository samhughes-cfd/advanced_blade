"""
Gauss-point to nodal projection for spanwise beam sampling.

Ports the element-consistent extrapolation idea from a generic nodal result
projector: ``values_g ≈ N_mat @ values_nodal`` with ``N_mat[g,:] = [N1,N2]``
from the same axial Lagrange ``_shape`` as assembly, then averages contributions
at shared nodes. See ``element_gauss_shape_matrix``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from . import constitutive
from .element import element_gauss_shape_matrix, e7_vector, gp_z_coord, infer_z_node
from .interp import interp_K7
from ..core.types import BeamModel, NodeState, SectionStation


def _recover_nodal_from_gauss(
    N_mat: NDArray[np.float64],
    values_g: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    For each column of ``values_g`` (shape n_gauss × n_comp), solve
    ``N_mat @ x = col`` or least-squares if overdetermined.

    Parameters
    ----------
    N_mat
        (n_gauss, 2)
    values_g
        (n_gauss, n_comp)
    """
    n_gauss, n_nodes_elem = N_mat.shape
    n_comp = int(values_g.shape[1])
    out = np.zeros((n_nodes_elem, n_comp), dtype=np.float64)
    for j in range(n_comp):
        rhs = values_g[:, j]
        if n_gauss == n_nodes_elem:
            out[:, j] = np.linalg.solve(N_mat, rhs)
        else:
            out[:, j] = np.linalg.lstsq(N_mat, rhs, rcond=None)[0]
    return out


def project_beam_strains_resultants_to_nodes(
    model: BeamModel,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Project Gauss-sampled 7-strains and 7-resultants to mesh nodes and average
    patch contributions (same weights as standard nodal smoothing).

    Returns
    -------
    z_nodal, strains_nodal, resultants_nodal
        ``z_nodal`` (n_nodes,), ``strains_nodal`` and ``resultants_nodal``
        (n_nodes, 7).
    """
    n_nodes = model.n_nodes
    strains_acc = np.zeros((n_nodes, 7), dtype=np.float64)
    resultants_acc = np.zeros((n_nodes, 7), dtype=np.float64)
    weight = np.zeros(n_nodes, dtype=np.float64)

    xi_w, _, N_mat = element_gauss_shape_matrix(n_gauss)

    for el in model.elements:
        node_ids = el.node_ids
        es = np.zeros((n_gauss, 7), dtype=np.float64)
        Rs = np.zeros((n_gauss, 7), dtype=np.float64)
        for g, xi in enumerate(xi_w):
            zg = gp_z_coord(model, el, float(xi))
            K7 = interp_K7(np.array([zg], dtype=np.float64), stations)[0]
            e = e7_vector(model, el, nodes, float(xi), fd_h)
            es[g] = e
            Rs[g] = constitutive.section_resultants_seven(K7, e)

        e_nodal = _recover_nodal_from_gauss(N_mat, es)
        r_nodal = _recover_nodal_from_gauss(N_mat, Rs)

        for a, nid in enumerate(node_ids):
            strains_acc[int(nid), :] += e_nodal[a, :]
            resultants_acc[int(nid), :] += r_nodal[a, :]
            weight[int(nid)] += 1.0

    nz = weight > 0.0
    strains_acc[nz] /= weight[nz, np.newaxis]
    resultants_acc[nz] /= weight[nz, np.newaxis]

    z_nodal = infer_z_node(model)
    return z_nodal, strains_acc, resultants_acc
