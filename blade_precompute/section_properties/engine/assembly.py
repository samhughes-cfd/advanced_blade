"""
Sparse assembly for scalar warping on the midsurface line graph.

Element (2-node): :math:`K_e = (G b / L) \\begin{bmatrix}1&-1\\\\-1&1\\end{bmatrix}`.

RHS (constant body per edge, weak-form lumping):

.. math::

    f_e = \\frac{s_e L}{2}\\begin{bmatrix}1\\\\1\\end{bmatrix},\\quad
    s_e = G_e b_e\\bigl((z_m-z_e)t_y - (y_m-y_e)t_z\\bigr)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .elements import StripElementData
from .mesh import LineMesh


def assemble_line_laplacian(mesh: LineMesh, fe: StripElementData) -> sp.csr_matrix:
    n_n = mesh.nodes.shape[0]
    n_e = fe.n_edges
    if n_e == 0:
        return sp.csr_matrix((n_n, n_n))

    conduct = fe.G * fe.b / np.maximum(fe.L, 1e-18)
    # Element 2x2
    i0 = mesh.edges[:, 0]
    i1 = mesh.edges[:, 1]
    vals = np.empty(4 * n_e, dtype=np.float64)
    rws = np.empty(4 * n_e, dtype=np.int32)
    cls = np.empty(4 * n_e, dtype=np.int32)
    for k in range(n_e):
        c = conduct[k]
        base = 4 * k
        rws[base : base + 4] = [i0[k], i0[k], i1[k], i1[k]]
        cls[base : base + 4] = [i0[k], i1[k], i0[k], i1[k]]
        vals[base : base + 4] = [c, -c, -c, c]
    return sp.coo_matrix((vals, (rws, cls)), shape=(n_n, n_n)).tocsr()


def apply_pin_constraint(
    K: sp.csr_matrix,
    f: NDArray[np.float64],
    pin_node: int = 0,
) -> tuple[sp.csr_matrix, NDArray[np.float64]]:
    K_lil = K.tolil()
    K_lil[pin_node, :] = 0.0
    K_lil[:, pin_node] = 0.0
    K_lil[pin_node, pin_node] = 1.0
    f_pin = f.copy()
    f_pin[pin_node] = 0.0
    return K_lil.tocsr(), f_pin


def build_warping_rhs_line(
    mesh: LineMesh,
    fe: StripElementData,
    y_e: float,
    z_e: float,
) -> NDArray[np.float64]:
    n_n = mesh.nodes.shape[0]
    f = np.zeros(n_n, dtype=np.float64)
    for e in range(fe.n_edges):
        i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        G, b, L = fe.G[e], fe.b[e], fe.L[e]
        yc, zc = fe.y_mid[e], fe.z_mid[e]
        ty, tz = fe.ty[e], fe.tz[e]
        s = G * b * ((zc - z_e) * ty - (yc - y_e) * tz)
        lump = 0.5 * s * L
        f[i0] += lump
        f[i1] += lump
    return f
