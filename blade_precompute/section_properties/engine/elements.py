"""
Two-node line strip elements: composite (ABD) or isotropic membrane (C_iso).

Laminate / stiffness is expressed in **beam–tangent** shell axes:
local **1** = beam axis **x** (out of section), local **2** = midsurface tangent
in the **y–z** plane. Ply angles in :class:`~section_model.engine.laminate.LaminateDefinition`
are measured from local **1** (spanwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import SubcomponentGeometry
from .laminate import LaminateDefinition
from .materials import IsotropicMaterial, plane_stress_Q_isotropic
from .mesh import LineMesh


def _G_section_composite(lam: LaminateDefinition) -> float:
    """Representative section-plane shear stiffness [Pa] for warping Laplacian."""
    if not lam.plies:
        return 1.0
    ply0, _ = lam.plies[0]
    # G23 not in OrthotropicPly — use G12 as in-plane shear proxy for graph diffusion
    return float(max(ply0.G12, 1.0))


def _G_section_isotropic(mat: IsotropicMaterial) -> float:
    return float(mat.E / (2.0 * (1.0 + max(mat.nu, 1e-6))))


@dataclass
class StripElementData:
    """Per-edge quantities for assembly and recovery."""

    n_edges: int
    G: NDArray[np.float64]  # (n_e,)
    L: NDArray[np.float64]
    b: NDArray[np.float64]  # strip width
    y_mid: NDArray[np.float64]
    z_mid: NDArray[np.float64]
    ty: NDArray[np.float64]  # unit tangent y
    tz: NDArray[np.float64]  # unit tangent z
    subcomp_idx: NDArray[np.int32]
    is_composite: NDArray[np.bool_]
    ABD: NDArray[np.float64]  # (n_e, 6, 6) or zeros for iso
    C_iso: NDArray[np.float64]  # (n_e, 3, 3) or zeros for comp
    t_membrane: NDArray[np.float64]  # isotropic thickness
    sigma_allow: NDArray[np.float64]
    E_axial: NDArray[np.float64]  # effective E for K_ww / k_w


def build_strip_fe_data(section: SectionDefinition, mesh: LineMesh) -> StripElementData:
    n_e = mesh.edges.shape[0]
    if n_e == 0:
        z = np.zeros((0, 6, 6), dtype=np.float64)
        return StripElementData(
            n_edges=0,
            G=np.zeros(0),
            L=np.zeros(0),
            b=np.zeros(0),
            y_mid=np.zeros(0),
            z_mid=np.zeros(0),
            ty=np.zeros(0),
            tz=np.zeros(0),
            subcomp_idx=np.zeros(0, dtype=np.int32),
            is_composite=np.zeros(0, dtype=bool),
            ABD=z,
            C_iso=np.zeros((0, 3, 3)),
            t_membrane=np.zeros(0),
            sigma_allow=np.zeros(0),
            E_axial=np.zeros(0),
        )

    G = np.zeros(n_e, dtype=np.float64)
    L = mesh.edge_lengths.copy()
    b = np.zeros(n_e, dtype=np.float64)
    y_mid = np.zeros(n_e, dtype=np.float64)
    z_mid = np.zeros(n_e, dtype=np.float64)
    ty = np.zeros(n_e, dtype=np.float64)
    tz = np.zeros(n_e, dtype=np.float64)
    sub = mesh.edge_subcomp.astype(np.int32)
    is_comp = np.zeros(n_e, dtype=bool)
    ABD = np.zeros((n_e, 6, 6), dtype=np.float64)
    C_iso = np.zeros((n_e, 3, 3), dtype=np.float64)
    t_mem = np.zeros(n_e, dtype=np.float64)
    sig_allow = np.zeros(n_e, dtype=np.float64)
    E_ax = np.zeros(n_e, dtype=np.float64)

    nodes = mesh.nodes
    for e in range(n_e):
        i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        p0 = nodes[i0]
        p1 = nodes[i1]
        si = int(mesh.edge_subcomp[e])
        subc: SubcomponentGeometry = section.subcomponents[si]
        is_comp[e] = subc.is_composite
        dy, dz = p1[0] - p0[0], p1[1] - p0[1]
        le = float(np.hypot(dy, dz))
        if le < 1e-18:
            L[e] = 1e-18
            le = L[e]
        ty[e], tz[e] = dy / le, dz / le
        y_mid[e] = 0.5 * (p0[0] + p1[0])
        z_mid[e] = 0.5 * (p0[1] + p1[1])
        bw = subc.effective_strip_width()
        b[e] = bw

        if subc.is_composite:
            lam = subc.material
            assert isinstance(lam, LaminateDefinition)
            ABD[e] = lam.build_ABD()
            G[e] = _G_section_composite(lam)
            ply0, _ = lam.plies[0]
            E_ax[e] = float(ply0.E1)
        else:
            mat = subc.material
            assert isinstance(mat, IsotropicMaterial)
            C_iso[e] = plane_stress_Q_isotropic(mat.E, mat.nu)
            G[e] = _G_section_isotropic(mat)
            t_mem[e] = float(max(subc.thickness, 1e-12))
            sig_allow[e] = float(mat.sigma_allow)
            E_ax[e] = float(mat.E)

    return StripElementData(
        n_edges=n_e,
        G=G,
        L=L,
        b=b,
        y_mid=y_mid,
        z_mid=z_mid,
        ty=ty,
        tz=tz,
        subcomp_idx=sub,
        is_composite=is_comp,
        ABD=ABD,
        C_iso=C_iso,
        t_membrane=t_mem,
        sigma_allow=sig_allow,
        E_axial=E_ax,
    )
