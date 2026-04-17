"""
Blade reference geometry for geometrically exact beam models.

``r_ref`` is the **shear-centre locus** (default convention): the natural
reference axis for uncoupled bending when ``K7`` is assembled about the shear
centre. If a centroidal axis were used instead, elastic–inertial coupling would
appear in ``K7``; the caller must supply ``K7`` and ``r_ref`` in a **consistent**
frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import warnings

import numpy as np
from numpy.typing import NDArray

from ..core.types import BeamElement, BeamModel, SectionStation

MaterialAssignment = Any


@dataclass
class BladeGeometry:
    z_stations: NDArray[np.float64]  # (n_s,) spanwise arc coordinates [m]
    r_ref: NDArray[np.float64]  # (n_s, 3) shear-centre locus in global frame
    kappa0: NDArray[np.float64]  # (n_s, 3) initial curvature [kx, ky, kz] material frame
    tau0: NDArray[np.float64]  # (n_s,) initial twist rate [1/m]; merged into κ₀₁ where used
    chord: NDArray[np.float64]
    twist: NDArray[np.float64]
    airfoil_profiles: List[Any] = field(default_factory=list)
    web_positions: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float64)
    )
    subcomponent_materials: Dict[str, MaterialAssignment] = field(default_factory=dict)
    chi0: NDArray[np.float64] | None = None  # (n_s,) initial warping rate; default zeros

    def __post_init__(self) -> None:
        self.z_stations = np.asarray(self.z_stations, dtype=np.float64).ravel()
        self.r_ref = np.asarray(self.r_ref, dtype=np.float64)
        self.kappa0 = np.asarray(self.kappa0, dtype=np.float64)
        self.tau0 = np.asarray(self.tau0, dtype=np.float64).ravel()
        self.chord = np.asarray(self.chord, dtype=np.float64).ravel()
        self.twist = np.asarray(self.twist, dtype=np.float64).ravel()
        self.web_positions = np.asarray(self.web_positions, dtype=np.float64)
        if self.chi0 is None:
            self.chi0 = np.zeros_like(self.z_stations)
        else:
            self.chi0 = np.asarray(self.chi0, dtype=np.float64).ravel()


def _interp_columns(zq: NDArray, ztab: NDArray, ytab: NDArray) -> NDArray[np.float64]:
    """Linear interp each column of ytab (n_s, k) onto zq."""
    zq = np.asarray(zq, dtype=np.float64).ravel()
    zt = np.asarray(ztab, dtype=np.float64).ravel()
    y = np.asarray(ytab, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    out = np.zeros((zq.shape[0], y.shape[1]), dtype=np.float64)
    for j in range(y.shape[1]):
        out[:, j] = np.interp(zq, zt, y[:, j])
    return out


def beam_model_from_blade_geometry(
    geometry: BladeGeometry,
    n_nodes: int,
    section_stations: List[SectionStation],
    *,
    span_axis: int = 2,
    align_section_stations: bool = False,
) -> BeamModel:
    """
    Build a :class:`BeamModel` by resampling ``r_ref`` onto ``n_nodes`` along
    ``z_stations`` and tabulating ``kappa0``, ``chi0``, ``tau0`` on the mesh.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")
    zs = geometry.z_stations
    z0, z1 = float(zs[0]), float(zs[-1])
    z_node = np.linspace(z0, z1, n_nodes, dtype=np.float64)
    if align_section_stations:
        z_sec = np.asarray([s.z for s in section_stations], dtype=np.float64)
        z_sec = z_sec[(z_sec >= z0) & (z_sec <= z1)]
        z_node = np.unique(np.concatenate([z_node, z_sec], axis=0))
    if section_stations:
        z_sec = np.asarray([s.z for s in section_stations], dtype=np.float64)
        z_sec = np.sort(z_sec[(z_sec >= z0) & (z_sec <= z1)])
        if z_sec.size >= 2:
            min_dz_sec = float(np.min(np.diff(z_sec)))
            min_dz_node = float(np.min(np.diff(z_node)))
            if min_dz_node > 1.5 * min_dz_sec:
                warnings.warn(
                    "Beam node spacing is coarser than section-station spacing; consider align_section_stations=True.",
                    UserWarning,
                    stacklevel=2,
                )
    r_node = _interp_columns(z_node, zs, geometry.r_ref)
    kappa0_node = _interp_columns(z_node, zs, geometry.kappa0)
    tau0_node = np.interp(z_node, zs, geometry.tau0).reshape(-1, 1)
    chi0_node = np.interp(z_node, zs, geometry.chi0).ravel()
    # fold tau0 into first curvature component (twist about beam axis) in material frame
    kappa0_node[:, 0] = kappa0_node[:, 0] + tau0_node.ravel()

    elems: List[BeamElement] = []
    for e in range(n_nodes - 1):
        i, j = e, e + 1
        L0 = float(np.linalg.norm(r_node[j] - r_node[i]))
        if L0 < 1e-14:
            raise ValueError("Zero-length segment in resampled blade geometry.")
        z_mid = 0.5 * (z_node[i] + z_node[j])
        elems.append(BeamElement(node_ids=(i, j), L0=L0, z_mid=z_mid))

    return BeamModel(
        X_ref=r_node,
        elements=elems,
        section_stations=section_stations,
        span_axis=span_axis,
        z_node=z_node,
        kappa0_node=kappa0_node,
        chi0_node=chi0_node,
    )
