"""
Linear ``K7`` beam driver for optimisation (Tier B).

The extreme-load envelope is taken as **prescribed internal resultants** at
tabulated stations (common for ultimate blade checks). The optional seventh
component is the bimoment ``B`` (defaults to zero unless ``ExtremeLoads.B`` is
set). ``nodal_R`` applies a level-1 rigid rotation from ``kappa0`` at each station
via :func:`global_beam_model.engine.kinematics.rotmat_from_small_curvature`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.engine.kinematics import rotmat_from_small_curvature

from ..core.types import ExtremeLoads, OptimBladeGeometry


@dataclass
class PrescribedResultantBeamState:
    resultants: NDArray[np.float64]
    nodal_R: NDArray[np.float64]
    nodal_R_source: str = "small_curvature_kappa0"


def solve(
    K7_stack: NDArray[np.float64],
    extreme_loads: ExtremeLoads,
    blade_geometry: OptimBladeGeometry,
    *,
    nodal_R_override: NDArray[np.float64] | None = None,
) -> PrescribedResultantBeamState:
    """
    Parameters
    ----------
    K7_stack
        ``(n_s, 7, 7)`` section stiffness tables (used for consistency checks;
        internal resultants follow ``ExtremeLoads`` directly in this driver).
    """
    n_s = int(blade_geometry.z_stations.shape[0])
    if K7_stack.shape[0] != n_s:
        raise ValueError("K7_stack first axis must match number of stations.")
    B = extreme_loads.bimoment()
    R = np.stack(
        [
            extreme_loads.N,
            extreme_loads.My,
            extreme_loads.Mz,
            extreme_loads.T,
            extreme_loads.Vy,
            extreme_loads.Vz,
            B,
        ],
        axis=1,
    ).astype(np.float64)
    if nodal_R_override is not None:
        nodal_R = np.asarray(nodal_R_override, dtype=np.float64)
        if nodal_R.shape != (n_s, 3, 3):
            raise ValueError("nodal_R_override must have shape (n_station, 3, 3).")
        return PrescribedResultantBeamState(
            resultants=R, nodal_R=nodal_R, nodal_R_source="override"
        )
    nodal_R = np.zeros((n_s, 3, 3), dtype=np.float64)
    for i in range(n_s):
        nodal_R[i] = rotmat_from_small_curvature(blade_geometry.kappa0[i])
    return PrescribedResultantBeamState(
        resultants=R, nodal_R=nodal_R, nodal_R_source="small_curvature_kappa0"
    )
