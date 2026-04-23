"""
Synthetic tapered-blade stiffness fixture — solver regression testing only.

All stiffness values (EA, EIy, EIz, GJ, kAy, kAz, Kww) are polynomial fits
chosen to approximate a generic 12 m wind/tidal blade. They are NOT derived
from cross-section geometry or material properties.

For production use, build :class:`~blade_precompute.global_beam_model.core.types.SectionStation`
rows from ``section_properties`` outputs (``stations_from_arrays``) or from GBT in ``examples/section_beam_model``.
"""

from __future__ import annotations

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays


def _tapered_K7(z_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z0, z1 = float(z_nodes[0]), float(z_nodes[-1])
    n = z_nodes.shape[0]
    mats6 = np.zeros((n, 6, 6), dtype=np.float64)
    mats7 = np.zeros((n, 7, 7), dtype=np.float64)
    for i, z in enumerate(z_nodes):
        t = (z - z0) / (z1 - z0 + 1e-30)
        EA = 8.0e9 * (1.0 - 0.55 * t)
        EIy = 5.0e6 * (1.0 - 0.45 * t)
        EIz = 12.0e6 * (1.0 - 0.40 * t)
        GJ = 4.0e6 * (1.0 - 0.35 * t)
        kAy = 5.0e5 * (1.0 - 0.2 * t)
        kAz = 5.0e5 * (1.0 - 0.2 * t)
        Kww = 8.0e5 * (1.0 - 0.25 * t)
        mats6[i, 0, 0] = EA
        mats6[i, 1, 1] = EIy
        mats6[i, 2, 2] = EIz
        mats6[i, 3, 3] = GJ
        mats6[i, 4, 4] = kAy
        mats6[i, 5, 5] = kAz
        mats7[i, :6, :6] = mats6[i]
        mats7[i, 6, 6] = Kww
    return mats6, mats7


def _smoke_model() -> bm.BeamModel:
    L = 12.0
    n_st = 5
    z_st = np.linspace(0.0, L, n_st)
    x_pre = 0.02 * (z_st / L) ** 2
    r_ref = np.stack([x_pre, np.zeros_like(z_st), z_st], axis=1)
    kappa0 = np.zeros((n_st, 3), dtype=np.float64)
    for k in range(1, n_st - 1):
        zm, z0, z2 = z_st[k], z_st[k - 1], z_st[k + 1]
        xm = x_pre[k]
        x0 = x_pre[k - 1]
        x2 = x_pre[k + 1]
        d2x = ((x2 - xm) / (z2 - zm) - (xm - x0) / (zm - z0)) / (0.5 * (z2 - z0))
        kappa0[k, 1] = float(-d2x)
    geom = BladeGeometry(
        z_stations=z_st,
        r_ref=r_ref,
        kappa0=kappa0,
        tau0=np.zeros(n_st),
        chord=np.ones(n_st) * 0.5,
        twist=np.zeros(n_st),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2)),
        subcomponent_materials={},
        chi0=np.zeros(n_st),
    )
    n_nodes = 17
    K6s, K7s = _tapered_K7(z_st)
    stations = stations_from_arrays(z_st, K6s, K7s)
    return bm.BeamModel.from_blade_geometry(geom, n_nodes, stations, span_axis=2)
