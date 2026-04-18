"""
beam_model/interp.py
====================
Spanwise interpolation of ``K6(z)``, ``K7(z)``, and resampling of fields.
"""

from __future__ import annotations

from typing import List, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray

from ..core.types import K7Array, SectionStation
from ..k7_interpolation import K7Interpolator


def _sort_stations(stations: Sequence[SectionStation]) -> tuple[NDArray[np.float64], List[int]]:
    zs = np.array([s.z for s in stations], dtype=np.float64)
    order = np.argsort(zs)
    return zs[order], [int(i) for i in order]


def interp_matrix(
    z_query: NDArray[np.float64],
    stations: Sequence[SectionStation],
) -> NDArray[np.float64]:
    """Piecewise-linear ``K6`` at ``z_query``; shape ``(n, 6, 6)``."""
    if len(stations) < 2:
        raise ValueError("interp_matrix requires at least two SectionStation entries.")
    zs, order = _sort_stations(stations)
    mats = np.stack([stations[i].K6 for i in order], axis=0)

    zq = np.asarray(z_query, dtype=np.float64).ravel()
    out = np.zeros((zq.shape[0], 6, 6), dtype=np.float64)
    for k, z in enumerate(zq):
        if z <= zs[0]:
            out[k] = mats[0]
        elif z >= zs[-1]:
            out[k] = mats[-1]
        else:
            j = int(np.searchsorted(zs, z, side="right"))
            z0, z1 = zs[j - 1], zs[j]
            a = (z - z0) / (z1 - z0)
            out[k] = (1.0 - a) * mats[j - 1] + a * mats[j]
    return out


def interp_K7(
    z_query: NDArray[np.float64],
    stations: Sequence[SectionStation],
) -> NDArray[np.float64]:
    """PCHIP-interpolated ``(7,7)`` stiffness at ``z_query`` (upper triangle, symmetrised)."""
    if len(stations) < 2:
        raise ValueError("interp_K7 requires at least two SectionStation entries.")
    zs, order = _sort_stations(stations)
    mats = []
    n_synth = 0
    for idx in order:
        s = stations[idx]
        k6 = s.K6
        k7 = s.K7
        if k7 is None:
            from .constitutive import synthesize_K7

            mats.append(synthesize_K7(k6, None))
            n_synth += 1
        else:
            mats.append(np.asarray(k7, dtype=np.float64).reshape(7, 7))
    if n_synth:
        warnings.warn(
            f"interp_K7 synthesised warping blocks for {n_synth} station(s); supply full K7 for production fidelity.",
            UserWarning,
            stacklevel=2,
        )
    mats_arr = np.stack(mats, axis=0)
    k7_array = K7Array(s=zs, entries=mats_arr)
    zq = np.asarray(z_query, dtype=np.float64).ravel()
    out_arr = K7Interpolator(k7_array).interpolate(zq, allow_extrapolation=False)
    return np.asarray(out_arr.entries, dtype=np.float64)


def interp_scalar_stations(
    z_query: NDArray[np.float64],
    z_tab: NDArray[np.float64],
    y_tab: NDArray[np.float64],
) -> NDArray[np.float64]:
    zq = np.asarray(z_query, dtype=np.float64).ravel()
    zt = np.asarray(z_tab, dtype=np.float64).ravel()
    yt = np.asarray(y_tab, dtype=np.float64)
    order = np.argsort(zt)
    zt = zt[order]
    yt = yt[order]
    return np.interp(zq, zt, yt)


def sample_field_at_z(
    z_query: NDArray[np.float64],
    z_src: NDArray[np.float64],
    field: NDArray[np.float64],
) -> NDArray[np.float64]:
    zq = np.asarray(z_query, dtype=np.float64).ravel()
    zs = np.asarray(z_src, dtype=np.float64).ravel()
    f = np.asarray(field, dtype=np.float64)
    if f.shape[0] != zs.shape[0]:
        raise ValueError("field first dimension must match z_src.")
    flat = f.reshape(f.shape[0], -1)
    cols = []
    for c in range(flat.shape[1]):
        cols.append(np.interp(zq, zs, flat[:, c]))
    out = np.stack(cols, axis=1).reshape((zq.shape[0],) + f.shape[1:])
    return out


def stations_from_arrays(
    z: NDArray[np.float64],
    K6: NDArray[np.float64],
    K7: NDArray[np.float64] | None = None,
) -> List[SectionStation]:
    """Build ``SectionStation`` list from tabulated ``z`` and ``K6`` / optional ``K7``."""
    z = np.asarray(z, dtype=np.float64).ravel()
    K = np.asarray(K6, dtype=np.float64)
    if K.ndim == 2:
        K = K[None, ...]
    if K.shape[0] != z.shape[0]:
        raise ValueError("K6 first dim must match z length.")
    if K7 is None:
        return [SectionStation(z=float(z[i]), K6=K[i].copy()) for i in range(z.shape[0])]
    K7a = np.asarray(K7, dtype=np.float64)
    if K7a.ndim == 2:
        K7a = K7a[None, ...]
    if K7a.shape[0] != z.shape[0]:
        raise ValueError("K7 first dim must match z length.")
    return [
        SectionStation(z=float(z[i]), K6=K[i].copy(), K7=K7a[i].copy()) for i in range(z.shape[0])
    ]
