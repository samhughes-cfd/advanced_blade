"""
ASCII ``.dat`` I/O for spanwise distributed loads and operational time series.

Extreme (spanwise only)
-----------------------
Required columns (header row, ``#`` comments allowed):

- ``r_z_m`` — spanwise coordinate [m], **strictly increasing** root → tip
- ``q_y_Npm`` — flapwise line load [N/m]
- ``q_z_Npm`` — edgewise line load [N/m]
- ``m_x_Nmpm`` — distributed torque about beam ``x`` [N·m/m]

Optional bookkeeping column ``R_m`` is ignored if present.

Operational (long format)
-------------------------
One row per ``(t_s, r_z_m)`` sample:

- ``t_s`` — time [s]
- ``r_z_m``, ``q_y_Npm``, ``q_z_Npm``, ``m_x_Nmpm`` as above

Within each time slice, ``r_z_m`` must be strictly increasing and **unique** pairs
``(t_s, r_z_m)`` across the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blade_precompute.beam_model.engine.distributed_load_integrator import (
    DistributedLoadIntegrator,
    IntegratedResultants,
)
from blade_precompute.design_optimisation.core.types import ExtremeLoads
from blade_analysis.fatigue_damage.core.loads import ResultantHistory


def _read_dat_table(path: str | Path) -> tuple[NDArray[np.float64], list[str]]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").splitlines()
    header_idx = None
    for i, line in enumerate(raw):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("r_z_m") or s.lower().startswith("t_s"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header row found in {p}.")
    header = raw[header_idx].strip().split()
    data_lines = [ln for ln in raw[header_idx + 1 :] if ln.strip() and not ln.strip().startswith("#")]
    if not data_lines:
        raise ValueError(f"No data rows in {p}.")
    rows = []
    for ln in data_lines:
        parts = ln.replace(",", " ").split()
        rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(header):
        raise ValueError(f"Column count mismatch in {p}: header {len(header)} vs data {arr.shape}.")
    return arr, header


def _col_index(header: list[str], name: str) -> int:
    lower = [h.lower() for h in header]
    key = name.lower()
    try:
        return lower.index(key)
    except ValueError as e:
        raise ValueError(f"Missing required column {name!r} in header {header}.") from e


def validate_strictly_increasing_z(z: NDArray[np.float64], *, name: str = "r_z_m") -> None:
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size < 2:
        raise ValueError(f"{name}: need at least two stations.")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")


def validate_z_matches_geometry(
    z_load: NDArray[np.float64],
    z_geometry: NDArray[np.float64],
    *,
    tol: float = 1e-3,
) -> None:
    a = np.asarray(z_load, dtype=np.float64).ravel()
    b = np.asarray(z_geometry, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"Load z length {a.shape[0]} does not match geometry z_stations length {b.shape[0]}."
        )
    if not np.allclose(a, b, rtol=0.0, atol=tol):
        raise ValueError(
            f"Load r_z_m does not match OptimBladeGeometry.z_stations within tol={tol}."
        )


def load_extreme_distributed_loads_dat(
    path: str | Path,
    *,
    z_geometry: NDArray[np.float64] | None = None,
    z_match_tol: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load extreme distributed loads.

    Returns ``(z, q_y, q_z, m_x)`` each 1-D ``float64`` of length ``n_station``.
    """
    arr, header = _read_dat_table(path)
    iz = _col_index(header, "r_z_m")
    iqy = _col_index(header, "q_y_Npm")
    iqz = _col_index(header, "q_z_Npm")
    imx = _col_index(header, "m_x_Nmpm")
    z = arr[:, iz]
    q_y = arr[:, iqy]
    q_z = arr[:, iqz]
    m_x = arr[:, imx]
    validate_strictly_increasing_z(z)
    if z_geometry is not None:
        validate_z_matches_geometry(z, z_geometry, tol=z_match_tol)
    return z, q_y, q_z, m_x


def load_operational_distributed_loads_dat(
    path: str | Path,
    *,
    z_geometry: NDArray[np.float64] | None = None,
    z_match_tol: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load operational long-format table.

    Returns ``(t, z, q_y, q_z, m_x)`` where ``z`` is the station vector (same for all ``t``),
    and ``q_*`` have shape ``(n_t, n_z)`` with rows in ascending ``t`` order.
    """
    arr, header = _read_dat_table(path)
    it = _col_index(header, "t_s")
    iz = _col_index(header, "r_z_m")
    iqy = _col_index(header, "q_y_Npm")
    iqz = _col_index(header, "q_z_Npm")
    imx = _col_index(header, "m_x_Nmpm")
    t_vals = arr[:, it]
    z_vals = arr[:, iz]
    pairs = np.stack([t_vals, z_vals], axis=1)
    if pairs.shape[0] != np.unique(pairs, axis=0).shape[0]:
        raise ValueError("Duplicate (t_s, r_z_m) pairs in operational file.")
    uniq_t = np.unique(t_vals)
    n_t = int(uniq_t.size)
    z_ref: NDArray[np.float64] | None = None
    q_rows: list[NDArray[np.float64]] = []
    n_z_expect: int | None = None
    for tv in uniq_t:
        idx = np.nonzero(t_vals == tv)[0]
        sub = arr[idx]
        order = np.argsort(sub[:, iz], kind="mergesort")
        sub = sub[order]
        zk = np.asarray(sub[:, iz], dtype=np.float64).ravel()
        validate_strictly_increasing_z(zk)
        if z_ref is None:
            z_ref = zk.copy()
            if z_geometry is not None:
                validate_z_matches_geometry(z_ref, z_geometry, tol=z_match_tol)
        elif not np.allclose(zk, z_ref, rtol=0.0, atol=z_match_tol):
            raise ValueError("r_z_m grid differs between time slices.")
        n_z = zk.size
        if n_z_expect is None:
            n_z_expect = n_z
        elif n_z != n_z_expect:
            raise ValueError("Inconsistent number of z rows per time slice.")
        q_rows.append(
            np.concatenate(
                [
                    np.asarray(sub[:, iqy], dtype=np.float64).ravel()[:, None],
                    np.asarray(sub[:, iqz], dtype=np.float64).ravel()[:, None],
                    np.asarray(sub[:, imx], dtype=np.float64).ravel()[:, None],
                ],
                axis=1,
            )
        )
    assert z_ref is not None and n_z_expect is not None
    stacked = np.stack(q_rows, axis=0)
    q_y = stacked[:, :, 0]
    q_z = stacked[:, :, 1]
    m_x = stacked[:, :, 2]
    return uniq_t, z_ref, q_y, q_z, m_x


def extreme_loads_from_distributed(
    z: NDArray[np.float64],
    q_y: NDArray[np.float64],
    q_z: NDArray[np.float64],
    m_x: NDArray[np.float64],
) -> ExtremeLoads:
    """Integrate distributed loads → :class:`~design_optimisation.core.types.ExtremeLoads`."""
    res = DistributedLoadIntegrator.integrate(z, q_y, q_z, m_x)
    return extreme_loads_from_integrated(res)


def extreme_loads_from_integrated(res: IntegratedResultants) -> ExtremeLoads:
    B = np.zeros_like(res.N, dtype=np.float64)
    return ExtremeLoads(
        z_stations=res.z.copy(),
        N=res.N.copy(),
        Vy=res.Vy.copy(),
        Vz=res.Vz.copy(),
        My=res.My.copy(),
        Mz=res.Mz.copy(),
        T=res.T.copy(),
        B=B,
    )


def resultant_history_from_distributed(
    t: NDArray[np.float64],
    z: NDArray[np.float64],
    q_y: NDArray[np.float64],
    q_z: NDArray[np.float64],
    m_x: NDArray[np.float64],
) -> ResultantHistory:
    """Integrate each time row → :class:`~blade_analysis.fatigue_damage.core.loads.ResultantHistory`."""
    series = DistributedLoadIntegrator.integrate_timeseries(z, q_y, q_z, m_x)
    n_t = len(series)
    n_z = z.size
    N = np.zeros((n_t, n_z), dtype=np.float64)
    Vy = np.zeros((n_t, n_z), dtype=np.float64)
    Vz = np.zeros((n_t, n_z), dtype=np.float64)
    My = np.zeros((n_t, n_z), dtype=np.float64)
    Mz = np.zeros((n_t, n_z), dtype=np.float64)
    T = np.zeros((n_t, n_z), dtype=np.float64)
    for it, r in enumerate(series):
        N[it] = r.N
        Vy[it] = r.Vy
        Vz[it] = r.Vz
        My[it] = r.My
        Mz[it] = r.Mz
        T[it] = r.T
    B = np.zeros_like(N, dtype=np.float64)
    return ResultantHistory(
        z_stations=z.copy(),
        time=np.asarray(t, dtype=np.float64).ravel(),
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        B=B,
    )


def resultant_history_from_operational_dat(
    path: str | Path,
    *,
    z_geometry: NDArray[np.float64] | None = None,
    z_match_tol: float = 1e-3,
) -> ResultantHistory:
    """Parse operational ``.dat`` and return integrated :class:`~blade_analysis.fatigue_damage.core.loads.ResultantHistory`."""
    t, z, q_y, q_z, m_x = load_operational_distributed_loads_dat(
        path, z_geometry=z_geometry, z_match_tol=z_match_tol
    )
    return resultant_history_from_distributed(t, z, q_y, q_z, m_x)
