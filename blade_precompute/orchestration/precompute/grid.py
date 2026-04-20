"""Span/grid resampling and station selection for precompute."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute.containers import LinspaceSpec, PrecomputeInputs


def station_indices(n: int, spec: str) -> list[int]:
    s = (spec or "").strip().lower()
    if not s:
        s = "root,mid,tip"
    keys = [k.strip() for k in s.split(",") if k.strip()]
    if any(k == "all" for k in keys):
        return list(range(max(0, n)))
    out: list[int] = []
    for k in keys:
        if k == "root":
            out.append(0)
        elif k == "mid":
            out.append(max(0, (n - 1) // 2))
        elif k == "tip":
            out.append(max(0, n - 1))
        elif k.startswith("every-"):
            step = int(k.split("-", 1)[1])
            if step <= 0:
                raise ValueError("every-k requires k>0.")
            out.extend(list(range(0, max(0, n), step)))
        else:
            try:
                out.append(int(k))
            except ValueError as e:
                raise ValueError(
                    f"Unknown station selector {k!r}. Use root,mid,tip,all,every-k or integer indices."
                ) from e
    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        ii = int(np.clip(i, 0, max(0, n - 1)))
        if ii not in seen:
            uniq.append(ii)
            seen.add(ii)
    return uniq


def linspace_from_spec(spec: LinspaceSpec) -> NDArray[np.float64]:
    n = int(spec.n)
    if n < 1:
        raise ValueError("LinspaceSpec.n must be >= 1.")
    return np.linspace(float(spec.z_min), float(spec.z_max), n, dtype=np.float64)


def interp_series(
    z_src: NDArray[np.float64], y_src: NDArray[np.float64], z_dst: NDArray[np.float64]
) -> NDArray[np.float64]:
    zs = np.asarray(z_src, dtype=np.float64).ravel()
    ys = np.asarray(y_src, dtype=np.float64).ravel()
    zd = np.asarray(z_dst, dtype=np.float64).ravel()
    if zs.shape[0] != ys.shape[0]:
        raise ValueError("Interpolation source length mismatch.")
    if zs.shape[0] < 2:
        return np.full(zd.shape[0], float(ys[0]) if ys.size else 0.0, dtype=np.float64)
    return np.interp(zd, zs, ys)


def resample_precompute_inputs(inp: PrecomputeInputs, z_geom: NDArray[np.float64]) -> PrecomputeInputs:
    z0 = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    z1 = np.asarray(z_geom, dtype=np.float64).ravel()
    return dataclasses.replace(
        inp,
        span_r_z_m=z1,
        chord_m=interp_series(z0, inp.chord_m, z1),
        twist_deg=interp_series(z0, inp.twist_deg, z1),
        naca_m=interp_series(z0, inp.naca_m, z1),
        naca_p=interp_series(z0, inp.naca_p, z1),
        naca_xx=interp_series(z0, inp.naca_xx, z1),
    )


def resample_blade_geometry_to_z(bg: Any, z_struct: NDArray[np.float64]) -> Any:
    z_src = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_dst = np.asarray(z_struct, dtype=np.float64).ravel()
    r_ref = np.asarray(bg.r_ref, dtype=np.float64)
    kap = np.asarray(bg.kappa0, dtype=np.float64)
    airfoils = list(bg.airfoil_profiles)
    r_new = np.column_stack([interp_series(z_src, r_ref[:, j], z_dst) for j in range(r_ref.shape[1])])
    if r_new.shape[1] >= 3:
        r_new[:, 2] = z_dst
    k_new = np.column_stack([interp_series(z_src, kap[:, j], z_dst) for j in range(kap.shape[1])])
    af_new: list[Any] = []
    if len(airfoils) == z_src.shape[0]:
        for z in z_dst:
            i = int(np.argmin(np.abs(z_src - float(z))))
            af_new.append(airfoils[i])
    else:
        af_new = airfoils
    return dataclasses.replace(
        bg,
        z_stations=z_dst,
        r_ref=r_new,
        kappa0=k_new,
        tau0=interp_series(z_src, np.asarray(bg.tau0, dtype=np.float64), z_dst),
        chord=interp_series(z_src, np.asarray(bg.chord, dtype=np.float64), z_dst),
        twist=interp_series(z_src, np.asarray(bg.twist, dtype=np.float64), z_dst),
        airfoil_profiles=af_new,
    )


def require_columns(cols: Mapping[str, NDArray[np.float64]], required: Iterable[str], *, path: Path) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}. Present: {sorted(cols.keys())}")
