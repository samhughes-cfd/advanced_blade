"""Span/grid resampling and station selection for precompute."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute.containers import LinspaceSpec, PrecomputeInputs


def job_span_z_m(inp: PrecomputeInputs) -> tuple[float, float]:
    """Physical root/tip [m] along spanwise ``z`` from the loaded spanwise table (``span_r_z_m``)."""
    z = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    if z.size < 1:
        raise ValueError("PrecomputeInputs.span_r_z_m must contain at least one station.")
    return float(z[0]), float(z[-1])


def warn_geometry_shorter_than_job_span(
    z_tip_m: float,
    geometry_z_max_m: float,
    *,
    blade_spec: Path | str | None = None,
    tol: float = 1e-6,
) -> None:
    """Delegate to :func:`warn_job_span_exceeds_geometry` (blade geometry spec as reference)."""
    warn_job_span_exceeds_geometry(
        z_tip_m,
        geometry_z_max_m,
        source="blade spec",
        spec=str(blade_spec) if blade_spec is not None else None,
        tol=tol,
    )


def station_indices(n: int, spec: str) -> list[int]:
    """Resolve ``spec`` into indices for 2D section / shell / station PNGs (``GridConfig.section_plot_station_spec``).

    ``n`` must be the number of spanwise stations in the :class:`PrecomputeInputs` passed to the stage
    (in ``main_precompute`` this is the structural resample, length ``N_STRUCTURAL``). Keywords
    ``all`` and ``structural`` select every index ``0..n-1``.
    """
    s = (spec or "").strip().lower()
    if not s:
        s = "root,mid,tip"
    keys = [k.strip() for k in s.split(",") if k.strip()]
    if any(k in ("all", "structural") for k in keys):
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
                    f"Unknown station selector {k!r}. Use root,mid,tip,all,structural,every-k or integer indices."
                ) from e
    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        ii = int(np.clip(i, 0, max(0, n - 1)))
        if ii not in seen:
            uniq.append(ii)
            seen.add(ii)
    return uniq


def station_subdir_name(i: int, z_m: float) -> str:
    """Directory name for per-station artefacts under a stage output folder (``station_iNNN_zZ.ZZZ``)."""
    return f"station_i{int(i):03d}_z{float(z_m):.3f}"


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


def warn_job_span_exceeds_geometry(
    z_tip_m: float,
    geometry_z_max_m: float,
    *,
    source: str = "reference geometry",
    spec: str | None = None,
    tol: float = 1e-6,
) -> None:
    """Warn once if the structural job span extends past the last station of built/loaded blade geometry."""
    if float(z_tip_m) <= float(geometry_z_max_m) + float(tol):
        return
    detail = f" ({spec})" if spec is not None else ""
    warnings.warn(
        f"Spanwise z_tip={z_tip_m:.6g} m exceeds last {source} z={geometry_z_max_m:.6g} m{detail}; "
        "clamped/flat extrapolation may apply past the last geometry station.",
        UserWarning,
        stacklevel=2,
    )


def resample_precompute_inputs(inp: PrecomputeInputs, z_geom: NDArray[np.float64]) -> PrecomputeInputs:
    z0 = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    z1 = np.asarray(z_geom, dtype=np.float64).ravel()
    return dataclasses.replace(
        inp,
        span_r_z_m=z1,
        radial_r_m=interp_series(z0, inp.radial_r_m, z1),
        chord_m=interp_series(z0, inp.chord_m, z1),
        twist_deg=interp_series(z0, inp.twist_deg, z1),
        kappa0_x=interp_series(z0, inp.kappa0_x, z1),
        kappa0_y=interp_series(z0, inp.kappa0_y, z1),
        kappa0_z=interp_series(z0, inp.kappa0_z, z1),
        naca_m=interp_series(z0, inp.naca_m, z1),
        naca_p=interp_series(z0, inp.naca_p, z1),
        naca_xx=interp_series(z0, inp.naca_xx, z1),
        naca_series=np.asarray(
            np.clip(
                np.round(interp_series(z0, inp.naca_series.astype(np.float64), z1)),
                4.0,
                6.0,
            ),
            dtype=np.int64,
        ),
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
        chord=interp_series(z_src, np.asarray(bg.chord, dtype=np.float64), z_dst),
        twist=interp_series(z_src, np.asarray(bg.twist, dtype=np.float64), z_dst),
        airfoil_profiles=af_new,
    )


def require_columns(cols: Mapping[str, NDArray[np.float64]], required: Iterable[str], *, path: Path) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}. Present: {sorted(cols.keys())}")
