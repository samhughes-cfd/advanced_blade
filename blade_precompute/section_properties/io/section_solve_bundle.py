"""Save full :class:`~blade_precompute.section_properties.core.types.SectionSolveResult` stacks to NPZ + JSON sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.core.types import SectionSolveResult


def _shapes_ok(seq: Sequence[NDArray[np.float64]]) -> bool:
    if not seq:
        return True
    s0 = seq[0].shape
    return all(a.shape == s0 for a in seq)


def _stack(seq: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    return np.stack([np.asarray(a, dtype=np.float64) for a in seq], axis=0)


def save_section_solve_stations_bundle(
    out_dir: Path,
    z_stations: NDArray[np.float64],
    results: Sequence[SectionSolveResult],
) -> dict[str, Any]:
    """
    Write midsurface section solve results for all stations.

    When all per-station array shapes match, writes a single ``section_solve_stations.npz``
    plus ``section_solve_station_meta.json`` (subcomponent names, bundle mode).

    If any stacked array would be ragged, writes one ``station_XXXX.npz`` per station under
    ``section_solve_npz/`` and records paths in the meta JSON.

    Returns a JSON-serialisable dict suitable for ``section_properties/summary.json``
    (e.g. ``section_solve_bundle`` key).
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    n = len(results)
    if z.shape[0] != n:
        raise ValueError(f"z_stations length {z.shape[0]} != n results {n}")

    meta: dict[str, Any] = {
        "n_station": int(n),
        "z_stations": z.tolist(),
        "composite_subcomp_names": [list(r.composite_subcomp_names) for r in results],
        "isotropic_subcomp_names": [list(r.isotropic_subcomp_names) for r in results],
    }

    array_keys = [
        "K6",
        "K7",
        "M6",
        "warping_function",
        "k_w",
        "composite_resultant_basis",
        "isotropic_resultant_basis",
        "ABD_inv",
        "Q_bar",
        "T_ply",
        "z_ply",
        "iso_thickness",
        "iso_C",
        "iso_sigma_allow",
        "Zt",
        "S13",
        "S23",
    ]
    can_stack = True
    for key in array_keys:
        arrs = [getattr(r, key) for r in results]
        if not _shapes_ok(arrs):
            can_stack = False
            break

    eob_list = [r.E_omega_basis for r in results]
    if any(x is None for x in eob_list) and not all(x is None for x in eob_list):
        can_stack = False
    elif all(x is not None for x in eob_list):
        eob_arrs = [np.asarray(x, dtype=np.float64) for x in eob_list]
        if not _shapes_ok(eob_arrs):
            can_stack = False

    if can_stack:
        npz_path = (out_dir / "section_solve_stations.npz").resolve()
        payload: dict[str, Any] = {"z_stations": z}
        for key in array_keys:
            payload[f"{key}_stack"] = _stack([getattr(r, key) for r in results])
        if all(x is not None for x in eob_list):
            payload["E_omega_basis_stack"] = _stack([np.asarray(x, dtype=np.float64) for x in eob_list])
        payload["K_ww"] = np.asarray([float(r.K_ww) for r in results], dtype=np.float64)
        payload["k_y"] = np.asarray([float(r.k_y) for r in results], dtype=np.float64)
        payload["k_z"] = np.asarray([float(r.k_z) for r in results], dtype=np.float64)
        payload["area"] = np.asarray([float(r.area) for r in results], dtype=np.float64)
        payload["mass_per_length"] = np.asarray([float(r.mass_per_length) for r in results], dtype=np.float64)
        payload["shear_center"] = _stack([r.shear_center for r in results])
        payload["mass_center"] = _stack([r.mass_center for r in results])
        payload["elastic_center"] = _stack([r.elastic_center for r in results])
        np.savez_compressed(npz_path, **payload)
        meta["mode"] = "stacked"
        meta["npz"] = str(npz_path)
        meta_json = (out_dir / "section_solve_station_meta.json").resolve()
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta["meta_json"] = str(meta_json)
        return meta

    sub = (out_dir / "section_solve_npz").resolve()
    sub.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, r in enumerate(results):
        p = (sub / f"station_{i:04d}.npz").resolve()
        eob = r.E_omega_basis
        kw: dict[str, Any] = {
            "z_station": np.array([float(z[i])], dtype=np.float64),
            "K6": np.asarray(r.K6, dtype=np.float64),
            "K7": np.asarray(r.K7, dtype=np.float64),
            "M6": np.asarray(r.M6, dtype=np.float64),
            "warping_function": np.asarray(r.warping_function, dtype=np.float64),
            "K_ww": np.array([float(r.K_ww)], dtype=np.float64),
            "k_w": np.asarray(r.k_w, dtype=np.float64),
            "composite_resultant_basis": np.asarray(r.composite_resultant_basis, dtype=np.float64),
            "isotropic_resultant_basis": np.asarray(r.isotropic_resultant_basis, dtype=np.float64),
            "ABD_inv": np.asarray(r.ABD_inv, dtype=np.float64),
            "Q_bar": np.asarray(r.Q_bar, dtype=np.float64),
            "T_ply": np.asarray(r.T_ply, dtype=np.float64),
            "z_ply": np.asarray(r.z_ply, dtype=np.float64),
            "iso_thickness": np.asarray(r.iso_thickness, dtype=np.float64),
            "iso_C": np.asarray(r.iso_C, dtype=np.float64),
            "iso_sigma_allow": np.asarray(r.iso_sigma_allow, dtype=np.float64),
            "Zt": np.asarray(r.Zt, dtype=np.float64),
            "S13": np.asarray(r.S13, dtype=np.float64),
            "S23": np.asarray(r.S23, dtype=np.float64),
            "area": np.array([float(r.area)], dtype=np.float64),
            "mass_per_length": np.array([float(r.mass_per_length)], dtype=np.float64),
            "shear_center": np.asarray(r.shear_center, dtype=np.float64),
            "mass_center": np.asarray(r.mass_center, dtype=np.float64),
            "elastic_center": np.asarray(r.elastic_center, dtype=np.float64),
            "k_y": np.array([float(r.k_y)], dtype=np.float64),
            "k_z": np.array([float(r.k_z)], dtype=np.float64),
        }
        if eob is not None:
            kw["E_omega_basis"] = np.asarray(eob, dtype=np.float64)
        np.savez_compressed(p, **kw)
        paths.append(str(p))
    meta["mode"] = "per_station"
    meta["npz_dir"] = str(sub)
    meta["npz_paths"] = paths
    meta_json = (out_dir / "section_solve_station_meta.json").resolve()
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["meta_json"] = str(meta_json)
    return meta
