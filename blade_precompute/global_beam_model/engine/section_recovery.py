"""
Section stress/strain recovery on the blade ``station_z`` grid.

Uses :mod:`blade_utilities.recovery` — :class:`~blade_utilities.recovery.tensor_cache.cache.RecoveryCache`
for ply stresses, Hashin envelope FI, and von Mises (isotropic), plus operator helpers
``apply_strain_operator``, ``apply_section_stress_operator``, ``apply_span_derivative``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from blade_utilities.recovery import (
    RecoveryCache,
    RecoveryCacheBuilder,
    apply_section_stress_operator,
    apply_span_derivative,
    apply_strain_operator,
    build_recovery_operator_bundle,
)

from .constitutive import beam_resultants_to_section_recovery_order

if TYPE_CHECKING:
    from ..core.types import BeamSolveResult


def _interp_columns(z_src: NDArray[np.float64], vals: NDArray[np.float64], z_tgt: NDArray[np.float64]) -> NDArray[np.float64]:
    z_src = np.asarray(z_src, dtype=np.float64).ravel()
    vals = np.asarray(vals, dtype=np.float64)
    order = np.argsort(z_src)
    zs = z_src[order]
    vs = vals[order]
    out = np.zeros((z_tgt.shape[0], vals.shape[1]), dtype=np.float64)
    for j in range(vals.shape[1]):
        out[:, j] = np.interp(np.asarray(z_tgt, dtype=np.float64).ravel(), zs, vs[:, j])
    return out


def _interp_nodal_R(
    z_src: NDArray[np.float64], R_src: NDArray[np.float64], z_tgt: NDArray[np.float64]
) -> NDArray[np.float64]:
    z_src = np.asarray(z_src, dtype=np.float64).ravel()
    R_src = np.asarray(R_src, dtype=np.float64)
    order = np.argsort(z_src)
    zs = z_src[order]
    Rord = R_src[order]
    zt = np.asarray(z_tgt, dtype=np.float64).ravel()
    out = np.zeros((zt.shape[0], 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out[:, i, j] = np.interp(zt, zs, Rord[:, i, j])
    return out


def _aggregate_max_abs_voigt(sig: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n_case, n_s, n_comp, n_ply, 3)`` → ``(n_case, n_s, 3)``."""
    a = np.abs(np.asarray(sig, dtype=np.float64))
    return np.max(a, axis=(2, 3))


def _strain_max_abs_over_components(eps: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n_case, n_s, n_comp, 6)`` → ``(n_case, n_s, 6)``."""
    a = np.abs(np.asarray(eps, dtype=np.float64))
    return np.max(a, axis=2)


def _hashin_max_over_plies(fi: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n_case, n_s, n_comp, n_ply)`` → ``(n_case, n_s)``."""
    return np.max(np.asarray(fi, dtype=np.float64), axis=(2, 3))


def _von_mises_max_over_iso(fi: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n_case, n_s, n_iso)`` → ``(n_case, n_s)``."""
    return np.max(np.asarray(fi, dtype=np.float64), axis=2)


def _recovery_path_arrays(
    cache: RecoveryCache,
    bundle: Any,
    r_case: NDArray[np.float64],
) -> Dict[str, NDArray[np.float64] | None]:
    """Single (n_case=1) resultant strip → all derived spanwise series."""
    sig_mat = cache.recover_ply_stresses(r_case)
    voigt_mat = _aggregate_max_abs_voigt(sig_mat)[0]
    eps = apply_strain_operator(bundle, r_case)
    strain_max = _strain_max_abs_over_components(eps)[0]
    h_full_arr = cache.eval_hashin_fi(r_case)
    h_max = _hashin_max_over_plies(h_full_arr)[0]
    h_ply_env = np.max(np.asarray(h_full_arr, dtype=np.float64)[0], axis=1)
    vm_raw = cache.eval_von_mises_fi(r_case)
    if vm_raw.shape[2] == 0:
        vm_max = np.zeros(int(h_max.shape[0]), dtype=np.float64)
    else:
        vm_max = _von_mises_max_over_iso(vm_raw)[0]

    sig_sec = apply_section_stress_operator(bundle, r_case)
    voigt_sec = _aggregate_max_abs_voigt(sig_sec)[0]

    h_row = h_max.reshape(1, -1)
    d_h = apply_span_derivative(bundle, h_row)[0]

    return {
        "section_stress_voigt": voigt_mat,
        "section_strain_maxabs": strain_max,
        "section_hashin_fi_max": h_max,
        "section_hashin_fi_ply_envelope": h_ply_env,
        "section_von_mises_fi_max": vm_max,
        "section_stress_voigt_secframe": voigt_sec,
        "section_d_hashin_fi_dz": d_h,
    }


def build_beam_section_recovery_artifacts(
    res: "BeamSolveResult",
    *,
    station_z: NDArray[np.float64],
    section_results: tuple[object, ...],
    section_definitions: tuple[object, ...],
) -> tuple[RecoveryCache, Any] | None:
    """
    Build :class:`RecoveryCache` and operator bundle once (same as used by
    :func:`enrich_beam_result_with_section_stress`) for reuse e.g. with NPZ save.
    """
    z_sec = np.asarray(station_z, dtype=np.float64).ravel()
    n_s = int(z_sec.shape[0])
    if n_s == 0 or len(section_results) != n_s or len(section_definitions) == 0:
        return None
    sub0 = section_definitions[0].subcomponents
    sr_list = list(section_results)
    z_n = res.z_nodal_out
    R_n = res.nodal_R
    if z_n is not None and R_n is not None and R_n.ndim == 3:
        nodal_R = _interp_nodal_R(z_n, R_n, z_sec)
    else:
        nodal_R = None
    storage = RecoveryCacheBuilder.build(
        section_results=sr_list,
        section0_subcomponents=sub0,
        z_stations=z_sec,
        nodal_R_stack=nodal_R,
    )
    cache = RecoveryCache(**dataclasses.asdict(storage))
    bundle = build_recovery_operator_bundle(
        section_results=sr_list,
        z_stations=z_sec,
        nodal_R=nodal_R,
        section0_subcomponents=sub0,
        include_interlaminar_operator=False,
    )
    return cache, bundle


def _empty_section_extras(z_sec: NDArray[np.float64]) -> Dict[str, Any]:
    z = z_sec if z_sec.size else None
    keys = (
        "section_stress_voigt_gp",
        "section_stress_voigt_nodal",
        "section_strain_maxabs_gp",
        "section_strain_maxabs_nodal",
        "section_hashin_fi_max_gp",
        "section_hashin_fi_max_nodal",
        "section_von_mises_fi_max_gp",
        "section_von_mises_fi_max_nodal",
        "section_stress_voigt_secframe_gp",
        "section_stress_voigt_secframe_nodal",
        "section_d_hashin_fi_dz_gp",
        "section_d_hashin_fi_dz_nodal",
        "section_hashin_fi_ply_envelope_gp",
        "section_hashin_fi_ply_envelope_nodal",
    )
    return {k: None for k in keys} | {"z_section_recovery": z}


def enrich_beam_result_with_section_stress(
    res: "BeamSolveResult",
    *,
    station_z: NDArray[np.float64],
    section_results: tuple[object, ...],
    section_definitions: tuple[object, ...],
    recovery_cache: Optional[RecoveryCache] = None,
    recovery_bundle: Any = None,
) -> "BeamSolveResult":
    z_sec = np.asarray(station_z, dtype=np.float64).ravel()
    n_s = int(z_sec.shape[0])
    if n_s == 0 or len(section_results) != n_s or len(section_definitions) == 0:
        return dataclasses.replace(res, **_empty_section_extras(z_sec))

    if recovery_cache is not None and recovery_bundle is not None:
        cache, bundle = recovery_cache, recovery_bundle
    else:
        built = build_beam_section_recovery_artifacts(
            res, station_z=z_sec, section_results=section_results, section_definitions=section_definitions
        )
        if built is None:
            return dataclasses.replace(res, **_empty_section_extras(z_sec))
        cache, bundle = built

    out: Dict[str, Any] = {
        "z_section_recovery": z_sec,
        "section_stress_voigt_gp": None,
        "section_stress_voigt_nodal": None,
        "section_strain_maxabs_gp": None,
        "section_strain_maxabs_nodal": None,
        "section_hashin_fi_max_gp": None,
        "section_hashin_fi_max_nodal": None,
        "section_von_mises_fi_max_gp": None,
        "section_von_mises_fi_max_nodal": None,
        "section_stress_voigt_secframe_gp": None,
        "section_stress_voigt_secframe_nodal": None,
        "section_d_hashin_fi_dz_gp": None,
        "section_d_hashin_fi_dz_nodal": None,
        "section_hashin_fi_ply_envelope_gp": None,
        "section_hashin_fi_ply_envelope_nodal": None,
    }

    z_gp = res.z_stations_out
    r_gp = res.resultants
    if z_gp is not None and r_gp is not None and r_gp.size > 0:
        r_gp_sec = beam_resultants_to_section_recovery_order(_interp_columns(z_gp, r_gp, z_sec))
        r_case = r_gp_sec.reshape(1, n_s, 7)
        d = _recovery_path_arrays(cache, bundle, r_case)
        out["section_stress_voigt_gp"] = d["section_stress_voigt"]
        out["section_strain_maxabs_gp"] = d["section_strain_maxabs"]
        out["section_hashin_fi_max_gp"] = d["section_hashin_fi_max"]
        out["section_von_mises_fi_max_gp"] = d["section_von_mises_fi_max"]
        out["section_stress_voigt_secframe_gp"] = d["section_stress_voigt_secframe"]
        out["section_d_hashin_fi_dz_gp"] = d["section_d_hashin_fi_dz"]
        out["section_hashin_fi_ply_envelope_gp"] = d["section_hashin_fi_ply_envelope"]

    z_nd = res.z_nodal_out
    r_nd = res.resultants_nodal
    if z_nd is not None and r_nd is not None and r_nd.size > 0:
        r_nd_sec = beam_resultants_to_section_recovery_order(_interp_columns(z_nd, r_nd, z_sec))
        r_case_n = r_nd_sec.reshape(1, n_s, 7)
        d = _recovery_path_arrays(cache, bundle, r_case_n)
        out["section_stress_voigt_nodal"] = d["section_stress_voigt"]
        out["section_strain_maxabs_nodal"] = d["section_strain_maxabs"]
        out["section_hashin_fi_max_nodal"] = d["section_hashin_fi_max"]
        out["section_von_mises_fi_max_nodal"] = d["section_von_mises_fi_max"]
        out["section_stress_voigt_secframe_nodal"] = d["section_stress_voigt_secframe"]
        out["section_d_hashin_fi_dz_nodal"] = d["section_d_hashin_fi_dz"]
        out["section_hashin_fi_ply_envelope_nodal"] = d["section_hashin_fi_ply_envelope"]

    return dataclasses.replace(res, **out)


def build_section_recovery_cache_for_save(
    *,
    station_z: NDArray[np.float64],
    section_results: tuple[object, ...],
    section_definitions: tuple[object, ...],
    nodal_R: NDArray[np.float64] | None,
) -> RecoveryCache:
    """Build fused recovery cache for optional NPZ persistence (same recipe as enrich)."""
    z_sec = np.asarray(station_z, dtype=np.float64).ravel()
    n_s = int(z_sec.shape[0])
    if n_s == 0 or len(section_results) != n_s or len(section_definitions) == 0:
        raise ValueError("Invalid section inputs for recovery cache.")
    sub0 = section_definitions[0].subcomponents
    storage = RecoveryCacheBuilder.build(
        section_results=list(section_results),
        section0_subcomponents=sub0,
        z_stations=z_sec,
        nodal_R_stack=nodal_R,
    )
    return RecoveryCache(**dataclasses.asdict(storage))


def save_section_recovery_cache_to_npz(
    res: "BeamSolveResult",
    *,
    station_z: NDArray[np.float64],
    section_results: tuple[object, ...],
    section_definitions: tuple[object, ...],
    path: Path,
    cache: RecoveryCache | None = None,
) -> None:
    """Write fused :class:`RecoveryCache` operators for fatigue / reuse."""
    from blade_utilities.recovery import save_cache

    z_sec = np.asarray(station_z, dtype=np.float64).ravel()
    if cache is None:
        z_n = res.z_nodal_out
        R_n = res.nodal_R
        if z_n is not None and R_n is not None and R_n.ndim == 3:
            nodal_R = _interp_nodal_R(z_n, R_n, z_sec)
        else:
            nodal_R = None
        cache = build_section_recovery_cache_for_save(
            station_z=z_sec,
            section_results=section_results,
            section_definitions=section_definitions,
            nodal_R=nodal_R,
        )
    outp = Path(path).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_cache(cache, str(outp))
