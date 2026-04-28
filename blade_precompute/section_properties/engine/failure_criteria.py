"""Vectorised failure indices (Hashin envelope, von Mises) for batch stress evaluation."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

_UNITS_WARNED = False


def _active_ply_mask(*strength_arrays: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True for plies where at least one strength value is non-trivially positive.

    Padded slots (filled with zeros) are excluded so they do not trigger
    spurious unit warnings or inflate failure indices via σ/1e-12 division.
    Threshold 1.0 Pa is below any real composite allowable (Xt > 1 MPa).
    """
    combined = np.stack([np.asarray(a, dtype=np.float64) for a in strength_arrays], axis=0)
    return np.any(combined > 1.0, axis=0)


def _warn_if_suspect_units(
    sigma_mat: NDArray[np.float64],
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
) -> None:
    """Warn once if strengths look like MPa while stresses are in Pa.

    Only active (non-padded) plies are checked so zero-strength padding slots
    do not trigger a false positive.
    """
    global _UNITS_WARNED
    if _UNITS_WARNED:
        return
    mask = _active_ply_mask(Xt, Xc, Yt, Yc, S12)
    strength_stack = np.concatenate(
        [
            np.asarray(Xt, dtype=np.float64).ravel()[mask.ravel()],
            np.asarray(Xc, dtype=np.float64).ravel()[mask.ravel()],
            np.asarray(Yt, dtype=np.float64).ravel()[mask.ravel()],
            np.asarray(Yc, dtype=np.float64).ravel()[mask.ravel()],
            np.asarray(S12, dtype=np.float64).ravel()[mask.ravel()],
        ]
    )
    strength_stack = strength_stack[np.isfinite(strength_stack)]
    if strength_stack.size == 0:
        return
    sigma_abs_max = float(np.nanmax(np.abs(np.asarray(sigma_mat, dtype=np.float64))))
    strength_min = float(np.nanmin(strength_stack))
    # Composite allowables are typically O(1e7..1e10) Pa.
    if sigma_abs_max > 1.0e6 and strength_min < 1.0e6:
        warnings.warn(
            "Hashin strengths appear suspiciously small for Pa units "
            f"(min active strength={strength_min:.3e}, max |stress|={sigma_abs_max:.3e}). "
            "Check MPa-vs-Pa consistency in material inputs.",
            RuntimeWarning,
            stacklevel=2,
        )
        _UNITS_WARNED = True


def hashin_fi(
    sigma_mat: NDArray[np.float64],
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Hashin (1980) 2D plane-stress **envelope** failure index (FI = 1 on onset).

    ``sigma_mat`` ``(..., 3)`` = ``[σ11, σ22, τ12]`` [Pa] in material axes.
    Strengths broadcast to ``sigma_mat[..., 0]`` (same convention as legacy Tsai–Wu batch).

    Uses Macaulay splits on σ11/σ22 and includes ``(τ12/S)²`` in all four mode quadratics;
    returns the **maximum** of the four mode FIs (aligned with ``laminate_clpt.hashin_fi``).
    """
    _warn_if_suspect_units(sigma_mat, Xt, Xc, Yt, Yc, S12)
    s11 = sigma_mat[..., 0]
    s22 = sigma_mat[..., 1]
    t = sigma_mat[..., 2]
    Xt_a = np.asarray(Xt, dtype=np.float64)
    Xc_a = np.asarray(Xc, dtype=np.float64)
    Yt_a = np.asarray(Yt, dtype=np.float64)
    Yc_a = np.asarray(Yc, dtype=np.float64)
    S12_a = np.asarray(S12, dtype=np.float64)
    # Padded plies have zero strengths; mask them out so FI = 0 instead of σ/1e-12
    active = _active_ply_mask(Xt_a, Xc_a, Yt_a, Yc_a, S12_a)
    Xt_c = np.maximum(Xt_a, 1e-12)
    Xc_c = np.maximum(Xc_a, 1e-12)
    Yt_c = np.maximum(Yt_a, 1e-12)
    Yc_c = np.maximum(Yc_a, 1e-12)
    S = np.maximum(S12_a, 1e-12)
    tq = (t / S) ** 2
    s11p = np.maximum(s11, 0.0)
    s11n = np.minimum(s11, 0.0)
    s22p = np.maximum(s22, 0.0)
    s22n = np.minimum(s22, 0.0)
    fi_ft = (s11p / Xt_c) ** 2 + tq
    fi_fc = (s11n / Xc_c) ** 2 + tq
    fi_mt = (s22p / Yt_c) ** 2 + tq
    fi_mc = (s22n / Yc_c) ** 2 + tq
    fi = np.maximum(np.maximum(fi_ft, fi_fc), np.maximum(fi_mt, fi_mc))
    # Zero out padded-ply slots so they don't pollute the KS aggregate
    return np.where(active, fi, 0.0)


def hashin_fi_plies(
    sigma_mat: NDArray[np.float64],
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``sigma_mat`` shape ``(..., n_ply, 3)``; strengths ``(..., n_ply)`` or broadcast."""
    return hashin_fi(sigma_mat, Xt, Xc, Yt, Yc, S12)


def von_mises_plane_stress_fi(
    sigma_iso: NDArray[np.float64],
    sigma_allow: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``sigma_iso[...,3]`` = ``[σ11, σ22, τ12]``; ``sigma_allow`` broadcastable on batch."""
    s11 = sigma_iso[..., 0]
    s22 = sigma_iso[..., 1]
    t12 = sigma_iso[..., 2]
    sigma_vm = np.sqrt(np.maximum(0.0, s11**2 - s11 * s22 + s22**2 + 3.0 * t12**2))
    allow = np.maximum(np.asarray(sigma_allow, dtype=np.float64), 1e-12)
    while allow.ndim > sigma_vm.ndim and allow.shape[-1] == 1:
        allow = np.squeeze(allow, axis=-1)
    return sigma_vm / allow
