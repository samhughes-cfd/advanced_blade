"""Backward-compatible failure-helper re-exports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.failure_criteria import (
    hashin_fi,
    hashin_fi_plies,
    von_mises_plane_stress_fi,
)


def tsai_wu_strength_tensors(
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Legacy Tsai-Wu coefficient helper retained for old import paths."""
    Xt = np.maximum(Xt, 1e-12)
    Xc = np.maximum(Xc, 1e-12)
    Yt = np.maximum(Yt, 1e-12)
    Yc = np.maximum(Yc, 1e-12)
    S12 = np.maximum(S12, 1e-12)
    F1lin = 1.0 / Xt - 1.0 / Xc
    F2lin = 1.0 / Yt - 1.0 / Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S12 * S12)
    F12 = -0.5 * np.sqrt(np.maximum(F11 * F22, 0.0))
    F1 = np.stack([F1lin, F2lin, np.zeros_like(F1lin)], axis=-1)
    F2 = np.zeros(F1.shape + (3,), dtype=np.float64)
    F2[..., 0, 0] = F11
    F2[..., 1, 1] = F22
    F2[..., 2, 2] = F66
    F2[..., 0, 1] = F12
    F2[..., 1, 0] = F12
    return F1, F2


def tsai_wu_fi(
    sigma_mat: NDArray[np.float64],
    F1: NDArray[np.float64],
    F2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Legacy vectorised Tsai-Wu FI retained for old import paths."""
    F1b = np.broadcast_to(F1, sigma_mat.shape)
    F2b = np.broadcast_to(F2, sigma_mat.shape + (3,))
    fi_lin = np.einsum("...ki,...ki->...k", sigma_mat, F1b, optimize=True)
    fi_quad = np.einsum("...ki,...kj,...kij->...k", sigma_mat, sigma_mat, F2b, optimize=True)
    return fi_lin + fi_quad


def tsai_wu_fi_plies(
    sigma_mat: NDArray[np.float64],
    Xt: NDArray[np.float64],
    Xc: NDArray[np.float64],
    Yt: NDArray[np.float64],
    Yc: NDArray[np.float64],
    S12: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Legacy ply-wise Tsai-Wu FI retained for old import paths."""
    F1, F2 = tsai_wu_strength_tensors(Xt, Xc, Yt, Yc, S12)
    return tsai_wu_fi(sigma_mat, F1, F2)


__all__ = [
    "hashin_fi",
    "hashin_fi_plies",
    "tsai_wu_strength_tensors",
    "tsai_wu_fi",
    "tsai_wu_fi_plies",
    "von_mises_plane_stress_fi",
]
