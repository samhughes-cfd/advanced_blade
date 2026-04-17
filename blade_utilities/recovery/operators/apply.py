"""Vectorized application helpers for recovery-operator bundles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_utilities.recovery.operators.types import RecoveryOperatorBundleProtocol


def apply_strain_operator(
    bundle: RecoveryOperatorBundleProtocol,
    beam_resultants: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(n_case, n_s, n_comp, 6)``."""
    return np.einsum("spaj,csj->cspa", bundle.H_eps, beam_resultants, optimize=True)


def apply_section_stress_operator(
    bundle: RecoveryOperatorBundleProtocol,
    beam_resultants: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(n_case, n_s, n_comp, n_ply_max, 3)`` in section frame."""
    return np.einsum("spkaj,csj->cspka", bundle.L_sec, beam_resultants, optimize=True)


def apply_span_derivative(
    bundle: RecoveryOperatorBundleProtocol,
    field: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply ``D_z`` along station axis.

    - For ``field.ndim == 1``, station axis is 0.
    - For ``field.ndim >= 2``, station axis is 1 (case-major arrays).
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 1:
        return bundle.D_z @ arr
    moved = np.moveaxis(arr, 1, 0)
    out = np.tensordot(bundle.D_z, moved, axes=([1], [0]))
    return np.moveaxis(out, 0, 1)


def apply_interlaminar_transfer(
    bundle: RecoveryOperatorBundleProtocol,
    sigma_section: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(n_case, n_s, n_comp, n_interface, 2)`` interface quantities."""
    if bundle.G_if is None:
        raise ValueError("G_if is not available. Rebuild bundle with include_interlaminar_operator=True.")
    return np.einsum("spiakb,cspkb->cspia", bundle.G_if, sigma_section, optimize=True)
