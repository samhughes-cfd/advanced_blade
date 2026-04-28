"""
Exact linear conversion: beam resultant history → ply / isotropic stress history.

No FE solves — algebraic application of precomputed ``L_rec`` / ``L_iso`` from
:class:`recovery_cache.engine.cache.RecoveryCache`. Chunk over time steps only.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from blade_utilities.recovery import RecoveryCache

from ..core.loads import ResultantHistory, StressHistory

# Beam order [N,Vy,Vz,My,Mz,T,B] → cache order [N,My,Mz,T,Vy,Vz,B] for fused operators.
_BEAM_TO_CACHE_IDX = np.array([0, 3, 4, 5, 1, 2, 6], dtype=np.int64)


def beam_resultants_to_cache_order(R_beam: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(..., 7)`` beam stacking → ``(..., 7)`` cache mode index ``j``."""
    return np.take(R_beam, _BEAM_TO_CACHE_IDX, axis=-1)


def stress_history_memory_mb(history: ResultantHistory, cache: RecoveryCache) -> float:
    """Estimated full ``StressHistory`` footprint [MB] if materialised (float64)."""
    n_t = int(history.time.shape[0])
    n_s = int(cache.L_rec.shape[0])
    n_cp = int(cache.L_rec.shape[1])
    n_ply = int(cache.L_rec.shape[2])
    n_ip = int(cache.L_iso.shape[1])
    comp = n_t * n_s * n_cp * n_ply * 3 * 8
    iso = n_t * n_s * n_ip * 3 * 8
    return float(comp + iso) / (1024.0 * 1024.0)


def resultants_to_stress_history(
    history: ResultantHistory,
    cache: RecoveryCache,
    chunk_size: int = 256,
) -> StressHistory:
    """
    Convert beam resultant time history to ply-level stress time history using
    ``L_rec`` and ``L_iso``. Exact linear map; chunk over time only.

    ``R`` uses beam ``to_array()`` order; permuted to cache mode before einsum
    to match :meth:`RecoveryCache.recover_ply_stresses`.
    """
    R = beam_resultants_to_cache_order(history.to_array())
    n_t, n_s, _ = R.shape
    n_cp = cache.L_rec.shape[1]
    n_ply = cache.L_rec.shape[2]
    n_ip = cache.L_iso.shape[1]

    sigma_comp = np.empty((n_t, n_s, n_cp, n_ply, 3), dtype=np.float64)
    sigma_iso = np.empty((n_t, n_s, n_ip, 3), dtype=np.float64)

    # No FE / CLPT here — only fused L_rec / L_iso (precomputed).
    for i in range(0, n_t, chunk_size):
        sl = slice(i, i + chunk_size)
        chunk = R[sl]
        sigma_comp[sl] = np.einsum("spkqj,tsj->tspkq", cache.L_rec, chunk, optimize=True)
        sigma_iso[sl] = np.einsum("spqj,tsj->tspq", cache.L_iso, chunk, optimize=True)

    return StressHistory(
        z_stations=history.z_stations,
        time=history.time,
        sigma_composite=sigma_comp,
        sigma_isotropic=sigma_iso,
        composite_subcomp_names=list(cache.composite_subcomp_names),
        isotropic_subcomp_names=list(cache.isotropic_subcomp_names),
    )


def resultants_to_stress_history_lazy(
    history: ResultantHistory,
    cache: RecoveryCache,
    chunk_size: int = 256,
) -> Iterator[StressHistory]:
    """
    Yield time chunks of ``StressHistory`` without materialising the full tensors.

    Each chunk has ``time`` and stress arrays sized to the chunk along axis 0.
    ``z_stations`` and name lists match the full blade.
    """
    R = beam_resultants_to_cache_order(history.to_array())
    n_t, n_s, _ = R.shape
    n_cp = cache.L_rec.shape[1]
    n_ply = cache.L_rec.shape[2]
    n_ip = cache.L_iso.shape[1]
    time = history.time

    for i in range(0, n_t, chunk_size):
        sl = slice(i, i + chunk_size)
        chunk = R[sl]
        nt_c = chunk.shape[0]
        sigma_comp = np.einsum("spkqj,tsj->tspkq", cache.L_rec, chunk, optimize=True)
        sigma_iso = np.einsum("spqj,tsj->tspq", cache.L_iso, chunk, optimize=True)
        yield StressHistory(
            z_stations=history.z_stations,
            time=np.asarray(time[sl], dtype=np.float64),
            sigma_composite=sigma_comp,
            sigma_isotropic=sigma_iso,
            composite_subcomp_names=list(cache.composite_subcomp_names),
            isotropic_subcomp_names=list(cache.isotropic_subcomp_names),
        )
