"""
Axis conventions for :class:`RecoveryCache` (all stress-related arrays):

- **c**: load case index
- **s**: spanwise station index
- **p**: subcomponent row within the composite or isotropic routing table (not a global mesh index)
- **k**: ply index (padded to ``n_ply_max`` for composites)
- **j**: beam resultant mode ``[N, My, Mz, T, Vy, Vz, B]`` (seven-vector order used in ``beam_model`` / ``design_optimisation``)
- **q** (when used): normal stress / shear Voigt component ``[σ11, σ22, τ12]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class RecoveryCacheStorage:
    """Serializable fields for :class:`~recovery_cache.engine.cache.RecoveryCache`."""

    L_rec: NDArray[np.float64]
    L_iso: NDArray[np.float64]
    L_rec_sec: Optional[NDArray[np.float64]]
    F1: NDArray[np.float64]
    F2: NDArray[np.float64]
    sigma_allow_iso: NDArray[np.float64]
    Zt: NDArray[np.float64]
    S13: NDArray[np.float64]
    S23: NDArray[np.float64]
    spanwise_dz: NDArray[np.float64]
    z_stations: NDArray[np.float64]
    z_ply_ref: NDArray[np.float64]
    composite_subcomp_idx: List[int]
    isotropic_subcomp_idx: List[int]
    composite_subcomp_names: List[str]
    isotropic_subcomp_names: List[str]
    ply_count: NDArray[np.int32]
    K7: NDArray[np.float64]
    K6: NDArray[np.float64]
    M6: NDArray[np.float64]
    shear_center: NDArray[np.float64]
    mass_center: NDArray[np.float64]
    enable_tier3: bool


class RecoveryCacheProtocol(Protocol):
    """Structural typing hook for callers that only need recovery / FI APIs."""

    def recover_ply_stresses(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def recover_iso_stresses(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def eval_tsai_wu_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def eval_von_mises_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def eval_delamination_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64] | None: ...

    def recover_all_fi(
        self, beam_resultants: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]: ...

    def recover_ply_stresses_chunked(
        self, beam_resultants: NDArray[np.float64], chunk_size: int = 512
    ): ...


RecoveryEvaluatorProtocol = RecoveryCacheProtocol
