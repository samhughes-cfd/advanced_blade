"""Runtime :class:`RecoveryCache` with vectorised recovery and failure indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.failure_criteria import hashin_fi_plies, von_mises_plane_stress_fi

from blade_utilities.recovery.core.cache_types import RecoveryCacheStorage

CACHE_VERSION = 2
NPZ_VERSION_KEY = "recovery_cache_version"


@dataclass
class RecoveryCache(RecoveryCacheStorage):
    """
    Precomputed fused operators and allowables.

    Beam resultants ``beam_resultants`` have shape ``(n_case, n_s, 7)`` with modes
    ``j``: ``[N, My, Mz, T, Vy, Vz, B]``.
    """

    def recover_ply_stresses(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_case, n_s, n_comp_sub, n_ply_max, 3)`` material-frame ply stresses."""
        return np.einsum("spkqj,csj->cspkq", self.L_rec, beam_resultants, optimize=True)

    def recover_iso_stresses(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_case, n_s, n_iso_sub, 3)`` membrane stresses in the rotated shell basis."""
        return np.einsum("spqj,csj->cspq", self.L_iso, beam_resultants, optimize=True)

    def eval_hashin_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_case, n_s, n_comp_sub, n_ply_max)`` Hashin envelope failure index per ply."""
        sig = self.recover_ply_stresses(beam_resultants)
        return hashin_fi_plies(sig, self.Xt, self.Xc, self.Yt, self.Yc, self.S12)

    def eval_von_mises_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_case, n_s, n_iso_sub)`` von Mises FI vs ``sigma_allow_iso``."""
        sig = self.recover_iso_stresses(beam_resultants)
        return von_mises_plane_stress_fi(sig, self.sigma_allow_iso)

    def recover_all_fi(
        self, beam_resultants: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (self.eval_hashin_fi(beam_resultants), self.eval_von_mises_fi(beam_resultants))

    def recover_ply_stresses_chunked(
        self, beam_resultants: NDArray[np.float64], chunk_size: int = 512
    ) -> Iterator[NDArray[np.float64]]:
        """Yield ply stress chunks ``(chunk, n_s, n_comp_sub, n_ply_max, 3)`` along case axis."""
        n_case = int(beam_resultants.shape[0])
        for start in range(0, n_case, chunk_size):
            end = min(start + chunk_size, n_case)
            yield self.recover_ply_stresses(beam_resultants[start:end])

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            NPZ_VERSION_KEY: np.int32(CACHE_VERSION),
            "L_rec": self.L_rec,
            "L_iso": self.L_iso,
            "Xt": self.Xt,
            "Xc": self.Xc,
            "Yt": self.Yt,
            "Yc": self.Yc,
            "S12": self.S12,
            "sigma_allow_iso": self.sigma_allow_iso,
            "Zt": self.Zt,
            "S13": self.S13,
            "S23": self.S23,
            "spanwise_dz": self.spanwise_dz,
            "z_stations": self.z_stations,
            "z_ply_ref": self.z_ply_ref,
            "ply_count": self.ply_count,
            "K7": self.K7,
            "K6": self.K6,
            "M6": self.M6,
            "shear_center": self.shear_center,
            "mass_center": self.mass_center,
            "composite_subcomp_idx": np.array(self.composite_subcomp_idx, dtype=np.int32),
            "isotropic_subcomp_idx": np.array(self.isotropic_subcomp_idx, dtype=np.int32),
        }
        d["composite_subcomp_names"] = np.array(self.composite_subcomp_names, dtype=object)
        d["isotropic_subcomp_names"] = np.array(self.isotropic_subcomp_names, dtype=object)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RecoveryCache":
        def _arr(key: str) -> NDArray[np.float64]:
            return np.asarray(d[key], dtype=np.float64)

        raw_v = d.get(NPZ_VERSION_KEY, np.int32(0))
        ver = int(np.asarray(raw_v, dtype=np.int32).ravel()[0])
        if ver < CACHE_VERSION:
            raise ValueError(
                f"Unsupported recovery cache version {ver} (minimum {CACHE_VERSION}). "
                "v1 caches used Tsai–Wu F1/F2; rebuild the NPZ with the current code (Hashin strengths)."
            )
        if "F1" in d and "Xt" not in d:
            raise ValueError(
                "Recovery cache appears to be v1 (F1/F2 keys only). Rebuild the NPZ; "
                "Hashin ply strengths (Xt, Xc, Yt, Yc, S12) are required."
            )

        def _idx_arr(key: str) -> List[int]:
            return [int(x) for x in np.asarray(d[key]).ravel()]

        def _names(key: str) -> List[str]:
            return [str(x) for x in np.asarray(d[key], dtype=object).ravel()]

        return RecoveryCache(
            L_rec=_arr("L_rec"),
            L_iso=_arr("L_iso"),
            Xt=_arr("Xt"),
            Xc=_arr("Xc"),
            Yt=_arr("Yt"),
            Yc=_arr("Yc"),
            S12=_arr("S12"),
            sigma_allow_iso=_arr("sigma_allow_iso"),
            Zt=_arr("Zt"),
            S13=_arr("S13"),
            S23=_arr("S23"),
            spanwise_dz=_arr("spanwise_dz"),
            z_stations=_arr("z_stations"),
            z_ply_ref=_arr("z_ply_ref"),
            composite_subcomp_idx=_idx_arr("composite_subcomp_idx"),
            isotropic_subcomp_idx=_idx_arr("isotropic_subcomp_idx"),
            composite_subcomp_names=_names("composite_subcomp_names"),
            isotropic_subcomp_names=_names("isotropic_subcomp_names"),
            ply_count=np.asarray(d["ply_count"], dtype=np.int32),
            K7=_arr("K7"),
            K6=_arr("K6"),
            M6=_arr("M6"),
            shear_center=_arr("shear_center"),
            mass_center=_arr("mass_center"),
        )


# Back-compat for modules that imported private _NPZ_VERSION_KEY
_NPZ_VERSION_KEY = NPZ_VERSION_KEY
