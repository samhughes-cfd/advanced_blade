"""Runtime :class:`RecoveryCache` with vectorised recovery and failure indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.failure_criteria import tsai_wu_fi, von_mises_plane_stress_fi
from blade_precompute.section_properties.engine.interlaminar_recovery import delamination_fi, interlaminar_stress_recovery

from blade_utilities.recovery.core.cache_types import RecoveryCacheStorage

CACHE_VERSION = 1
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

    def eval_tsai_wu_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_case, n_s, n_comp_sub, n_ply_max)`` Tsai–Wu failure index per ply."""
        sig = self.recover_ply_stresses(beam_resultants)
        return tsai_wu_fi(sig, self.F1, self.F2)

    def eval_von_mises_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n_case, n_s, n_iso_sub)`` von Mises FI vs ``sigma_allow_iso``."""
        sig = self.recover_iso_stresses(beam_resultants)
        return von_mises_plane_stress_fi(sig, self.sigma_allow_iso)

    def eval_delamination_fi(self, beam_resultants: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """
        ``(n_case, n_s, n_comp_sub, n_interface)`` or ``None`` if Tier-3 was not built.

        Uses section-frame ply stresses (``L_rec_sec``) and spanwise gradients
        consistent with :func:`blade_precompute.section_properties.engine.interlaminar_recovery.interlaminar_stress_recovery`.
        """
        if not self.enable_tier3 or self.L_rec_sec is None:
            return None
        n_case = int(beam_resultants.shape[0])
        n_s = int(self.z_stations.shape[0])
        n_comp = int(self.L_rec_sec.shape[1])
        n_if = int(self.L_rec_sec.shape[2]) + 1
        out = np.zeros((n_case, n_s, n_comp, n_if), dtype=np.float64)
        if n_s < 2:
            return out
        z_ply = self.z_ply_ref
        for c in range(n_case):
            r = beam_resultants[c : c + 1]
            sig_sec = np.einsum("spkqj,csj->cspkq", self.L_rec_sec, r, optimize=True)[0]
            tau_if = interlaminar_stress_recovery(sig_sec, self.z_stations, z_ply)
            for s in range(n_s):
                out[c, s] = delamination_fi(
                    tau_if[s : s + 1],
                    self.Zt[s],
                    self.S13[s],
                    self.S23[s],
                )[0]
        return out

    def recover_all_fi(
        self, beam_resultants: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        return (
            self.eval_tsai_wu_fi(beam_resultants),
            self.eval_von_mises_fi(beam_resultants),
            self.eval_delamination_fi(beam_resultants),
        )

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
            "F1": self.F1,
            "F2": self.F2,
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
            "enable_tier3": np.array([self.enable_tier3], dtype=np.bool_),
            "composite_subcomp_idx": np.array(self.composite_subcomp_idx, dtype=np.int32),
            "isotropic_subcomp_idx": np.array(self.isotropic_subcomp_idx, dtype=np.int32),
        }
        if self.L_rec_sec is not None:
            d["L_rec_sec"] = self.L_rec_sec
        d["composite_subcomp_names"] = np.array(self.composite_subcomp_names, dtype=object)
        d["isotropic_subcomp_names"] = np.array(self.isotropic_subcomp_names, dtype=object)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RecoveryCache":
        def _arr(key: str) -> NDArray[np.float64]:
            return np.asarray(d[key], dtype=np.float64)

        def _idx_arr(key: str) -> List[int]:
            return [int(x) for x in np.asarray(d[key]).ravel()]

        def _names(key: str) -> List[str]:
            return [str(x) for x in np.asarray(d[key], dtype=object).ravel()]

        l_rec_sec_opt: Optional[NDArray[np.float64]]
        raw_sec = d.get("L_rec_sec")
        if raw_sec is None:
            l_rec_sec_opt = None
        else:
            l_rec_sec_opt = np.asarray(raw_sec, dtype=np.float64)

        en = d["enable_tier3"]
        if isinstance(en, np.ndarray):
            enable_t3 = bool(en.ravel()[0])
        else:
            enable_t3 = bool(en)

        return RecoveryCache(
            L_rec=_arr("L_rec"),
            L_iso=_arr("L_iso"),
            L_rec_sec=l_rec_sec_opt,
            F1=_arr("F1"),
            F2=_arr("F2"),
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
            enable_tier3=enable_t3,
        )


# Back-compat for modules that imported private _NPZ_VERSION_KEY
_NPZ_VERSION_KEY = NPZ_VERSION_KEY
