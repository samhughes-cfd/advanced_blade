"""
Fatigue pipeline: stress recovery → rainflow → Miner damage.

No FE / CLPT / Tsai–Wu in the fatigue hot path; static FI uses cached Tsai–Wu / VM only.
Rainflow is never applied to beam resultants — only to scalar stress histories.
"""

from __future__ import annotations

import logging
import time
import numpy as np
from numpy.typing import NDArray

from blade_utilities.stress_recovery.engine.cache import RecoveryCache

from .conversion import (
    beam_resultants_to_cache_order,
    resultants_to_stress_history,
    resultants_to_stress_history_lazy,
    stress_history_memory_mb,
)
from .damage import life_from_damage, miner_damage
from ..core.loads import ResultantHistory, StressHistory
from ..util.logging_utils import log_json
from .rainflow import IncrementalRainflowAccumulator, count_cycles_iso_vm, count_cycles_ply_stresses, merge_rainflow_bins
from .sn_curves import SNcurve
from .stress_range import von_mises_plane_stress
from ..core.types import FatigueResult, RainflowBins

logger = logging.getLogger(__name__)


def _shared_range_edges(stress: StressHistory, stress_component: int, n_range_bins: int) -> NDArray[np.float64]:
    """Peak-to-peak over all composite σ_driver and isotropic σ_VM series → linear edges ``[0, S_max]``."""
    sig = np.asarray(stress.sigma_composite[..., stress_component], dtype=np.float64)
    s11 = np.asarray(stress.sigma_isotropic[..., 0], dtype=np.float64)
    s22 = np.asarray(stress.sigma_isotropic[..., 1], dtype=np.float64)
    t12 = np.asarray(stress.sigma_isotropic[..., 2], dtype=np.float64)
    vm = von_mises_plane_stress(s11, s22, t12)
    ptps: list[float] = []
    for arr in (sig, vm):
        for idx in np.ndindex(arr.shape[1:]):
            col = arr[(slice(None),) + idx]
            if col.size > 1:
                ptps.append(float(np.ptp(col)))
    s_max = max(ptps) if ptps else 1.0
    s_max = max(s_max, 1e-12)
    return np.linspace(0.0, s_max, int(n_range_bins) + 1, dtype=np.float64)


def _resolve_sn_curve(name: str, sn_curves: dict[str, SNcurve]) -> SNcurve:
    if name in sn_curves:
        return sn_curves[name]
    lower = {k.lower(): v for k, v in sn_curves.items()}
    if name.lower() in lower:
        return lower[name.lower()]
    ln = name.lower()
    if "carbon" in ln or "cfrp" in ln:
        return sn_curves.get("CFRP") or sn_curves.get("cfrp") or next(iter(sn_curves.values()))
    if "glass" in ln or "gfrp" in ln or "grp" in ln:
        return sn_curves.get("GFRP") or sn_curves.get("gfrp") or next(iter(sn_curves.values()))
    if "alum" in ln or "alumin" in ln:
        return sn_curves.get("aluminium") or sn_curves.get("aluminum") or next(iter(sn_curves.values()))
    if "steel" in ln or "metal" in ln:
        c_st = sn_curves.get("steel")
        return c_st if c_st is not None else SNcurve.steel_dnv()
    if "default" in sn_curves:
        return sn_curves["default"]
    return next(iter(sn_curves.values()))


def _damage_composite_stacked(
    bins: RainflowBins,
    cache: RecoveryCache,
    sn_curves: dict[str, SNcurve],
    apply_goodman: bool,
) -> NDArray[np.float64]:
    n_s, n_cp, n_ply = bins.counts_comp.shape[1], bins.counts_comp.shape[2], bins.counts_comp.shape[3]
    out = np.zeros((n_s, n_cp, n_ply), dtype=np.float64)
    names = cache.composite_subcomp_names
    for p in range(n_cp):
        sn = _resolve_sn_curve(names[p], sn_curves)
        out[:, p, :] = miner_damage(
            bins.ranges_comp[:, :, p, :],
            bins.means_comp[:, :, p, :],
            bins.counts_comp[:, :, p, :],
            sn,
            apply_goodman=apply_goodman,
        )
    return out


def _damage_isotropic_stacked(
    bins: RainflowBins,
    cache: RecoveryCache,
    sn_curves: dict[str, SNcurve],
    apply_goodman: bool,
) -> NDArray[np.float64]:
    n_s, n_ip = bins.counts_iso.shape[1], bins.counts_iso.shape[2]
    out = np.zeros((n_s, n_ip), dtype=np.float64)
    names = cache.isotropic_subcomp_names
    for p in range(n_ip):
        sn = _resolve_sn_curve(names[p], sn_curves)
        out[:, p] = miner_damage(
            bins.ranges_iso[:, :, p],
            bins.means_iso[:, :, p],
            bins.counts_iso[:, :, p],
            sn,
            apply_goodman=apply_goodman,
        )
    return out


def _life_masked(
    damage: NDArray[np.float64],
    design_life_years: float,
    active_mask: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    life = life_from_damage(damage, design_life_years)
    if active_mask is not None:
        life = np.where(active_mask, life, np.inf)
    return life


def _worst_composite(
    damage: NDArray[np.float64],
    ply_count: NDArray[np.int32],
    names: list[str],
) -> tuple[float, tuple[int, str, int]]:
    n_s, n_cp, n_ply = damage.shape
    masked = np.asarray(damage, dtype=np.float64).copy()
    for s in range(n_s):
        for p in range(n_cp):
            kmax = int(ply_count[s, p])
            masked[s, p, kmax:] = -np.inf
    flat = int(np.argmax(masked))
    idx = np.unravel_index(flat, damage.shape)
    return float(damage[idx]), (int(idx[0]), str(names[int(idx[1])]), int(idx[2]))


def _worst_isotropic(damage: NDArray[np.float64], names: list[str]) -> tuple[float, tuple[int, str]]:
    flat = int(np.argmax(damage))
    idx = np.unravel_index(flat, damage.shape)
    return float(damage[idx]), (int(idx[0]), str(names[int(idx[1])]))


class FatiguePipeline:
    def __init__(
        self,
        cache: RecoveryCache,
        sn_curves: dict[str, SNcurve],
        chunk_size: int = 256,
        stress_component: int = 0,
        n_range_bins: int = 128,
        apply_goodman: bool = False,
        enable_tier3_delam: bool = False,
        design_life_years: float = 25.0,
    ) -> None:
        self.cache = cache
        self.sn_curves = sn_curves
        self.chunk_size = int(chunk_size)
        self.stress_component = int(stress_component)
        self.n_range_bins = int(n_range_bins)
        self.apply_goodman = bool(apply_goodman)
        self.enable_tier3_delam = bool(enable_tier3_delam)
        self.design_life_years = float(design_life_years)

    def run(self, history: ResultantHistory, memory_limit_mb: float = 512.0) -> FatigueResult:
        # Fatigue hot path: algebraic L_rec/L_iso only; rainflow on stresses; chunk time only.
        # No FE, CLPT, or Tsai–Wu coefficient work here (static FI below uses cache.eval_* only).
        estimated_mb = stress_history_memory_mb(history, self.cache)
        n_t = int(history.time.shape[0])
        memory_mode = "incremental" if estimated_mb > float(memory_limit_mb) else "full"

        log_json(
            logger,
            logging.INFO,
            "conversion_start",
            {
                "n_t": n_t,
                "n_s": int(self.cache.L_rec.shape[0]),
                "estimated_memory_mb": float(estimated_mb),
                "memory_mode": memory_mode,
            },
        )

        if memory_mode == "incremental":
            t_conv0 = time.perf_counter()
            acc = IncrementalRainflowAccumulator(
                self.cache,
                n_t=n_t,
                n_range_bins=self.n_range_bins,
                stress_component=self.stress_component,
            )
            chunk_i = 0
            for chunk in resultants_to_stress_history_lazy(history, self.cache, chunk_size=self.chunk_size):
                t0 = time.perf_counter()
                acc.update(chunk)
                log_json(
                    logger,
                    logging.INFO,
                    "conversion_chunk",
                    {
                        "chunk": chunk_i,
                        "elapsed_s": float(time.perf_counter() - t0),
                        "sigma11_max_chunk": float(np.max(chunk.sigma_composite[..., 0])),
                    },
                )
                chunk_i += 1
            log_json(
                logger,
                logging.INFO,
                "conversion_complete",
                {"elapsed_s_total": float(time.perf_counter() - t_conv0)},
            )
            bins = acc.finalise()
        else:
            t_conv0 = time.perf_counter()
            stress = resultants_to_stress_history(history, self.cache, chunk_size=self.chunk_size)
            log_json(
                logger,
                logging.INFO,
                "conversion_complete",
                {"elapsed_s_total": float(time.perf_counter() - t_conv0)},
            )
            edges = _shared_range_edges(stress, self.stress_component, self.n_range_bins)
            bins_c = count_cycles_ply_stresses(
                stress,
                component=self.stress_component,
                n_range_bins=self.n_range_bins,
                range_edges=edges,
            )
            bins_i = count_cycles_iso_vm(stress, n_range_bins=self.n_range_bins, range_edges=edges)
            bins = merge_rainflow_bins(bins_c, bins_i)

        log_json(
            logger,
            logging.INFO,
            "rainflow_complete",
            {
                "n_cycles_composite": int(np.sum(bins.counts_comp)),
                "n_cycles_isotropic": int(np.sum(bins.counts_iso)),
            },
        )

        damage_c = _damage_composite_stacked(bins, self.cache, self.sn_curves, self.apply_goodman)
        damage_i = _damage_isotropic_stacked(bins, self.cache, self.sn_curves, self.apply_goodman)

        n_s, n_cp, n_ply = damage_c.shape
        pc = np.asarray(self.cache.ply_count, dtype=np.int32)
        if pc.shape[0] != n_s or pc.shape[1] != n_cp:
            pc = np.broadcast_to(pc, (n_s, n_cp)).copy()
        active = np.zeros((n_s, n_cp, n_ply), dtype=bool)
        for s in range(n_s):
            for p in range(n_cp):
                active[s, p, : int(pc[s, p])] = True

        life_c = _life_masked(damage_c, self.design_life_years, active)
        life_i = life_from_damage(damage_i, self.design_life_years)

        max_dc, worst_c = _worst_composite(damage_c, pc, self.cache.composite_subcomp_names)
        max_di, worst_i = _worst_isotropic(damage_i, self.cache.isotropic_subcomp_names)
        critical = "composite" if max_dc >= max_di else "isotropic"

        R = beam_resultants_to_cache_order(history.to_array())
        norms = np.linalg.norm(R, axis=-1)
        t_star = np.argmax(norms, axis=0)
        n_s2 = R.shape[1]
        R_peak = np.stack([R[int(t_star[s]), s, :] for s in range(n_s2)], axis=0)[None, :, :]
        fi_tw = self.cache.eval_tsai_wu_fi(R_peak)[0]
        fi_vm = self.cache.eval_von_mises_fi(R_peak)[0]

        damage_delam: NDArray[np.float64] | None = None
        if self.enable_tier3_delam:
            reason = "cache_disabled_tier3" if not self.cache.enable_tier3 else "not_implemented"
            log_json(
                logger,
                logging.INFO,
                "tier3_delam_skipped",
                {"reason": reason, "cache_enable_tier3": bool(self.cache.enable_tier3)},
            )

        result = FatigueResult(
            damage_composite=damage_c,
            damage_isotropic=damage_i,
            damage_delam=damage_delam,
            life_composite=life_c,
            life_isotropic=life_i,
            max_damage_composite=max_dc,
            max_damage_isotropic=max_di,
            worst_composite=worst_c,
            worst_isotropic=worst_i,
            fatigue_critical_material=critical,
            fi_static_tw=fi_tw,
            fi_static_vm=fi_vm,
            stress_component_used=self.stress_component,
            goodman_applied=self.apply_goodman,
            design_life_years=self.design_life_years,
            memory_mode=memory_mode,
            rainflow_bins=bins,
        )

        life_c_act = life_c[active]
        min_c = float(np.min(life_c_act)) if life_c_act.size else float("inf")
        min_life = min(min_c, float(np.min(life_i)))
        log_json(
            logger,
            logging.INFO,
            "damage_complete",
            {
                "max_damage_composite": float(result.max_damage_composite),
                "max_damage_isotropic": float(result.max_damage_isotropic),
                "fatigue_critical": result.fatigue_critical_material,
                "min_life_years": min_life,
            },
        )
        return result
