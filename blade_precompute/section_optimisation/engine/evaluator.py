"""Per-design evaluation: sections → beam → stress → failure indices."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_properties.engine import failure_criteria as fail_mod
from blade_precompute.section_properties.engine.clpt_recovery import clpt_ply_stresses_section_frame, rotate_plies_to_material
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry
from blade_precompute.section_properties.engine.interlaminar_recovery import delamination_fi, interlaminar_stress_recovery
from blade_precompute.section_properties.engine.laminate import LaminateDefinition

from .cache import dirty_indices, init_station_caches
from .mass import mass_objective
from .stiffness_metric import integrated_k7_trace
from .parallel import solve_dirty_stations
from ..core.protocols import BeamResultantDriverProtocol, PrescribedResultantDriver
from .section_builder import SectionBuilder
from ..core.types import DesignEvaluation, DesignProblem, DesignVector


def _ply_strength_fields(lam: LaminateDefinition) -> tuple[np.ndarray, ...]:
    n = len(lam.plies)
    Xt = np.zeros(n, dtype=np.float64)
    Xc = np.zeros(n, dtype=np.float64)
    Yt = np.zeros(n, dtype=np.float64)
    Yc = np.zeros(n, dtype=np.float64)
    S12 = np.zeros(n, dtype=np.float64)
    for i, (ply, _) in enumerate(lam.plies):
        Xt[i] = ply.Xt
        Xc[i] = ply.Xc
        Yt[i] = ply.Yt
        Yc[i] = ply.Yc
        S12[i] = ply.S12
    return Xt, Xc, Yt, Yc, S12


def _composite_strength_stack(
    section0_subs: list[SubcomponentGeometry],
    n_ply_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for sub in section0_subs:
        if not sub.is_composite:
            continue
        lam = sub.material
        assert isinstance(lam, LaminateDefinition)
        tXt, tXc, tYt, tYc, tS = _ply_strength_fields(lam)
        n = tXt.shape[0]
        pad = max(n_ply_max - n, 0)

        def padv(v: np.ndarray) -> np.ndarray:
            return np.pad(v, (0, pad), mode="constant") if pad > 0 else v

        rows.append((padv(tXt), padv(tXc), padv(tYt), padv(tYc), padv(tS)))
    if not rows:
        z = np.zeros((1, n_ply_max), dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy(), z.copy()
    Xt = np.row_stack([r[0] for r in rows])
    Xc = np.row_stack([r[1] for r in rows])
    Yt = np.row_stack([r[2] for r in rows])
    Yc = np.row_stack([r[3] for r in rows])
    S12 = np.row_stack([r[4] for r in rows])
    return Xt, Xc, Yt, Yc, S12


class DesignEvaluator:
    def __init__(
        self,
        problem: DesignProblem,
        *,
        resultant_driver: BeamResultantDriverProtocol | None = None,
    ):
        self.problem = problem
        self._resultant_driver: BeamResultantDriverProtocol = (
            resultant_driver if resultant_driver is not None else PrescribedResultantDriver()
        )
        n = int(problem.blade_geometry.z_stations.shape[0])
        self._caches = init_station_caches(n)

    def evaluate(self, dv: DesignVector) -> DesignEvaluation:
        p = self.problem
        bg = p.blade_geometry
        n_s = int(bg.z_stations.shape[0])
        section_defs = SectionBuilder.build(dv, bg)
        d_idx = dirty_indices(dv, self._caches)
        res_map = solve_dirty_stations(section_defs, d_idx, n_workers=p.n_workers)
        for i in d_idx:
            self._caches[i].result = res_map[i]
            self._caches[i].t_skin = float(dv.t_skin[i])
            self._caches[i].t_cap = float(dv.t_cap[i])
            self._caches[i].t_web = float(dv.t_web[i])
            self._caches[i].dirty = False

        K7_stack = np.stack([self._caches[i].result.K7 for i in range(n_s)], axis=0)
        stiffness_metric = integrated_k7_trace(K7_stack, bg.z_stations)
        beam_res = self._resultant_driver.drive(K7_stack, p.extreme_loads, bg)
        resultants = beam_res.resultants
        for i in range(n_s):
            section_defs[i].R_deformed = beam_res.nodal_R[i]

        ref = self._caches[0].result
        assert ref is not None
        n_ply_max = ref.Q_bar.shape[1]

        comp_basis = np.stack([self._caches[i].result.composite_resultant_basis for i in range(n_s)], axis=0)
        iso_basis = np.stack([self._caches[i].result.isotropic_resultant_basis for i in range(n_s)], axis=0)

        comp_res = np.einsum("sm,spmr->spr", resultants, comp_basis, optimize=True)
        iso_res = np.einsum("sm,spmr->spr", resultants, iso_basis, optimize=True)

        ABD_inv = np.stack([self._caches[i].result.ABD_inv for i in range(n_s)], axis=0)
        Q_bar = np.stack([self._caches[i].result.Q_bar for i in range(n_s)], axis=0)
        T_ply = np.stack([self._caches[i].result.T_ply for i in range(n_s)], axis=0)
        z_ply = np.stack([self._caches[i].result.z_ply for i in range(n_s)], axis=0)

        sigma_sec = clpt_ply_stresses_section_frame(comp_res, ABD_inv, Q_bar, z_ply)
        sigma_mat = rotate_plies_to_material(sigma_sec, T_ply)

        Xt, Xc, Yt, Yc, S12 = _composite_strength_stack(section_defs[0].subcomponents, n_ply_max)
        Xt_b = Xt[None, :, :]
        Xc_b = Xc[None, :, :]
        Yt_b = Yt[None, :, :]
        Yc_b = Yc[None, :, :]
        S12_b = S12[None, :, :]
        fi_tw = fail_mod.tsai_wu_fi_plies(sigma_mat, Xt_b, Xc_b, Yt_b, Yc_b, S12_b)

        iso_t = np.stack([self._caches[i].result.iso_thickness for i in range(n_s)], axis=0)
        iso_sig = np.stack([self._caches[i].result.iso_sigma_allow for i in range(n_s)], axis=0)
        sigma_iso = iso_res / np.maximum(iso_t[:, :, None], 1e-12)
        fi_vm = fail_mod.von_mises_plane_stress_fi(sigma_iso, iso_sig[:, :, None])

        fi_delam = None
        max_del = None
        if p.enable_tier3_delam and n_s >= 2:
            Zt = ref.Zt
            S13 = ref.S13
            S23 = ref.S23
            tau_if = interlaminar_stress_recovery(sigma_sec, bg.z_stations, z_ply[0])
            fi_delam = delamination_fi(tau_if, Zt, S13, S23)
            max_del = float(np.max(fi_delam))

        mass = mass_objective(dv, bg)
        max_tw = float(np.max(fi_tw))
        max_vm = float(np.max(fi_vm))

        return DesignEvaluation(
            dv=dv,
            mass=mass,
            stiffness_metric=stiffness_metric,
            resultants=resultants,
            fi_tw=fi_tw,
            fi_vm=fi_vm,
            fi_delam=fi_delam,
            max_fi_tw=max_tw,
            max_fi_vm=max_vm,
            max_fi_delam=max_del,
        )
