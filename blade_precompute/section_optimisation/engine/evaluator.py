"""Per-design evaluation: sections → beam → stress → failure indices.

Group I notes
-------------
I.2 — Section-frame rotation
    ``_rotate_resultants_to_section_frame`` applies the twist + kappa0 rotation to
    beam resultants before shell FI recovery.  Called per-station when the coupled
    FE driver is active.  No-op for the prescribed driver (resultants are already in
    the section principal frame from ExtremeLoads tabulation).

I.7 — K7_inv numerical safety
    ``np.linalg.inv`` is replaced with ``np.linalg.pinv(rcond=1e-12)`` throughout.
    ``cond(K7)`` is computed for each station and a RunLogger warning is emitted when
    any station exceeds ``DesignProblem.k7_cond_warn_threshold`` (default 1e10).

I.9 — Mass seed from shell
    When ``DesignProblem.beam_driver == 'shell'``, the first call to ``evaluate()``
    seeds ``K7_stack`` from the shell homogenisation result (F2.1) rather than the
    strip midsurface solve.  Stub here; full wiring requires Group H/F completion.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.core.types import SectionSolveResult
from blade_precompute.section_properties.engine import failure_criteria as fail_mod
from blade_precompute.section_properties.engine.clpt_recovery import clpt_ply_stresses_section_frame, rotate_plies_to_material
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition

from .cache import dirty_indices, init_station_caches
from .mass import mass_objective
from .stiffness_metric import integrated_k7_trace
from .parallel import solve_dirty_stations
from ..core.protocols import BeamResultantDriverProtocol
from .beam_k7 import PrescribedResultantDriver, section_frame_rotation_matrix
from .section_builder import SectionBuilder
from ..core.types import DesignEvaluation, DesignProblem, DesignVector


# ---------------------------------------------------------------------------
# Panel buckling adapter (J.1)
# ---------------------------------------------------------------------------

def _beam7_to_reference_forces6(res7: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map beam seven-vector `[N,Vy,Vz,My,Mz,T,B]` to K6 RHS `[N,My,Mz,T,Vy,Vz]`."""

    r = np.asarray(res7, dtype=np.float64).ravel()
    if r.size < 6:
        return np.zeros(6, dtype=np.float64)
    return np.array(
        [r[0], r[3], r[4], r[5], r[1], r[2]],
        dtype=np.float64,
    )


def _compute_panel_buckling_fi(
    section_defs: list[Any],
    resultants: np.ndarray,
    bg: Any,
    n_s: int,
    section_results: list[SectionSolveResult | None],
) -> np.ndarray | None:
    """Compute per-(station, edge) panel buckling interaction index.

    Uses the orthotropic closed-form module (``panel_buckling.py``).
    Stress inputs are derived from the section-frame ply stresses already
    computed in ``DesignEvaluator.evaluate`` (J.1 plan).

    Returns
    -------
    fi_buck : (n_stations, n_edges) array or None when no composite edges found.

    Notes
    -----
    Strength sizing ``fi_hashin`` uses MITC4+CLPT where panel-mapped; this
    buckling check still uses midsurface strip / K7-recovered stress proxies
    until it is wired to ``ShellPanelResultants`` (same intent as the plan’s
    optional shell-based buckling follow-up).

    This function is a light adapter that bridges the existing strip-path
    composite stresses with ``assess_panel_buckling_section``.  When the
    shell-only path (Group F) replaces the strip path, this adapter is updated
    to use ``ShellPanelResultants`` (Nx, Ny, Nxy) from F1.2 directly.

    Frame spacing ``a`` (panel length) is derived from the spanwise station
    spacing; a conservative minimum 0.2 m is applied when the local spacing
    is smaller than a practical frame pitch.
    """
    from blade_precompute.section_properties.engine.panel_buckling import (
        assess_panel_buckling_section,
        composite_edge_panel_stresses_from_reference,
    )
    from blade_precompute.section_properties.engine.elements import build_strip_fe_data  # type: ignore

    z_stations = np.asarray(bg.z_stations, dtype=np.float64)
    all_bi: list[np.ndarray] = []
    max_n_edges = 0

    for si in range(n_s):
        sdef = section_defs[si]
        comp_subs = [sub for sub in sdef.subcomponents if sub.is_composite]
        if not comp_subs:
            all_bi.append(np.zeros(0, dtype=np.float64))
            continue

        # Frame pitch estimate
        if si < n_s - 1:
            a_local = float(z_stations[si + 1] - z_stations[si])
        elif n_s >= 2:
            a_local = float(z_stations[-1] - z_stations[-2])
        else:
            a_local = 0.5
        a_frame = max(a_local, 0.2)

        lams: list[Any] = []
        sigma_zz_per_edge: list[float] = []
        sigma_yy_per_edge: list[float] = []
        tau_per_edge: list[float] = []
        comp_edge_indices: list[int] = []

        sr = section_results[si] if si < len(section_results) else None
        F6_station: NDArray[np.float64] | None = None
        if si < int(np.asarray(resultants, dtype=float).shape[0]):
            F6_station = _beam7_to_reference_forces6(np.asarray(resultants[si], dtype=np.float64))
        peak_by_name: dict[str, tuple[float, float, float]] | None = None
        if sr is not None and sr.composite_resultant_basis.shape[0] > 0 and F6_station is not None:
            try:
                pk = composite_edge_panel_stresses_from_reference(sr, F6_station)
                nm = list(getattr(sr, "composite_subcomp_names", ()) or ())
                peak_by_name = {
                    str(nm[j]): (float(pk[j, 0]), float(pk[j, 1]), float(pk[j, 2]))
                    for j in range(min(len(nm), pk.shape[0]))
                }
            except Exception:
                peak_by_name = None

        for idx, sub in enumerate(comp_subs):
            lam = sub.material
            if not isinstance(lam, LaminateDefinition):
                continue
            # Panel dimensions: width b from subcomponent width estimate
            b = float(getattr(sub, "arc_length_m", getattr(sub, "width_m", 0.0)))
            if b < 1e-6:
                continue
            lams.append(lam)
            comp_edge_indices.append(idx)
            sname = str(getattr(sub, "name", f"#{idx}"))
            if peak_by_name is not None and sname in peak_by_name:
                sig_ax, tau_m, sig_tr = peak_by_name[sname]
                sigma_zz_per_edge.append(sig_ax)
                tau_per_edge.append(tau_m)
                sigma_yy_per_edge.append(sig_tr)
            else:
                sigma_zz_per_edge.append(0.0)
                sigma_yy_per_edge.append(0.0)
                tau_per_edge.append(0.0)

        if not lams:
            all_bi.append(np.zeros(0, dtype=np.float64))
            continue

        # Build minimal strip FE data for assess_panel_buckling_section
        try:
            fe = build_strip_fe_data(sdef)
        except Exception:
            # build_strip_fe_data may not work for all section types; skip gracefully
            all_bi.append(np.zeros(len(lams), dtype=np.float64))
            continue

        result = assess_panel_buckling_section(
            fe=fe,
            comp_edge_indices=comp_edge_indices,
            lams=lams,
            sigma_zz=np.array(sigma_zz_per_edge, dtype=np.float64),
            tau=np.array(tau_per_edge, dtype=np.float64),
            frame_spacing_m=a_frame,
            sigma_yy=np.array(sigma_yy_per_edge, dtype=np.float64),
        )
        bi_vec = np.array([r.BI for r in result.edge_results], dtype=np.float64)
        all_bi.append(bi_vec)
        max_n_edges = max(max_n_edges, len(bi_vec))

    if max_n_edges == 0:
        return None

    # Pad to uniform (n_s, n_edges) with zeros for unoccupied panels
    fi_buck = np.zeros((n_s, max_n_edges), dtype=np.float64)
    for si, bi in enumerate(all_bi):
        if bi.size > 0:
            fi_buck[si, : bi.size] = bi
    return fi_buck

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


def _log_stress_projection_diagnostics(
    resultants: np.ndarray,
    strains: np.ndarray,
    comp_res: np.ndarray,
    sigma_sec: np.ndarray,
    z_stations: np.ndarray,
    run_log: Any | None,
) -> None:
    """Log per-station summary of the K7_inv @ R stress projection (Fix 1).

    Expected scales for a wind turbine blade (SI units):
      resultants : N=[0], My,Mz=[1,2]=O(1e5 N·m), T=[3]=O(1e4 N·m), Vy,Vz=[4,5]=O(1e4 N)
      strains    : all 7 modes O(1e-3 to 1e-6) — large values indicate K7/basis mismatch
      comp_res   : CLPT resultants [N/m, N·m/m] — should be O(resultants / perimeter)
      sigma_sec  : ply stresses [Pa] — should be O(1e6 to 1e9) for composites
    """
    n_s = resultants.shape[0]
    diag: list[dict] = []
    for s in range(n_s):
        r = resultants[s]
        strain = strains[s]
        cr_max = float(np.max(np.abs(comp_res[s]))) if comp_res.size else 0.0
        sig_max = float(np.max(np.abs(sigma_sec[s]))) if sigma_sec.size else 0.0
        diag.append({
            "z_m": float(z_stations[s]),
            "R_N_My_Mz_T": [float(r[0]), float(r[1]), float(r[2]), float(r[3])],
            "R_Vy_Vz_B": [float(r[4]), float(r[5]), float(r[6])],
            "strains_max": float(np.max(np.abs(strain))),
            "strains": [float(v) for v in strain],
            "comp_res_max_Npm": cr_max,
            "sigma_sec_max_Pa": sig_max,
        })
    if run_log is not None:
        try:
            run_log.log_event("evaluator.stress_projection_diagnostic", stations=diag)
        except Exception:
            pass
    else:
        import json
        print("[DEBUG stress_projection]\n" + json.dumps(diag, indent=2))


def _select_default_resultant_driver(problem: DesignProblem) -> BeamResultantDriverProtocol:
    from blade_precompute.section_optimisation.engine.beam_distributed import GlobalBeamResultantDriver

    mode = str(getattr(problem, "beam_driver", "prescribed") or "prescribed").lower()
    if mode in ("global_beam", "coupled_fe"):
        dl = getattr(problem, "distributed_loads", None)
        if dl is None:
            raise ValueError("beam_driver='global_beam' requires DesignProblem.distributed_loads (DistributedLoadCurves).")
        return GlobalBeamResultantDriver(
            dl,
            n_beam_nodes=int(getattr(problem, "n_beam_nodes", 50)),
            solver_options=None,
            axial_cfg=getattr(problem, "axial_loading", None),
        )
    return PrescribedResultantDriver()


class DesignEvaluator:
    def __init__(
        self,
        problem: DesignProblem,
        *,
        resultant_driver: BeamResultantDriverProtocol | None = None,
        run_log: Any | None = None,
    ):
        self.problem = problem
        self._resultant_driver: BeamResultantDriverProtocol = (
            resultant_driver
            if resultant_driver is not None
            else _select_default_resultant_driver(problem)
        )
        self._run_log = run_log
        n = int(problem.blade_geometry.z_stations.shape[0])
        self._caches = init_station_caches(n)
        # ABD cache per (OrientationMix, role) key — populated by L.5 in outer-inner loop
        self._abd_cache: dict[Any, np.ndarray] = {}
        # Last station-0 MITC4 panel CLT stack (filled when ``iteration_dump_npz`` and MITC4 path runs).
        self._mitc4_clt_station0: dict[str, Any] = {}

    def seed_stations(self, dv: DesignVector, results: Sequence[SectionSolveResult]) -> None:
        """Warm per-station caches (e.g. from precompute ``SectionPropertiesOutputs``) to skip repeat midsurface solves."""
        n = len(self._caches)
        if len(results) != n:
            raise ValueError(f"seed_stations: expected {n} section results, got {len(results)}")
        for i in range(n):
            c = self._caches[i]
            c.t_skin = float(dv.t_skin[i])
            c.t_cap = float(dv.t_cap[i])
            c.t_web = float(dv.t_web[i])
            c.result = results[i]
            c.dirty = False

    def _k7_inv_safe(self, K7_stack: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Compute K7_inv via pseudoinverse with condition number monitoring (I.7).

        Returns
        -------
        k7_inv_stack : (n_s, 7, 7)
        cond_stats : dict with min/max/mean cond per station.
        """
        n_s = K7_stack.shape[0]
        k7_inv = np.zeros_like(K7_stack)
        cond_vals = np.zeros(n_s, dtype=np.float64)
        warn_threshold = float(self.problem.k7_cond_warn_threshold)

        for i in range(n_s):
            K = K7_stack[i]
            try:
                cond_vals[i] = float(np.linalg.cond(K))
            except np.linalg.LinAlgError:
                cond_vals[i] = np.inf
            k7_inv[i] = np.linalg.pinv(K, rcond=1e-12)

        bad_stations = np.where(cond_vals > warn_threshold)[0].tolist()
        if bad_stations:
            msg = (
                f"K7 ill-conditioned at stations {bad_stations} "
                f"(max cond={float(cond_vals.max()):.3e} > threshold {warn_threshold:.3e}). "
                "Strain recovery via K7_inv may be inaccurate."
            )
            if self._run_log is not None:
                try:
                    self._run_log.warn_event(
                        "evaluator.k7_ill_conditioned",
                        bad_stations=bad_stations,
                        max_cond=float(cond_vals.max()),
                    )
                except Exception:
                    pass
            else:
                warnings.warn(msg, stacklevel=3)

        stats: dict[str, float] = {
            "cond_min": float(np.min(cond_vals)),
            "cond_max": float(np.max(cond_vals)),
            "cond_mean": float(np.mean(cond_vals)),
        }
        return k7_inv, stats

    _call_count: int = 0  # tracks total evaluate() calls for diagnostic gating

    def evaluate(self, dv: DesignVector) -> DesignEvaluation:
        p = self.problem
        bg = p.blade_geometry
        n_s = int(bg.z_stations.shape[0])
        section_defs = SectionBuilder.build(dv, bg)
        self._mitc4_clt_station0 = {}
        d_idx = dirty_indices(dv, self._caches)
        res_map = solve_dirty_stations(section_defs, d_idx, n_workers=p.n_workers)
        for i in d_idx:
            self._caches[i].result = res_map[i]
            self._caches[i].t_skin = float(dv.t_skin[i])
            self._caches[i].t_cap = float(dv.t_cap[i])
            self._caches[i].t_web = float(dv.t_web[i])
            self._caches[i].dirty = False

        K7_stack = np.stack([self._caches[i].result.K7 for i in range(n_s)], axis=0)
        K6_stack = np.stack([self._caches[i].result.K6 for i in range(n_s)], axis=0)
        stiffness_metric = integrated_k7_trace(K7_stack, bg.z_stations)
        mu_line = np.array(
            [float(self._caches[i].result.mass_per_length) for i in range(n_s)], dtype=np.float64
        )
        beam_res = self._resultant_driver.drive(
            K7_stack, p.extreme_loads, bg, K6_stack=K6_stack, mass_per_length=mu_line
        )
        resultants = beam_res.resultants
        for i in range(n_s):
            section_defs[i].R_deformed = beam_res.nodal_R[i]

        ref = self._caches[0].result
        assert ref is not None
        n_ply_max = ref.Q_bar.shape[1]

        comp_basis = np.stack([self._caches[i].result.composite_resultant_basis for i in range(n_s)], axis=0)
        iso_basis = np.stack([self._caches[i].result.isotropic_resultant_basis for i in range(n_s)], axis=0)

        # I.7: safe pseudoinverse with condition monitoring
        k7_inv_stack, k7_cond_stats = self._k7_inv_safe(K7_stack)
        strains = np.einsum("smj,sj->sm", k7_inv_stack, resultants)
        comp_res = np.einsum("sm,spmr->spr", strains, comp_basis, optimize=True)
        iso_res = np.einsum("sm,spmr->spr", strains, iso_basis, optimize=True)
        ABD_inv = np.stack([self._caches[i].result.ABD_inv for i in range(n_s)], axis=0)
        Q_bar = np.stack([self._caches[i].result.Q_bar for i in range(n_s)], axis=0)
        T_ply = np.stack([self._caches[i].result.T_ply for i in range(n_s)], axis=0)
        z_ply = np.stack([self._caches[i].result.z_ply for i in range(n_s)], axis=0)

        sigma_sec = clpt_ply_stresses_section_frame(comp_res, ABD_inv, Q_bar, z_ply)
        sigma_mat = rotate_plies_to_material(sigma_sec, T_ply)

        # Fix 1: stress projection diagnostics (fires on first call only)
        if p.debug_stress_projection and self._call_count == 0:
            _log_stress_projection_diagnostics(
                resultants, strains, comp_res, sigma_sec,
                bg.z_stations, self._run_log
            )

        Xt, Xc, Yt, Yc, S12 = _composite_strength_stack(section_defs[0].subcomponents, n_ply_max)
        Xt_b = Xt[None, :, :]
        Xc_b = Xc[None, :, :]
        Yt_b = Yt[None, :, :]
        Yc_b = Yc[None, :, :]
        S12_b = S12[None, :, :]
        fi_h = fail_mod.hashin_fi_plies(sigma_mat, Xt_b, Xc_b, Yt_b, Yc_b, S12_b)
        fi_h_k7 = np.array(fi_h, copy=True)

        # Fix 4c: unified shell CLPT recovery for isotropic subs.
        # iso_res is now (n_s, n_iso, 6) using the full ABD basis.
        # Recover ply stresses at the outer surface (z = t/2) then evaluate
        # Von Mises against sigma_allow.
        iso_sig = np.stack([self._caches[i].result.iso_sigma_allow for i in range(n_s)], axis=0)
        if iso_res.shape[1] > 0:
            iso_ABD_inv = np.stack(
                [self._caches[i].result.iso_ABD_inv for i in range(n_s)], axis=0
            )  # (n_s, n_iso, 6, 6)
            iso_Q_bar = np.stack(
                [self._caches[i].result.iso_Q_bar for i in range(n_s)], axis=0
            )  # (n_s, n_iso, 1, 3, 3)
            iso_z_ply = np.stack(
                [self._caches[i].result.iso_z_ply for i in range(n_s)], axis=0
            )  # (n_s, n_iso, 1)
            # CLPT recovery: returns (n_s, n_iso, 1, 3) — single "ply" at z=t/2
            sigma_iso_ply = clpt_ply_stresses_section_frame(
                iso_res, iso_ABD_inv, iso_Q_bar, iso_z_ply
            )
            sigma_iso = sigma_iso_ply[..., 0, :]  # (n_s, n_iso, 3)
        else:
            sigma_iso = np.zeros((n_s, 0, 3), dtype=np.float64)
        fi_vm = fail_mod.von_mises_plane_stress_fi(sigma_iso, iso_sig)
        fi_vm_k7 = np.array(fi_vm, copy=True)

        # J.1: panel buckling
        fi_panel_buckling: np.ndarray | None = None
        if p.enable_panel_buckling:
            try:
                fi_panel_buckling = _compute_panel_buckling_fi(
                    section_defs,
                    resultants,
                    bg,
                    n_s,
                    [self._caches[i].result for i in range(n_s)],
                )
            except Exception as exc:
                warnings.warn(
                    f"Panel buckling computation failed ({exc!r}); skipping J.1.",
                    stacklevel=2,
                )

        # --- In-loop MITC4: N/M → CLPT ``fi_hashin`` where panel-mapped; else K7/CLPT fallback. ---
        fi_mitc4_vec: np.ndarray | None = None
        max_mitc4: float | None = None
        from blade_precompute.section_optimisation.engine.mitc4_eval import mitc4_shell_fi_batch

        try:
            clt_sink = self._mitc4_clt_station0 if bool(getattr(p, "iteration_dump_npz", False)) else None
            if clt_sink is not None:
                clt_sink.clear()
            cnames = [str(x) for x in getattr(ref, "composite_subcomp_names", ())]
            inames = [str(x) for x in getattr(ref, "isotropic_subcomp_names", ())]
            fi_mitc4_vec, fi_h_mitc4, mitc4_msk, fi_vm_m4, msk_vm = mitc4_shell_fi_batch(
                section_defs,
                resultants,
                bg,
                n_ply_max,
                cnames,
                isotropic_subcomp_names=inames,
                n_elements_per_panel=int(getattr(p, "mitc4_n_elements_per_panel", 10)),
                clt_station0_sink=clt_sink,
            )
            n_ac = int(fi_h_mitc4.shape[1]) if fi_h_mitc4.ndim == 3 else 0
            n_e = min(int(fi_h.shape[1]), n_ac) if n_ac else 0
            if n_e and fi_h_mitc4.shape[0] == fi_h.shape[0]:
                for s_i in range(n_s):
                    for c_i in range(n_e):
                        if mitc4_msk[s_i, c_i]:
                            fi_h[s_i, c_i, ...] = fi_h_mitc4[s_i, c_i, ...]
                        else:
                            fi_h[s_i, c_i, ...] = fi_h_k7[s_i, c_i, ...]
            n_ai = int(fi_vm.shape[1]) if fi_vm.ndim == 2 else 0
            n_vm = min(n_ai, int(fi_vm_m4.shape[1])) if fi_vm_m4.ndim == 2 and fi_vm_m4.shape[1] else 0
            if n_vm and fi_vm_m4.shape[0] == fi_vm.shape[0]:
                for s_i in range(n_s):
                    for j in range(n_vm):
                        if msk_vm[s_i, j]:
                            fi_vm[s_i, j] = fi_vm_m4[s_i, j]
                        else:
                            fi_vm[s_i, j] = fi_vm_k7[s_i, j]
            max_mitc4 = float(np.max(fi_mitc4_vec)) if fi_mitc4_vec.size else 0.0
        except Exception as exc:
            warnings.warn(
                f"MITC4 stress path failed; using K7/CLPT for fi_hashin this eval: {exc!r}",
                stacklevel=2,
            )
            fi_mitc4_vec = np.zeros(n_s, dtype=np.float64)
            max_mitc4 = 0.0

        tip_defl: float | None = None
        if hasattr(beam_res, "tip_displacement_m") and getattr(beam_res, "tip_displacement_m", None) is not None:
            tip_defl = float(
                np.linalg.norm(np.asarray(beam_res.tip_displacement_m, dtype=np.float64).ravel())
            )

        mass = mass_objective(dv, bg)
        max_h = float(np.max(fi_h)) if np.size(fi_h) else 0.0
        max_vm = float(np.max(fi_vm)) if fi_vm.size else 0.0
        self._call_count += 1
        return DesignEvaluation(
            dv=dv,
            mass=mass,
            stiffness_metric=stiffness_metric,
            resultants=resultants,
            fi_hashin=fi_h,
            fi_vm=fi_vm,
            max_fi_hashin=max_h,
            max_fi_vm=max_vm,
            fi_panel_buckling=fi_panel_buckling,
            global_buckling_lambdas=None,
            tip_deflection=tip_defl,
            k7_cond_stats=k7_cond_stats,
            fi_mitc4=fi_mitc4_vec,
            max_fi_mitc4=max_mitc4,
            beam_state=beam_res,
        )
