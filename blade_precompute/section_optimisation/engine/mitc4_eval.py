"""MITC4 section stress for optimisation: shell N/M → CLPT ply Hashin, isotropic von Mises, scalar index."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.failure_criteria import von_mises_plane_stress_fi
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial
from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_optimisation.engine.beam_k7 import bimoment_derivative_z
from blade_precompute.section_shell_model.lib.recovery_adapter import _ensure_stress_imports, run_section_with_mitc4_shell


def _airfoil_xy_from_profile(prof: Any) -> NDArray[np.float64]:
    v = np.asarray(
        getattr(prof, "vertices", prof),
        dtype=np.float64,
    )
    if v.ndim != 2 or v.shape[1] < 2:
        raise ValueError("airfoil profile must be Nx2 or more in last dim.")
    return v[:, :2].copy()


def _spar_x_metres(web_positions: NDArray[np.float64], chord_m: float) -> list[float]:
    w = np.asarray(web_positions, dtype=np.float64).ravel()
    chord = float(chord_m)
    return [float((xi + 0.5) * chord) for xi in w]


def _first_skin_laminate(section_def: SectionDefinition) -> LaminateDefinition | None:
    for sub in section_def.subcomponents:
        if "skin" in sub.name.lower() and isinstance(sub.material, LaminateDefinition):
            return sub.material
    for sub in section_def.subcomponents:
        if isinstance(sub.material, LaminateDefinition):
            return sub.material
    return None


def _strength_ref_pa(lam: LaminateDefinition | None) -> float:
    if lam is None or not lam.plies:
        return 500e6
    p0, _ = lam.plies[0]
    return float(getattr(p0, "Xt", 500e6)) if getattr(p0, "Xt", None) is not None else 500e6


class LaminateDefinitionMitc4SkinAdapter:
    """Bridge ``LaminateDefinition`` → thin-wall + MITC4 skin contract.

    ``multi_cell_blade_section.Panel`` expects ``lam.E``, ``lam.nu``, ``lam.t``, and
    ``lam.build_plies()`` returning ``examples/section_stress_model/lib/laminate_clpt.Ply`` objects.
    :class:`~blade_precompute.section_properties.engine.laminate.LaminateDefinition` stores
    ``OrthotropicPly`` + angles instead; this adapter maps ply-by-ply so ``abd_stack`` in MITC4
    matches the design CLT stack (not an isotropic surrogate stack).
    """

    __slots__ = ("E", "nu", "t", "name", "_mc_plies")

    def __init__(self, ld: LaminateDefinition) -> None:
        if not ld.plies:
            raise ValueError("LaminateDefinitionMitc4SkinAdapter requires non-empty plies.")
        _ensure_stress_imports()
        from lib.laminate_clpt import Ply, homogenized_axial_modulus  # type: ignore[import-untyped]

        mc_plies: list[Any] = []
        for oply, ang in ld.plies:
            mc_plies.append(
                Ply(
                    E1=float(oply.E1),
                    E2=float(oply.E2),
                    G12=float(oply.G12),
                    nu12=float(oply.nu12),
                    theta_deg=float(ang),
                    t=float(oply.t_ply),
                )
            )
        self._mc_plies = mc_plies
        h = float(sum(p.t for p in mc_plies))
        self.t = max(h, 1e-12)
        e_ax = float(homogenized_axial_modulus(mc_plies))
        self.E = max(e_ax, 1e3)
        nu_acc = sum(float(oply.nu12) * float(oply.t_ply) for oply, _ in ld.plies)
        self.nu = float(nu_acc / self.t) if self.t > 0.0 else 0.3
        self.name = "LaminateDefinition_skin"

    def build_plies(self) -> list[Any]:
        """Orthotropic stack bottom→top (same order as :meth:`LaminateDefinition.build_ABD`)."""
        return list(self._mc_plies)


def _coerce_skin_lam_for_mitc4(skin: Any) -> Any:
    """Pass ``LaminateDefinition`` through :class:`LaminateDefinitionMitc4SkinAdapter`; leave other types unchanged."""
    if skin is None or not isinstance(skin, LaminateDefinition):
        return skin
    if not skin.plies:
        return skin
    return LaminateDefinitionMitc4SkinAdapter(skin)


def _composite_subcomponents(section_def: SectionDefinition) -> list:
    return [s for s in section_def.subcomponents if s.is_composite]


def _isotropic_subcomponents(section_def: SectionDefinition) -> list[SubcomponentGeometry]:
    return [s for s in section_def.subcomponents if s.is_isotropic]


def _isotropic_plies_clpt(iso: SubcomponentGeometry) -> list[Any] | None:
    """Single isotropic layer as ``lib.laminate_clpt.Ply`` for CLPT stress from (N, M)."""
    mat = iso.material
    if not isinstance(mat, IsotropicMaterial):
        return None
    _ensure_stress_imports()
    from lib.laminate_clpt import Ply  # type: ignore[import-untyped]

    e = float(mat.E)
    nu = float(mat.nu)
    t = max(float(iso.thickness), 1e-12)
    g12 = e / (2.0 * (1.0 + nu))
    return [Ply(E1=e, E2=e, G12=g12, nu12=nu, theta_deg=0.0, t=t)]


def _iso_name_for_mitc4_panel(panel_name: str, iso_names: list[str]) -> str | None:
    """Map MITC4 panel label to an isotropic subcomponent ``name`` (heuristic)."""
    pl = (panel_name or "").lower()
    if "uskin" in pl or "lskin" in pl:
        return None
    if "web" in pl and "skin" not in pl:
        for n in iso_names:
            if "web" in n.lower():
                return n
    if "cap" in pl or "spar" in pl or "boom" in pl:
        for n in iso_names:
            nl = n.lower()
            if "cap" in nl or "spar" in nl or "boom" in nl:
                return n
        return None
    if "leading" in pl or pl.strip() in ("le", "le_u", "le_l"):
        for n in iso_names:
            if "lead" in n.lower():
                return n
    return None


def _ci_for_mitc4_panel(panel_name: str, comp_subs: list) -> int | None:
    """Map a multi_cell panel ``name`` to the composite subcomponent row index ``ci``."""
    pl = (panel_name or "").lower()
    if "uskin" in pl or "lskin" in pl or pl.startswith("uskin") or pl.startswith("lskin"):
        for i, s in enumerate(comp_subs):
            nl = s.name.lower()
            if nl == "skin" or (
                "skin" in nl and "web" not in nl and "shear" not in nl
            ):
                return i
        for i, s in enumerate(comp_subs):
            if "skin" in s.name.lower():
                return i
    if "web" in pl and "uskin" not in pl and "lskin" not in pl:
        for i, s in enumerate(comp_subs):
            if "web" in s.name.lower():
                return i
    return None


def _mitc4_clt_sink_station0(
    bundle: Any, clt_station0_sink: dict[str, Any] | None, s_idx: int
) -> None:
    if clt_station0_sink is None or s_idx != 0:
        return
    clt_station0_sink.clear()
    abd = getattr(bundle, "mitc4_panel_abd", None)
    if abd is not None:
        clt_station0_sink["mitc4_panel_abd"] = np.asarray(abd, dtype=np.float64).copy()
    tm = getattr(bundle, "mitc4_panel_thickness_m", None)
    if tm is not None:
        clt_station0_sink["mitc4_panel_thickness_m"] = np.asarray(tm, dtype=np.float64).copy()
    ge = getattr(bundle, "mitc4_panel_G_eff", None)
    if ge is not None:
        clt_station0_sink["mitc4_panel_G_eff"] = np.asarray(ge, dtype=np.float64).copy()
    lab = getattr(bundle, "mitc4_panel_labels", None)
    if lab:
        clt_station0_sink["mitc4_panel_labels"] = np.asarray(lab, dtype=object)


def _fi_mitc4_scalar_from_bundle(bundle: Any, sig_ref: float) -> float:
    sig_vals: list[float] = []
    if hasattr(bundle, "sig_p") and bundle.sig_p is not None:
        sp = bundle.sig_p
        if isinstance(sp, (list, tuple)):
            for x in sp:
                try:
                    a = np.asarray(x, dtype=np.float64)
                    if a.size:
                        sig_vals.extend(np.abs(a.ravel()).tolist())
                except Exception:
                    pass
    if not sig_vals and bundle.all_panel_mitc4_results:
        for pr in (bundle.all_panel_mitc4_results or []):
            for row in pr or []:
                for attr in ("sigma_xx_pa", "tau_xy_pa"):
                    if hasattr(row, attr):
                        sig_vals.append(abs(float(getattr(row, attr, 0.0))))
    if not sig_vals:
        return 0.0
    return float(np.max(sig_vals) / max(sig_ref, 1.0))


def mitc4_shell_fi_batch(
    section_defs: list[SectionDefinition],
    R_sec: NDArray[np.float64],
    bg: OptimBladeGeometry,
    n_ply_max: int,
    composite_subcomp_names: list[str] | None,
    *,
    isotropic_subcomp_names: list[str] | None = None,
    n_elements_per_panel: int = 10,
    clt_station0_sink: dict[str, Any] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.bool_]]:
    """One MITC4 section solve per station: scalar index + CLPT Hashin + CLPT von Mises (isotropic).

    Uses ``ShellPanelResultants`` (N, M) with ``clpt_ply_failure_indices`` for composite rows and
    mid-ply stress → von Mises for mapped isotropic rows.

    Returns
    -------
    fi_mitc4
        ``(n_stations,)`` — |σ|/X_ref scalar shell index.
    fi_hashin_mitc4
        ``(n_stations, n_composite, n_ply_max)``.
    row_mask_composite
        ``(n_stations, n_composite)`` — Hashin row updated from MITC4+CLPT.
    fi_vm_mitc4
        ``(n_stations, n_isotropic)`` — von Mises FI from MITC4 N/M where mapped.
    row_mask_vm
        ``(n_stations, n_isotropic)`` — VM row updated from MITC4.
    """
    n_s = len(section_defs)
    if not section_defs:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0, max(0, n_ply_max)), dtype=np.float64),
            np.zeros((0, 0), dtype=np.bool_),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.bool_),
        )
    s0 = section_defs[0]
    comp0 = _composite_subcomponents(s0)
    iso0 = _isotropic_subcomponents(s0)
    names = [str(x) for x in (composite_subcomp_names or [])]
    if not names and comp0:
        names = [s.name for s in comp0]
    inames = [str(x) for x in (isotropic_subcomp_names or [])]
    if not inames and iso0:
        inames = [s.name for s in iso0]
    n_c = len(names)
    n_i = len(inames)
    if n_c == 0:
        return (
            np.zeros(n_s, dtype=np.float64),
            np.zeros((n_s, 0, max(0, n_ply_max)), dtype=np.float64),
            np.zeros((n_s, 0), dtype=np.bool_),
            np.zeros((n_s, max(0, n_i)), dtype=np.float64),
            np.zeros((n_s, max(0, n_i)), dtype=np.bool_),
        )
    out_h = np.zeros((n_s, n_c, n_ply_max), dtype=np.float64)
    msk = np.zeros((n_s, n_c), dtype=np.bool_)
    out_vm = np.zeros((n_s, n_i), dtype=np.float64)
    msk_vm = np.zeros((n_s, n_i), dtype=np.bool_)
    fi_s = np.zeros(n_s, dtype=np.float64)

    z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    B = R_sec[:, 6] if R_sec.shape[1] > 6 else np.zeros(n_s, dtype=np.float64)
    dB = bimoment_derivative_z(B, z)
    wpos = np.asarray(bg.web_positions, dtype=np.float64).ravel()

    _ensure_stress_imports()
    from lib.laminate_clpt import clpt_ply_failure_indices  # type: ignore[import-untyped]

    for s_idx in range(n_s):
        try:
            sdef = section_defs[s_idx]
            comp_subs = _composite_subcomponents(sdef)
            if not comp_subs:
                continue
            prof = bg.airfoil_profiles[s_idx]
            airfoil = _airfoil_xy_from_profile(prof)
            chord = float(bg.chord[s_idx])
            spars = _spar_x_metres(wpos, chord)
            skin_ld = _first_skin_laminate(sdef)
            sig_ref = _strength_ref_pa(skin_ld)
            skin = _coerce_skin_lam_for_mitc4(skin_ld)
            r = R_sec[s_idx]
            bundle = run_section_with_mitc4_shell(
                airfoil,
                spars,
                skin_lam=skin,
                N=float(r[0]),
                Vy=float(r[4]),
                Vz=float(r[5]),
                My=float(r[1]),
                Mz=float(r[2]),
                T=float(r[3]),
                B=float(B[s_idx]),
                dB_dx=float(dB[s_idx]),
                n_elements_per_panel=int(n_elements_per_panel),
            )
            _mitc4_clt_sink_station0(bundle, clt_station0_sink, s_idx)
            fi_s[s_idx] = _fi_mitc4_scalar_from_bundle(bundle, sig_ref)

            panels = getattr(bundle, "panels", None) or []
            all_res = getattr(bundle, "all_panel_mitc4_results", None) or []
            iso_by_name = {s.name: s for s in sdef.subcomponents if s.is_isotropic}
            vm_acc = np.zeros(n_i, dtype=np.float64)
            vm_have = np.zeros(n_i, dtype=np.bool_)

            for pi, panel in enumerate(panels):
                if pi >= len(all_res):
                    continue
                pnm = str(getattr(panel, "name", f"p{pi}"))
                ci = _ci_for_mitc4_panel(pnm, comp_subs)
                if ci is not None and 0 <= ci < n_c and ci < len(comp_subs):
                    lam = comp_subs[ci].material
                    if isinstance(lam, LaminateDefinition) and lam.plies:
                        n_p = len(lam.plies)
                        row_max = np.zeros(n_p, dtype=np.float64)
                        have = False
                        for row in all_res[pi] or []:
                            if row is None:
                                continue
                            n_vec = row.to_N_vec()
                            m_vec = row.to_M_vec()
                            ad = LaminateDefinitionMitc4SkinAdapter(lam)
                            plies = ad.build_plies()
                            p0, _ = lam.plies[0]
                            fi_vec, _, _, _ = clpt_ply_failure_indices(
                                plies,
                                np.asarray(n_vec, dtype=np.float64).ravel()[:3],
                                np.asarray(m_vec, dtype=np.float64).ravel()[:3],
                                float(getattr(p0, "Xt", 1e9)),
                                float(getattr(p0, "Xc", 1e9)),
                                float(getattr(p0, "Yt", 1e9)),
                                float(getattr(p0, "Yc", 1e9)),
                                float(getattr(p0, "S12", 1e9)),
                            )
                            fi_v = np.asarray(fi_vec, dtype=np.float64).ravel()
                            for k in range(min(n_p, int(fi_v.shape[0]))):
                                row_max[k] = max(row_max[k], float(fi_v[k]))
                            have = True
                        if have:
                            msk[s_idx, ci] = True
                            out_h[s_idx, ci, :n_p] = row_max

                if n_i:
                    matched_iso = _iso_name_for_mitc4_panel(pnm, inames)
                    if matched_iso is None or matched_iso not in iso_by_name:
                        continue
                    sub_iso = iso_by_name[matched_iso]
                    plies_i = _isotropic_plies_clpt(sub_iso)
                    mat_i = sub_iso.material
                    if not plies_i or not isinstance(mat_i, IsotropicMaterial):
                        continue
                    try:
                        j = inames.index(matched_iso)
                    except ValueError:
                        continue
                    sig_allow = float(mat_i.sigma_allow)
                    for row in all_res[pi] or []:
                        if row is None:
                            continue
                        n_vec = row.to_N_vec()
                        m_vec = row.to_M_vec()
                        _, _, _, sig_lam = clpt_ply_failure_indices(
                            plies_i,
                            np.asarray(n_vec, dtype=np.float64).ravel()[:3],
                            np.asarray(m_vec, dtype=np.float64).ravel()[:3],
                            1e9,
                            1e9,
                            1e9,
                            1e9,
                            1e9,
                        )
                        sig0 = np.asarray(sig_lam[0], dtype=np.float64).ravel()[:3]
                        vm_one = von_mises_plane_stress_fi(
                            sig0.reshape(1, 3),
                            np.array([sig_allow], dtype=np.float64),
                        )
                        vm_acc[j] = max(vm_acc[j], float(vm_one[0]))
                        vm_have[j] = True
            if n_i:
                for j in range(n_i):
                    if vm_have[j]:
                        msk_vm[s_idx, j] = True
                        out_vm[s_idx, j] = vm_acc[j]
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"mitc4_shell_fi_batch: station {s_idx} skipped ({exc!r})",
                stacklevel=2,
            )
    return fi_s, out_h, msk, out_vm, msk_vm
