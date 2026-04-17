from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial as SPIsotropic
from blade_precompute.section_properties.engine.mesh import build_line_mesh

from blade_precompute.section_beam_model.gbt import (
    BoundaryConditions,
    CrossSection,
    CrossSectionModalAnalysis,
    IsotropicMaterial as GBTIsotropic,
    Lamina,
    LaminateMaterial,
    MemberBucklingAnalysis,
    MemberBucklingResult,
    SectionLoads,
    WallDefinition,
)

ThicknessRole = Literal["skin", "cap", "web", "fixed"]


def _thickness_role_from_subcomponent_name(name: str) -> ThicknessRole:
    """Match ``SectionBuilder`` / ``_infer_role`` heuristics (no explicit role map)."""
    n = name.lower()
    if "skin" in n:
        return "skin"
    if "cap" in n:
        return "cap"
    if "web" in n:
        return "web"
    return "fixed"


def _subcomponent_to_gbt_material(sub: SubcomponentGeometry) -> GBTIsotropic | LaminateMaterial:
    th = float(max(sub.thickness, 1e-9))
    if isinstance(sub.material, SPIsotropic):
        return GBTIsotropic(E=float(sub.material.E), nu=float(sub.material.nu), t=th)
    if isinstance(sub.material, LaminateDefinition):
        return _laminate_to_gbt(sub.material)
    raise TypeError(f"Unsupported material type for GBT bridge: {type(sub.material)}")


def _laminate_to_gbt(lam: LaminateDefinition) -> LaminateMaterial:
    plies = []
    for ply, ang_deg in lam.plies:
        plies.append(
            Lamina(
                E1=float(ply.E1),
                E2=float(ply.E2),
                G12=float(ply.G12),
                nu12=float(ply.nu12),
                angle=float(ang_deg),
                t=float(ply.t_ply),
            )
        )
    return LaminateMaterial(plies)


def _n_strips_for_segment(seg_len: float, thickness: float) -> int:
    th = max(thickness, 1e-9)
    return int(np.clip(np.ceil(seg_len / max(4.0 * th, 1e-6)), 2, 14))


def wall_definitions_from_line_mesh(
    sd: SectionDefinition,
    *,
    merge_tolerance: float = 1e-6,
    roles: frozenset[ThicknessRole] | None = None,
) -> list[tuple[int, str, ThicknessRole, WallDefinition]]:
    """
    Build GBT ``WallDefinition`` list from the merged midsurface line mesh.

    Returns tuples ``(subcomponent_index, sub_name, role, wall)`` for each kept edge.
    """
    allowed: frozenset[ThicknessRole] = roles or frozenset({"skin", "cap", "web"})
    lm = build_line_mesh(sd, merge_tolerance=merge_tolerance)
    out: list[tuple[int, str, ThicknessRole, WallDefinition]] = []
    for e in range(lm.edges.shape[0]):
        si = int(lm.edge_subcomp[e])
        sub = sd.subcomponents[si]
        role = _thickness_role_from_subcomponent_name(sub.name)
        if role not in allowed:
            continue
        i0, i1 = int(lm.edges[e, 0]), int(lm.edges[e, 1])
        p0 = lm.nodes[i0]
        p1 = lm.nodes[i1]
        seg_len = float(lm.edge_lengths[e]) if e < lm.edge_lengths.shape[0] else float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-12:
            continue
        mat = _subcomponent_to_gbt_material(sub)
        n_strip = _n_strips_for_segment(seg_len, float(sub.thickness))
        wd = WallDefinition(p0.tolist(), p1.tolist(), mat, n_strips=n_strip, name=sub.name)
        out.append((si, sub.name, role, wd))
    return out


def section_definition_to_gbt_cross_section(
    sd: SectionDefinition,
    *,
    use_line_mesh: bool = True,
    merge_tolerance: float = 1e-6,
) -> CrossSection:
    """
    Build a ``CrossSection`` from section midsurface geometry.

    By default uses the merged **line mesh** (1D graph) restricted to skin / cap / web roles.
    Falls back to raw polyline segments if the filtered line mesh is empty.
    """
    if use_line_mesh:
        pairs = wall_definitions_from_line_mesh(sd, merge_tolerance=merge_tolerance)
        walls = [wd for _, _, _, wd in pairs]
        if walls:
            return CrossSection(walls)

    walls_fb: list[WallDefinition] = []
    for sub in sd.subcomponents:
        pts = np.asarray(sub.midsurface_coords, dtype=np.float64)
        th = float(max(sub.thickness, 1e-9))
        mat = _subcomponent_to_gbt_material(sub)
        if pts.shape[0] < 2:
            continue
        for k in range(pts.shape[0] - 1):
            p0 = pts[k]
            p1 = pts[k + 1]
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-12:
                continue
            n_strip = _n_strips_for_segment(seg_len, th)
            walls_fb.append(WallDefinition(p0.tolist(), p1.tolist(), mat, n_strips=n_strip, name=sub.name))
    if not walls_fb:
        raise ValueError("No walls produced from SectionDefinition (empty subcomponents?).")
    return CrossSection(walls_fb)


def cross_section_for_subcomponent_indices(
    sd: SectionDefinition,
    indices: set[int],
    *,
    merge_tolerance: float = 1e-6,
) -> CrossSection:
    """GBT cross-section built only from line-mesh edges belonging to given subcomponents."""
    pairs = wall_definitions_from_line_mesh(sd, merge_tolerance=merge_tolerance)
    walls = [wd for si, _, _, wd in pairs if si in indices]
    if not walls:
        raise ValueError(f"No line-mesh walls for subcomponent indices {sorted(indices)}.")
    return CrossSection(walls)


def line_mesh_meta(sd: SectionDefinition, *, merge_tolerance: float = 1e-6) -> dict[str, Any]:
    lm = build_line_mesh(sd, merge_tolerance=merge_tolerance)
    pairs = wall_definitions_from_line_mesh(sd, merge_tolerance=merge_tolerance)
    by_si: dict[int, int] = {}
    for si, _, _, _ in pairs:
        by_si[si] = by_si.get(si, 0) + 1
    return {
        "n_nodes": int(lm.nodes.shape[0]),
        "n_edges": int(lm.edges.shape[0]),
        "n_wall_definitions": len(pairs),
        "walls_per_subcomponent_index": {str(k): v for k, v in sorted(by_si.items())},
    }


def _effective_axial_force_for_buckling(N_interp: float) -> tuple[float, str]:
    if N_interp < -1e-3:
        return float(N_interp), "extreme_loads"
    nominal = -max(1.0e3, abs(float(N_interp)))
    return nominal, "nominal_compression"


def _combined_lateral_mode_shape(res: MemberBucklingResult) -> NDArray[np.float64]:
    n_nodes = res.n_elem + 1
    n_modes_cs = res.n_modes
    n_dof_per_mode = 2 * n_nodes
    w = np.zeros(n_nodes, dtype=np.float64)
    for k in range(n_modes_cs):
        base = k * n_dof_per_mode
        for i in range(n_nodes):
            idx = base + 2 * i
            if idx < len(res.buckling_mode):
                w[i] += float(res.buckling_mode[idx]) ** 2
    return np.sqrt(w)


def _analyze_cross_section_member(
    section: CrossSection,
    *,
    section_loads: SectionLoads,
    member_length_m: float,
    n_cross_section_modes: int,
    n_member_modes: int,
    n_elem: int,
    signature_n_pts: int,
    convergence_elem_counts: list[int],
) -> tuple[dict[str, Any], Any, MemberBucklingResult | None]:
    validate_issues = section.validate()
    N_use, N_source = _effective_axial_force_for_buckling(float(section_loads.N))
    loads_use = SectionLoads(
        N=N_use,
        My=float(section_loads.My),
        Mz=float(section_loads.Mz),
        Vy=float(section_loads.Vy),
        Vz=float(section_loads.Vz),
        T=float(section_loads.T),
    )

    modal = CrossSectionModalAnalysis(section, loads_use).run(n_modes=int(n_cross_section_modes))
    nm = min(int(n_member_modes), len(modal.eigenvalues))

    ana = MemberBucklingAnalysis(
        modal,
        float(member_length_m),
        BoundaryConditions.simply_supported(),
        n_elem=int(n_elem),
        n_modes=nm,
        loads=loads_use,
        section=section,
    )

    err: str | None = None
    member_res: MemberBucklingResult | None = None
    conv: dict[str, Any] | None = None
    sig: dict[str, Any] | None = None

    try:
        member_res = ana.run(n_eigs=min(24, max(10, 2 * nm)))
    except Exception as e:  # pragma: no cover
        err = f"{type(e).__name__}: {e}"

    if member_res is not None:
        try:
            conv = ana.convergence_study(elem_counts=convergence_elem_counts)
        except Exception as e:
            conv = {"error": f"{type(e).__name__}: {e}"}
        try:
            L = float(member_length_m)
            sig = ana.signature_curve(L_min=max(L / 40.0, 1e-4), L_max=max(5.0 * L, 0.05), n_pts=int(signature_n_pts))
        except Exception as e:
            sig = {"error": f"{type(e).__name__}: {e}"}

    out: dict[str, Any] = {
        "validation_issues": validate_issues,
        "section_loads_requested": asdict(section_loads),
        "section_loads_effective": asdict(loads_use),
        "axial_force_source": N_source,
        "member_length_m": float(member_length_m),
        "n_elem": int(n_elem),
        "n_cross_section_modes": int(n_cross_section_modes),
        "n_member_modes": int(nm),
        "error": err,
    }

    if member_res is not None:
        x = np.linspace(0.0, float(member_length_m), member_res.n_elem + 1)
        w_line = _combined_lateral_mode_shape(member_res)
        out["member_buckling"] = {
            "lambda_cr": float(member_res.lambda_cr),
            "eigenvalues": np.asarray(member_res.eigenvalues, dtype=np.float64),
            "n_half_waves_est": int(member_res.n_half_waves()),
            "modal_participation": np.asarray(
                member_res.modal_participation(nm, member_res.n_elem + 1),
                dtype=np.float64,
            ),
            "buckling_mode_x_m": x,
            "buckling_mode_w_combined": w_line,
        }
    if conv is not None:
        out["convergence"] = conv
    if sig is not None:
        out["signature_curve"] = {
            "half_wave_lengths_m": np.asarray(sig.get("half_wave_lengths"), dtype=np.float64),
            "lambda_cr": np.asarray(sig.get("lambda_cr"), dtype=np.float64),
        }
    return out, modal, member_res


def analyze_station_buckling(
    sd: SectionDefinition,
    *,
    section_loads: SectionLoads,
    member_length_m: float,
    n_cross_section_modes: int = 8,
    n_member_modes: int = 6,
    n_elem: int = 16,
    signature_n_pts: int = 18,
    convergence_elem_counts: list[int] | None = None,
    include_per_subcomponent: bool = False,
    section_modes_wireframe_png: Path | None = None,
    member_coupled_section_wireframe_png: Path | None = None,
    part_modes_wireframe_out_dir: Path | None = None,
    part_modes_wireframe_tag: str | None = None,
) -> dict[str, Any]:
    """
    Coupled cross-section + member GBT buckling for one station.

    Uses line-mesh-derived walls (skin/cap/web) when available. Optionally runs isolated
    analyses per subcomponent chain (no strip–strip coupling).
    """
    if convergence_elem_counts is None:
        convergence_elem_counts = [4, 8, 16, 32]

    pairs = wall_definitions_from_line_mesh(sd)
    use_line_mesh = len(pairs) > 0
    section = section_definition_to_gbt_cross_section(sd, use_line_mesh=use_line_mesh)
    wall_source = "line_mesh" if use_line_mesh else "polyline_fallback"

    coupled, modal, member_res = _analyze_cross_section_member(
        section,
        section_loads=section_loads,
        member_length_m=member_length_m,
        n_cross_section_modes=n_cross_section_modes,
        n_member_modes=n_member_modes,
        n_elem=n_elem,
        signature_n_pts=signature_n_pts,
        convergence_elem_counts=convergence_elem_counts,
    )

    wf_paths: list[str] = []

    def _try_wireframe_plots(sec: CrossSection, mod: Any, *, title_suffix: str, part_safe: str | None) -> None:
        try:
            from blade_precompute.section_buckling.interface.plots import (
                plot_cross_section_mode_wireframes,
                plot_member_coupled_section_wireframe_approx,
            )
        except ImportError:
            return
        z_m = float(sd.station_z)
        if section_modes_wireframe_png is not None and part_safe is None:
            plot_cross_section_mode_wireframes(
                sec,
                mod,
                section_modes_wireframe_png,
                station_z_m=z_m,
                n_modes_plot=4,
                title_suffix=title_suffix,
            )
            wf_paths.append(str(section_modes_wireframe_png.resolve()))
        if (
            member_coupled_section_wireframe_png is not None
            and part_safe is None
            and member_res is not None
        ):
            plot_member_coupled_section_wireframe_approx(
                sec,
                mod,
                member_res,
                member_coupled_section_wireframe_png,
                station_z_m=z_m,
            )
            wf_paths.append(str(member_coupled_section_wireframe_png.resolve()))
        if (
            part_modes_wireframe_out_dir is not None
            and part_modes_wireframe_tag is not None
            and part_safe is not None
        ):
            part_sub = part_modes_wireframe_out_dir / part_safe
            part_sub.mkdir(parents=True, exist_ok=True)
            outp = part_sub / "section_modes_wireframe.png"
            plot_cross_section_mode_wireframes(
                sec,
                mod,
                outp,
                station_z_m=z_m,
                n_modes_plot=min(4, max(4, min(n_cross_section_modes, 8))),
                title_suffix=title_suffix,
            )
            wf_paths.append(str(outp.resolve()))

    _try_wireframe_plots(section, modal, title_suffix="coupled", part_safe=None)

    out: dict[str, Any] = {
        "station_z_m": float(sd.station_z),
        "wall_source": wall_source,
        "line_mesh": line_mesh_meta(sd),
        **coupled,
    }

    if include_per_subcomponent and use_line_mesh:
        by_si: dict[int, str] = {}
        for si, name, _, _ in pairs:
            by_si[si] = name
        per: list[dict[str, Any]] = []
        for si in sorted(by_si.keys()):
            name = by_si[si]
            role = _thickness_role_from_subcomponent_name(name)
            safe = safe_subcomponent_filename_label(str(name))
            part: dict[str, Any] = {
                "subcomponent_index": int(si),
                "name": name,
                "role": role,
            }
            try:
                sec_i = cross_section_for_subcomponent_indices(sd, {si})
                part_dict, modal_p, _mr = _analyze_cross_section_member(
                    sec_i,
                    section_loads=section_loads,
                    member_length_m=member_length_m,
                    n_cross_section_modes=max(4, min(n_cross_section_modes, 8)),
                    n_member_modes=max(3, min(n_member_modes, 6)),
                    n_elem=n_elem,
                    signature_n_pts=max(5, signature_n_pts // 2),
                    convergence_elem_counts=convergence_elem_counts,
                )
                part["analysis"] = part_dict
                _try_wireframe_plots(sec_i, modal_p, title_suffix=str(name), part_safe=safe)
            except Exception as e:  # pragma: no cover
                part["analysis"] = {"error": f"{type(e).__name__}: {e}"}
            per.append(part)
        out["per_subcomponent"] = per
        out["per_subcomponent_note"] = (
            "Isolated strip systems: no coupling between skin/cap/web. "
            "Loads are identical section resultants (not load apportioned per part)."
        )

    if wf_paths:
        out["_wireframe_png_paths"] = wf_paths

    return out


def safe_subcomponent_filename_label(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("_")
    return (s[:56] if s else "part")
