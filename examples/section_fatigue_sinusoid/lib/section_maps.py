"""
Miner damage and life on midsurface strip polylines (section plane y, z).

Scalars are **subcomponent / strip homogenised** (max over plies for composites), not
element-resolved like MITC4 heatmaps in ``section_shell_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from blade_analysis.fatigue_damage.core.types import FatigueResult
from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_utilities.recovery import RecoveryCache

_Field = Literal["damage", "life"]


def _chord_from_ref_section(ref_section: SectionDefinition) -> float:
    """Physical chord [m] from skin span (matches :class:`SectionBuilder` twist=0 scaling)."""
    span = 0.0
    for s in ref_section.subcomponents:
        if s.name == "skin":
            y = np.asarray(s.midsurface_coords[:, 0], dtype=np.float64)
            span = max(span, float(np.ptp(y)))
    if span > 0.0:
        return span
    for s in ref_section.subcomponents:
        y = np.asarray(s.midsurface_coords[:, 0], dtype=np.float64)
        span = max(span, float(np.ptp(y)))
    return span if span > 0.0 else 1.0


def _airfoil_outline_physical(chord: float, af: AirfoilSDF) -> NDArray[np.float64]:
    """Section-plane (y,z) [m]: chordwise centred on mid-chord, same convention as midsurface box."""
    v = np.asarray(af.vertices, dtype=np.float64)
    return np.column_stack([(v[:, 0] - 0.5) * chord, v[:, 1] * chord])


def _physical_vertical_at_y_norm(af: AirfoilSDF, chord: float, y_norm: float) -> NDArray[np.float64]:
    xn = float(np.clip(y_norm + 0.5, 1e-6, 1.0 - 1e-6))
    upper = af.upper_surface()
    lower = af.lower_surface()
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    zu = float(np.interp(xn, upper[:, 0], upper[:, 1]))
    zl = float(np.interp(xn, lower[:, 0], lower[:, 1]))
    if zl > zu:
        zl, zu = zu, zl
    return np.array(
        [[y_norm * chord, zl * chord], [y_norm * chord, zu * chord]],
        dtype=np.float64,
    )


def _draw_naca_underlay(ax: object, outline: NDArray[np.float64], web_curves: list[NDArray[np.float64]]) -> None:
    """Cyan airfoil outline + dashed webs (``section_shell_model`` CLPT style)."""
    if outline.shape[0] >= 2:
        closed = np.vstack([outline, outline[:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#5dade2",
            lw=1.1,
            alpha=0.55,
            zorder=1,
        )
    for w in web_curves:
        if w.shape[0] != 2:
            continue
        ax.plot(
            w[:, 0],
            w[:, 1],
            color="#ecf0f1",
            lw=1.35,
            ls="--",
            alpha=0.65,
            zorder=2,
        )


def _axis_limits_from_section(
    ref_section: SectionDefinition,
    segs: NDArray[np.float64],
    outline: NDArray[np.float64] | None = None,
    pad_frac: float = 0.06,
) -> tuple[float, float, float, float]:
    pts: list[NDArray[np.float64]] = []
    for sub in ref_section.subcomponents:
        pts.append(np.asarray(sub.midsurface_coords, dtype=np.float64))
    if segs.size:
        pts.append(segs.reshape(-1, 2))
    if outline is not None and outline.size:
        pts.append(np.asarray(outline, dtype=np.float64))
    all_xy = np.vstack(pts) if pts else np.zeros((1, 2), dtype=np.float64)
    wx = float(np.ptp(all_xy[:, 0]))
    wy = float(np.ptp(all_xy[:, 1]))
    pad = max(pad_frac * max(wx, wy, 1e-6), 1e-6)
    return (
        float(all_xy[:, 0].min()) - pad,
        float(all_xy[:, 0].max()) + pad,
        float(all_xy[:, 1].min()) - pad,
        float(all_xy[:, 1].max()) + pad,
    )


def _row_composite(cache: RecoveryCache, name: str) -> int | None:
    names = list(cache.composite_subcomp_names)
    return names.index(name) if name in names else None


def _row_isotropic(cache: RecoveryCache, name: str) -> int | None:
    names = list(cache.isotropic_subcomp_names)
    return names.index(name) if name in names else None


def scalar_damage_for_subcomponent(
    cache: RecoveryCache,
    result: FatigueResult,
    station: int,
    sub: SubcomponentGeometry,
) -> float | None:
    """Max Miner damage over active plies (composite) or isotropic row damage."""
    if sub.is_composite:
        row = _row_composite(cache, sub.name)
        if row is None:
            return None
        n_ply = int(cache.ply_count[station, row])
        if n_ply <= 0:
            return None
        return float(np.max(result.damage_composite[station, row, :n_ply]))
    row_i = _row_isotropic(cache, sub.name)
    if row_i is None:
        return None
    return float(result.damage_isotropic[station, row_i])


def scalar_life_for_subcomponent(
    cache: RecoveryCache,
    result: FatigueResult,
    station: int,
    sub: SubcomponentGeometry,
) -> float | None:
    """Min life [yr] over active plies (composite) or isotropic life (finite positive only)."""
    if sub.is_composite:
        row = _row_composite(cache, sub.name)
        if row is None:
            return None
        n_ply = int(cache.ply_count[station, row])
        if n_ply <= 0:
            return None
        block = np.asarray(result.life_composite[station, row, :n_ply], dtype=np.float64)
        good = np.isfinite(block) & (block > 0.0)
        if not np.any(good):
            return None
        return float(np.min(block[good]))
    row_i = _row_isotropic(cache, sub.name)
    if row_i is None:
        return None
    v = float(result.life_isotropic[station, row_i])
    if not np.isfinite(v) or v <= 0.0:
        return None
    return v


def _collect_airfoil_style_segments(
    ref_section: SectionDefinition,
    cache: RecoveryCache,
    result: FatigueResult,
    station: int,
    field: _Field,
    *,
    naca_code: str = "2412",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """
    Map fatigue scalars onto a **NACA 4-digit schematic** on the section chord.

    The midsurface **solve** still uses the optimisation box strips; this layout is
    presentation-only so figures read like ``clpt_fi_on_section_geometry.png``.
    """
    chord = _chord_from_ref_section(ref_section)
    af = AirfoilSDF.from_naca(naca_code, n_points=120, chord=1.0)
    outline = _airfoil_outline_physical(chord, af)

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    values: list[float] = []
    web_curves: list[NDArray[np.float64]] = []

    for sub in ref_section.subcomponents:
        if field == "damage":
            v = scalar_damage_for_subcomponent(cache, result, station, sub)
        else:
            v = scalar_life_for_subcomponent(cache, result, station, sub)
        if v is None:
            continue
        name = sub.name
        if name == "skin":
            for k in range(int(outline.shape[0]) - 1):
                segments.append(
                    (
                        (float(outline[k, 0]), float(outline[k, 1])),
                        (float(outline[k + 1, 0]), float(outline[k + 1, 1])),
                    )
                )
                values.append(float(v))
            segments.append(
                (
                    (float(outline[-1, 0]), float(outline[-1, 1])),
                    (float(outline[0, 0]), float(outline[0, 1])),
                )
            )
            values.append(float(v))
        elif name in ("cap_ps", "web"):
            pts = np.asarray(sub.midsurface_coords, dtype=np.float64)
            if pts.shape[0] != 2:
                continue
            y_norm = float(pts[0, 0]) / max(chord, 1e-12)
            seg2 = _physical_vertical_at_y_norm(af, chord, y_norm)
            web_curves.append(seg2)
            segments.append(
                ((float(seg2[0, 0]), float(seg2[0, 1])), (float(seg2[1, 0]), float(seg2[1, 1])))
            )
            values.append(float(v))
        elif name == "leading_edge_insert":
            i0 = int(np.argmin(outline[:, 0]))
            i1 = min(i0 + 6, int(outline.shape[0]) - 1)
            p0, p1 = outline[i0], outline[i1]
            segments.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))))
            values.append(float(v))

    if not segments:
        z = np.zeros((0, 2, 2), dtype=np.float64)
        return z, np.zeros((0,), dtype=np.float64), outline, web_curves
    segs = np.asarray(segments, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    return segs, vals, outline, web_curves


def save_fatigue_damage_section_map(
    outfile: Path | str,
    cache: RecoveryCache,
    ref_section: SectionDefinition,
    result: FatigueResult,
    z_stations: NDArray[np.float64],
    station: int,
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
    cmap: str = "turbo",
    subtitle: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    if not (0 <= station < z.size):
        raise ValueError(f"station index {station} out of range for z_stations (size {z.size}).")

    segs, vals, outline, web_curves = _collect_airfoil_style_segments(
        ref_section, cache, result, station, "damage"
    )
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#1a1d2e")

    if segs.size == 0 or vals.size == 0:
        ax.set_title("No midsurface segments to plot.", color="#e0e0e0", fontsize=10)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        return outfile

    _draw_naca_underlay(ax, outline, web_curves)

    vmax_dat = float(np.nanmax(vals))
    if vmax_dat > 0.0 and vmax_dat < 1.0:
        v_hi = vmax_dat
    else:
        v_hi = max(1.0, vmax_dat) if vmax_dat > 0.0 else 1.0
    v_hi = max(v_hi, 1e-30)
    norm = mcolors.Normalize(vmin=0.0, vmax=v_hi)

    lc = LineCollection(
        segs,
        array=vals,
        cmap=cmap,
        norm=norm,
        linewidths=7.0,
        capstyle="round",
        joinstyle="round",
        zorder=4,
        antialiased=True,
    )
    lc.set_clip_on(False)
    ax.add_collection(lc)
    mid = 0.5 * (segs[:, 0, :] + segs[:, 1, :])
    ax.scatter(
        mid[:, 0],
        mid[:, 1],
        c=vals,
        cmap=cmap,
        norm=norm,
        s=64,
        zorder=5,
        edgecolors="#f8f8f8",
        linewidths=0.75,
        clip_on=False,
    )

    crit = vals >= 1.0
    if np.any(crit):
        segs_crit = segs[crit]
        lc_crit = LineCollection(
            segs_crit,
            colors="#e74c3c",
            linewidths=11.0,
            capstyle="round",
            joinstyle="round",
            zorder=7,
            alpha=0.95,
            label="Miner damage ≥ 1",
        )
        ax.add_collection(lc_crit)

    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(vals)
    cbar = fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Miner damage (max over plies)", color="#e0e0e0", fontsize=9)
    cbar.ax.tick_params(colors="#aaaaaa", labelsize=7)

    x0, x1, y0, y1 = _axis_limits_from_section(ref_section, segs, outline)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m] (chordwise)", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")

    z_here = float(z[station])
    line2 = f"station={station}, z={z_here:.3g} m ({subtitle})" if subtitle else f"station={station}, z={z_here:.3g} m"
    title = (
        "Fatigue damage — NACA 2412 schematic on chord (metrics from midsurface strips)\n" + line2
    )
    ax.set_title(title, color="#e0e0e0", fontsize=10)
    if np.any(crit):
        ax.legend(loc="upper right", fontsize=7, framealpha=0.35)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_fatigue_life_section_map(
    outfile: Path | str,
    cache: RecoveryCache,
    ref_section: SectionDefinition,
    result: FatigueResult,
    z_stations: NDArray[np.float64],
    station: int,
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
    cmap: str = "plasma",
    subtitle: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    if not (0 <= station < z.size):
        raise ValueError(f"station index {station} out of range for z_stations (size {z.size}).")

    segs, vals, outline, web_curves = _collect_airfoil_style_segments(
        ref_section, cache, result, station, "life"
    )
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#1a1d2e")

    if segs.size == 0 or vals.size == 0:
        ax.set_title("No finite life values to plot on section.", color="#e0e0e0", fontsize=10)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        return outfile

    v_pos = vals[np.isfinite(vals) & (vals > 0.0)]
    if v_pos.size == 0:
        ax.set_title("No finite life values to plot on section.", color="#e0e0e0", fontsize=10)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        return outfile

    _draw_naca_underlay(ax, outline, web_curves)

    vmin = max(float(np.min(v_pos)), 1e-6)
    vmax = max(float(np.max(v_pos)), vmin * 10.0)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    lc = LineCollection(
        segs,
        array=vals,
        cmap=cmap,
        norm=norm,
        linewidths=7.0,
        capstyle="round",
        joinstyle="round",
        zorder=4,
        antialiased=True,
    )
    lc.set_clip_on(False)
    ax.add_collection(lc)
    mid = 0.5 * (segs[:, 0, :] + segs[:, 1, :])
    ax.scatter(
        mid[:, 0],
        mid[:, 1],
        c=vals,
        cmap=cmap,
        norm=norm,
        s=64,
        zorder=5,
        edgecolors="#f8f8f8",
        linewidths=0.75,
        clip_on=False,
    )

    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(vals)
    cbar = fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("life [yr] (min over plies)", color="#e0e0e0", fontsize=9)
    cbar.ax.tick_params(colors="#aaaaaa", labelsize=7)

    xa0, xa1, ya0, ya1 = _axis_limits_from_section(ref_section, segs, outline)
    ax.set_xlim(xa0, xa1)
    ax.set_ylim(ya0, ya1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m] (chordwise)", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")

    z_here = float(z[station])
    dl = float(result.design_life_years)
    line2 = f"station={station}, z={z_here:.3g} m ({subtitle})" if subtitle else f"station={station}, z={z_here:.3g} m"
    title = f"Fatigue life — NACA 2412 schematic on chord (design life = {dl:.3g} yr)\n" + line2
    ax.set_title(title, color="#e0e0e0", fontsize=10)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile
