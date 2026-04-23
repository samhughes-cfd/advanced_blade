"""
Sinusoidal beam resultant history → fused recovery (``L_rec`` / ``L_iso``) → rainflow + Miner.

This uses **midsurface section homogenisation** and :class:`blade_utilities.recovery.RecoveryCache`
(see ``blade_precompute`` section solvers + tensor cache). It is **not** the educational thin-wall
``examples/section_stress_model`` NACA + ``run_section`` path.

From repository root::

    python examples/section_fatigue_sinusoid/run_example.py

    python examples/section_fatigue_sinusoid/run_example.py --f-hz 3.0 --amplitude 8e3 --load My
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
# Repo root (for `import blade_analysis` / `import blade_precompute` without install).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import numpy as np

from blade_analysis.fatigue_damage import FatigueAnalysis, SNcurve
from blade_analysis.fatigue_damage._smoke_fixtures import (
    build_smoke_recovery_cache_and_ref_section,
    default_fatigue_sn_curves,
    smoke_sinusoidal_resultant_history,
)
from blade_precompute.section_properties.engine.geometry import SectionDefinition
from blade_utilities.recovery import RecoveryCache, load_cache
from lib.section_maps import save_fatigue_damage_section_map, save_fatigue_life_section_map

_LOADS = ("N", "Vy", "Vz", "My", "Mz", "T", "B")


def _summarise(fr: object) -> None:
    print("FatigueResult:")
    print(f"  max_damage_composite: {getattr(fr, 'max_damage_composite'):.6g}")
    print(f"  max_damage_isotropic: {getattr(fr, 'max_damage_isotropic'):.6g}")
    print(f"  worst_composite (station, sub, ply): {getattr(fr, 'worst_composite')!r}")
    print(f"  worst_isotropic (station, name): {getattr(fr, 'worst_isotropic')!r}")
    print(f"  fatigue_critical_material: {getattr(fr, 'fatigue_critical_material')!r}")
    print(f"  stress_component_used: {getattr(fr, 'stress_component_used')}")
    print(f"  design_life_years: {getattr(fr, 'design_life_years')}")
    print(f"  memory_mode: {getattr(fr, 'memory_mode')!r}")


def _save_plots(
    res: object,
    z: np.ndarray,
    out_dir: Path,
    *,
    cache: RecoveryCache,
    ref_section: SectionDefinition | None,
    section_station: int | None,
    load_label: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blade_analysis.fatigue_damage.interface import plot as fplot

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, call in [
        (
            "damage_life_vs_span",
            lambda: fplot.plot_damage_life_vs_span(res, z),
        ),
        (
            "static_fi_vs_span",
            lambda: fplot.plot_static_fi_vs_span(res, z),
        ),
    ]:
        fig, _ = call()
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(p)

    bins = getattr(res, "rainflow_bins", None)
    worst_c = getattr(res, "worst_composite")
    worst_i = getattr(res, "worst_isotropic")
    s_c = int(worst_c[0])
    name_c = str(worst_c[1])
    ply_c = int(worst_c[2])
    sub_c = list(cache.composite_subcomp_names).index(name_c)
    s_i = int(worst_i[0])
    name_i = str(worst_i[1])
    sub_i = list(cache.isotropic_subcomp_names).index(name_i)

    if bins is not None:
        fig, _ = fplot.plot_rainflow_composite(
            bins,
            station=s_c,
            subcomp=sub_c,
            ply=ply_c,
            title=f"Rainflow (composite) worst: s={s_c} sub={name_c} ply={ply_c}",
        )
        p = out_dir / "rainflow_composite_worst.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(p)
        if bins.counts_iso.size and float(np.sum(bins.counts_iso)) > 0.0:
            fig, _ = fplot.plot_rainflow_isotropic(
                bins,
                station=s_i,
                subcomp=sub_i,
                title=f"Rainflow (isotropic VM) worst: s={s_i} sub={name_i}",
            )
            p2 = out_dir / "rainflow_isotropic_worst.png"
            fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved.append(p2)
        rc = np.asarray(bins.ranges_comp[:, s_c, sub_c, ply_c], dtype=np.float64)
        fig, _ = fplot.plot_sn_curve_with_ranges(
            SNcurve.gfrp_blade(),
            rc,
            title=f"S–N GFRP vs rainflow bin centres (worst ply: s={s_c} {name_c} ply={ply_c})",
        )
        p3 = out_dir / "sn_gfrp_with_ranges.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(p3)

    if ref_section is not None:
        n_s = int(z.size)
        s_map = int(section_station) if section_station is not None else int(worst_c[0])
        if not (0 <= s_map < n_s):
            raise ValueError(f"--section-station {s_map} out of range [0, {n_s - 1}]")
        sub_title = f"load={load_label}"
        saved.append(
            save_fatigue_damage_section_map(
                out_dir / "fatigue_damage_section_map.png",
                cache,
                ref_section,
                res,
                z,
                s_map,
                subtitle=sub_title,
            )
        )
        saved.append(
            save_fatigue_life_section_map(
                out_dir / "fatigue_life_section_map.png",
                cache,
                ref_section,
                res,
                z,
                s_map,
                subtitle=sub_title,
            )
        )
    else:
        print(
            "Skipping section-plane fatigue maps: no SectionDefinition (use default inline cache, not --cache .npz)."
        )

    return saved


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sinusoidal section load history → recovery operators → fatigue (Miner / S–N).",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional precomputed recovery cache .npz; default builds compact inline smoke model.",
    )
    p.add_argument("--n-t", type=int, default=256, help="Time samples (default 256).")
    p.add_argument("--t-end", type=float, default=1.0, help="Time window [s] (default 1).")
    p.add_argument("--f-hz", type=float, default=7.0, help="Load sinusoid frequency [Hz].")
    p.add_argument(
        "--amplitude",
        type=float,
        default=5.0e3,
        help="Sinusoid amplitude in chosen resultant units (e.g. N for N, N m for M).",
    )
    p.add_argument(
        "--load",
        type=str,
        default="My",
        choices=_LOADS,
        help="Which of the seven resultants carries the sinusoid (others zero).",
    )
    p.add_argument(
        "--no-spanwise-envelope",
        action="store_true",
        help="Use uniform amplitude at all span stations (else ramp 0.5 at root to 1.0 at tip).",
    )
    p.add_argument("--design-life-years", type=float, default=25.0, dest="design_life")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic PNGs (default: .../section_fatigue_sinusoid/outputs).",
    )
    p.add_argument(
        "--section-station",
        type=int,
        default=None,
        help="Span station index for section-plane damage/life maps (default: worst composite station).",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip writing matplotlib PNGs.")
    p.add_argument("--show", action="store_true", help="Open figures interactively (not with --no-plots).")
    args = p.parse_args()

    ref_section: SectionDefinition | None = None
    if args.cache is not None:
        cache = load_cache(args.cache)
    else:
        print("Building compact midsurface model + RecoveryCache (may take a few seconds)...")
        cache, ref_section = build_smoke_recovery_cache_and_ref_section()

    z = np.asarray(cache.z_stations, dtype=np.float64)
    hist = smoke_sinusoidal_resultant_history(
        z,
        n_t=args.n_t,
        t_end=args.t_end,
        f_hz=args.f_hz,
        amplitude=args.amplitude,
        load_component=args.load,  # type: ignore[arg-type]
        spanwise_envelope=not args.no_spanwise_envelope,
    )
    sn = default_fatigue_sn_curves()
    analysis = FatigueAnalysis.from_cache(
        cache,
        sn,
        design_life_years=float(args.design_life),
    )
    res = analysis.run(hist, memory_limit_mb=512.0)
    _summarise(res)

    if args.no_plots and args.show:
        p.error("--show requires plots (remove --no-plots).")
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "outputs"
    load_label = f"{args.load} amp={args.amplitude:g} f={args.f_hz:g} Hz"
    if not args.no_plots:
        if args.show:
            import matplotlib.pyplot as plt
            from blade_analysis.fatigue_damage.interface import plot as fplot

            fig, _ = fplot.plot_damage_life_vs_span(res, z)
            fig2, _ = fplot.plot_static_fi_vs_span(res, z)
            print("Close plot windows to exit.")
            plt.show()
        else:
            paths = _save_plots(
                res,
                z,
                out_dir,
                cache=cache,
                ref_section=ref_section,
                section_station=args.section_station,
                load_label=load_label,
            )
            print(f"Wrote {len(paths)} figure(s) under {out_dir.resolve()}")
            for q in paths:
                print(f"  {q}")


if __name__ == "__main__":
    main()
