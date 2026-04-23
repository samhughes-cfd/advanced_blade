"""CLI: stress history → rainflow → Miner; canonical :class:`~blade_analysis.fatigue_damage.core.types.FatigueResult`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blade_analysis.fatigue_damage import FatigueAnalysis, SNcurve
from blade_analysis.fatigue_damage._smoke_fixtures import (
    build_smoke_recovery_cache,
    default_fatigue_sn_curves,
    smoke_sinusoidal_resultant_history,
)
from blade_utilities.recovery import load_cache


def _summarise(fr: object) -> None:
    print("FatigueResult:")
    print(f"  max_damage_composite: {getattr(fr, 'max_damage_composite'):.6g}")
    print(f"  max_damage_isotropic: {getattr(fr, 'max_damage_isotropic'):.6g}")
    print(f"  fatigue_critical_material: {getattr(fr, 'fatigue_critical_material')!r}")
    print(f"  design_life_years: {getattr(fr, 'design_life_years')}")
    print(f"  memory_mode: {getattr(fr, 'memory_mode')!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run fatigue pipeline smoke (canonical: FatigueResult).")
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional recovery cache .npz; default matches recovery_cache built-in smoke.",
    )
    p.add_argument("--plot", action="store_true", help="Show matplotlib figures (requires matplotlib).")
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Save fatigue plots to a multi-page PDF at this path instead of showing.",
    )
    args = p.parse_args()
    if args.cache is not None:
        cache = load_cache(args.cache)
    else:
        cache = build_smoke_recovery_cache()
    hist = smoke_sinusoidal_resultant_history(np.asarray(cache.z_stations, dtype=np.float64))
    analysis = FatigueAnalysis.from_cache(cache, default_fatigue_sn_curves(), design_life_years=25.0)
    res = analysis.run(hist, memory_limit_mb=512.0)
    _summarise(res)

    if args.plot or args.plot_out is not None:
        from blade_analysis.fatigue_damage.interface import plot as fplot

        z = np.asarray(cache.z_stations, dtype=np.float64)
        figs = []
        fig, _ = fplot.plot_damage_life_vs_span(res, z)
        figs.append(fig)
        fig, _ = fplot.plot_static_fi_vs_span(res, z)
        figs.append(fig)
        bins = res.rainflow_bins
        if bins is not None:
            fig, _ = fplot.plot_rainflow_composite(bins, station=0, subcomp=0, ply=0)
            figs.append(fig)
            if bins.counts_iso.size and float(np.sum(bins.counts_iso)) > 0.0:
                fig, _ = fplot.plot_rainflow_isotropic(bins, station=0, subcomp=0)
                figs.append(fig)
            rc = np.asarray(bins.ranges_comp[:, 0, 0, 0], dtype=np.float64)
            fig, _ = fplot.plot_sn_curve_with_ranges(SNcurve.gfrp_blade(), rc)
            figs.append(fig)
        if args.plot_out is not None:
            outp = args.plot_out.resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(outp) as pdf:
                for f in figs:
                    pdf.savefig(f, bbox_inches="tight")
                    plt.close(f)
            print(f"Saved fatigue plots to {outp}")
        else:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
