"""Plot ``data_library`` columnar ``.dat`` files for QA (requires matplotlib)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .plot_inputs import (
    plot_blade_spanwise_dat,
    plot_extreme_load_distribution_dat,
    plot_operational_load_heatmap,
    plot_operational_timeseries_dat,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot data_library .dat files.")
    p.add_argument(
        "kind",
        choices=("spanwise", "extreme", "operational", "heatmap"),
        help="Which plot routine to run.",
    )
    p.add_argument("path", type=Path, help="Path to .dat file.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save figure to this path instead of showing.",
    )
    p.add_argument("--r-z", type=float, default=0.0, dest="r_z", help="Target r_z for operational plot.")
    p.add_argument(
        "--value-col",
        type=str,
        default="q_y_Npm",
        help="Column for heatmap (operational long-format).",
    )
    args = p.parse_args()
    import matplotlib.pyplot as plt

    if args.kind == "spanwise":
        fig, _ = plot_blade_spanwise_dat(args.path)
    elif args.kind == "extreme":
        fig, _ = plot_extreme_load_distribution_dat(args.path)
    elif args.kind == "operational":
        fig, _ = plot_operational_timeseries_dat(args.path, r_z_target=args.r_z)
    else:
        fig, _ = plot_operational_load_heatmap(args.path, value_col=args.value_col)

    if args.out is not None:
        outp = args.out.resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {outp}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
