from __future__ import annotations

import argparse
import sys
from pathlib import Path

_p = Path(__file__).resolve()
while _p.name != "blade_precompute" and _p.parent != _p:
    _p = _p.parent
if _p.name == "blade_precompute":
    sys.path.insert(0, str(_p.parent))

from blade_precompute.global_beam_model.examples.plot_common import (
    DEFAULT_BLADE_SPEC,
    DEFAULT_PDF_PATH,
    DEFAULT_PNG_DIR,
    solve_beam_examples_case,
)
from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import print_convergence_summary
from blade_precompute.orchestration.precompute.vis import write_beam_model_pngs


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run the global_beam_model smoke case and save plots to PDF. "
            "Default fixture is synthetic_tapered_blade; use --fixture gbt for blade-spec stiffness. "
            "Optional --png-dir writes the same PNG pack as precompute (default: examples/output/beam_diagnostic_pngs)."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Output PDF path (default: examples/output/beam_smoke.pdf).",
    )
    p.add_argument(
        "--png-dir",
        type=Path,
        nargs="?",
        const=DEFAULT_PNG_DIR,
        default=None,
        help=(
            "If set, also write PNG diagnostics to this directory. "
            "Use bare --png-dir to use the default examples/output/beam_diagnostic_pngs."
        ),
    )
    p.add_argument(
        "--fixture",
        choices=("synthetic", "gbt"),
        default="synthetic",
        help="synthetic: tapered K6/K7 from synthetic_tapered_blade.py; gbt: tabulated K from blade JSON.",
    )
    p.add_argument(
        "--blade-spec",
        type=Path,
        default=DEFAULT_BLADE_SPEC,
        help="Blade JSON for --fixture gbt.",
    )
    p.add_argument("--n-nodes", type=int, default=17, help="Beam nodes (gbt only).")
    p.add_argument("--max-iter", type=int, default=110)
    p.add_argument("--n-load-steps", type=int, default=72)
    p.add_argument("--span-plot-samples", type=int, default=400)
    args = p.parse_args()

    model, loads, res, _ = solve_beam_examples_case(
        args.fixture,
        blade_spec=args.blade_spec,
        n_nodes=int(args.n_nodes),
        load_vy=350.0,
        max_iter=int(args.max_iter),
        n_load_steps=int(args.n_load_steps),
        verbose=False,
    )
    print_convergence_summary(res, model=model)

    if args.png_dir is not None:
        png_dir = Path(args.png_dir).resolve()
        png_dir.mkdir(parents=True, exist_ok=True)
        paths = write_beam_model_pngs(
            png_dir,
            model,
            res,
            loads,
            span_plot_samples=int(args.span_plot_samples),
        )
        print(f"Wrote {len(paths)} PNG(s) under {png_dir}")

    from blade_precompute.global_beam_model.interface import plot as bmplot

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figs = []
    for fn in [
        bmplot.plot_centerline_ref_def,
        bmplot.plot_spanwise_resultants,
        bmplot.plot_spanwise_strains,
        bmplot.plot_nodal_warping,
        bmplot.plot_iteration_history,
        bmplot.plot_reactions,
        bmplot.plot_distributed_loads,
    ]:
        if (
            fn is bmplot.plot_centerline_ref_def
            or fn is bmplot.plot_nodal_warping
            or fn is bmplot.plot_distributed_loads
        ):
            fig, _ = fn(model, res) if fn is not bmplot.plot_distributed_loads else fn(model, loads)
        else:
            fig, _ = fn(res)
        figs.append(fig)

    outp = args.out.resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(outp) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)

    print(f"Saved: {outp}  (fixture={args.fixture}, converged={res.converged}, n_iter={res.n_iterations})")


if __name__ == "__main__":
    main()
