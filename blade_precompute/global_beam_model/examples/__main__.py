"""Run example beam plots from the package (PNG set and/or PDF).

Examples
--------
From the repository root::

    python -m blade_precompute.global_beam_model.examples
    python -m blade_precompute.global_beam_model.examples --pdf --png
    python -m blade_precompute.global_beam_model.examples --fixture gbt --png-dir ./my_pngs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from blade_precompute.global_beam_model.examples.plot_common import (
    DEFAULT_PDF_PATH,
    DEFAULT_PNG_DIR,
    DEFAULT_BLADE_SPEC,
    solve_beam_examples_case,
)
from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import print_convergence_summary
from blade_precompute.orchestration.precompute.vis import write_beam_model_pngs


def _write_pdf(model, res, loads, out_pdf: Path) -> None:
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

    out_pdf = out_pdf.resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate beam example plots under global_beam_model/examples (PNG pack and/or PDF)."
    )
    p.add_argument(
        "--pdf",
        action="store_true",
        help=f"Write multi-page PDF (default path: {DEFAULT_PDF_PATH}).",
    )
    p.add_argument(
        "--png",
        action="store_true",
        help=f"Write PNG pack like precompute (default dir: {DEFAULT_PNG_DIR}).",
    )
    p.add_argument(
        "--pdf-out",
        type=Path,
        default=None,
        help="PDF output path (default: examples/output/beam_smoke.pdf).",
    )
    p.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="PNG output directory (default: examples/output/beam_diagnostic_pngs).",
    )
    p.add_argument(
        "--fixture",
        choices=("synthetic", "gbt"),
        default="synthetic",
        help="Beam stiffness fixture.",
    )
    p.add_argument(
        "--blade-spec",
        type=Path,
        default=None,
        help=f"Blade JSON for --fixture gbt (default: {DEFAULT_BLADE_SPEC}).",
    )
    p.add_argument("--n-nodes", type=int, default=17, help="Beam nodes (gbt only).")
    p.add_argument("--load-vy", type=float, default=350.0, help="Uniform q_y [N/m].")
    p.add_argument("--max-iter", type=int, default=110)
    p.add_argument("--n-load-steps", type=int, default=72)
    p.add_argument("--span-plot-samples", type=int, default=400)
    args = p.parse_args()

    do_pdf = bool(args.pdf)
    do_png = bool(args.png)
    if not do_pdf and not do_png:
        do_png = True

    blade_spec = Path(args.blade_spec) if args.blade_spec is not None else DEFAULT_BLADE_SPEC

    model, loads, res, _ = solve_beam_examples_case(
        args.fixture,
        blade_spec=blade_spec,
        n_nodes=int(args.n_nodes),
        load_vy=float(args.load_vy),
        max_iter=int(args.max_iter),
        n_load_steps=int(args.n_load_steps),
        verbose=False,
    )
    print_convergence_summary(res, model=model)

    if do_png:
        png_dir = Path(args.png_dir) if args.png_dir is not None else DEFAULT_PNG_DIR
        png_dir = png_dir.resolve()
        png_dir.mkdir(parents=True, exist_ok=True)
        paths = write_beam_model_pngs(
            png_dir,
            model,
            res,
            loads,
            span_plot_samples=int(args.span_plot_samples),
        )
        print(f"Wrote {len(paths)} PNG(s) under {png_dir}")

    if do_pdf:
        pdf_out = Path(args.pdf_out) if args.pdf_out is not None else DEFAULT_PDF_PATH
        _write_pdf(model, res, loads, pdf_out)
        print(f"Wrote PDF {pdf_out.resolve()}")

    print(f"Done (fixture={args.fixture}, converged={res.converged}, n_iter={res.n_iterations}).")


if __name__ == "__main__":
    main()
