"""Run all global_beam_model example fixtures with plots + convergence debug artifacts.

From the repository root::

    python -m blade_precompute.global_beam_model.examples.debug_all
    python -m blade_precompute.global_beam_model.examples.debug_all --out-dir ./outputs/beam_debug_run
    python -m blade_precompute.global_beam_model.examples.debug_all --verbose

Each fixture gets a subdirectory (``synthetic/``, ``gbt/``) with PNG pack, PDF, and
``convergence_debug.{json,txt}``. Console prints ``SolverOptions.verbose`` NR lines when
``--verbose`` is set.

GBT requires a blade JSON: ``--blade-spec`` or the first ``example_blade*.json`` under
the repo root; otherwise GBT is skipped.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_p = Path(__file__).resolve()
while _p.name != "blade_precompute" and _p.parent != _p:
    _p = _p.parent
if _p.name == "blade_precompute":
    sys.path.insert(0, str(_p.parent))

from blade_precompute.global_beam_model.examples.convergence_debug import write_convergence_artifacts
from blade_precompute.global_beam_model.examples.plot_common import DEFAULT_BLADE_SPEC, solve_beam_examples_case
from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import print_convergence_summary
from blade_precompute.orchestration.precompute.vis import write_beam_model_pngs


def _repo_root() -> Path:
    # .../blade_precompute/global_beam_model/examples/debug_all.py -> repo root
    return Path(__file__).resolve().parents[2]


def _find_blade_json(preferred: Path | None) -> Path | None:
    if preferred is not None and Path(preferred).is_file():
        return Path(preferred).resolve()
    if DEFAULT_BLADE_SPEC.is_file():
        return DEFAULT_BLADE_SPEC.resolve()
    root = _repo_root()
    matches = sorted(root.glob("example_blade*.json"))
    if matches:
        return matches[0].resolve()
    return None


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

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf.resolve()) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Base output directory (default: examples/output/debug_all_<UTC timestamp>).",
    )
    p.add_argument("--blade-spec", type=Path, default=None, help="Blade JSON for GBT fixture.")
    p.add_argument("--n-nodes", type=int, default=17)
    p.add_argument("--load-vy", type=float, default=350.0)
    p.add_argument("--max-iter", type=int, default=110)
    p.add_argument("--n-load-steps", type=int, default=72)
    p.add_argument("--span-plot-samples", type=int, default=400)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Per load-step / NR diagnostics from the beam solver (noisy).",
    )
    p.add_argument(
        "--skip-gbt",
        action="store_true",
        help="Only run the synthetic tapered fixture.",
    )
    args = p.parse_args()

    ex_dir = Path(__file__).resolve().parent
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = ex_dir / "output" / f"debug_all_{stamp}"
    else:
        base = Path(args.out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    blade = None if args.skip_gbt else _find_blade_json(args.blade_spec)
    if not args.skip_gbt and blade is None:
        print("GBT: skipped (no example_blade*.json found; use --blade-spec PATH).")

    fixtures: list[tuple[str, Path | None]] = [("synthetic", None)]
    if not args.skip_gbt and blade is not None:
        fixtures.append(("gbt", blade))

    for name, spec in fixtures:
        sub = base / name
        sub.mkdir(parents=True, exist_ok=True)
        print(f"\n=== fixture={name} out={sub} ===")
        model, loads, res, opts = solve_beam_examples_case(
            name,  # type: ignore[arg-type]
            blade_spec=spec,
            n_nodes=int(args.n_nodes),
            load_vy=float(args.load_vy),
            max_iter=int(args.max_iter),
            n_load_steps=int(args.n_load_steps),
            verbose=bool(args.verbose),
        )
        print_convergence_summary(res, model=model)

        write_convergence_artifacts(
            sub,
            fixture=name,
            res=res,
            opts=opts,
            extra={"blade_spec": str(spec) if spec else None},
        )

        pngs = write_beam_model_pngs(
            sub,
            model,
            res,
            loads,
            span_plot_samples=int(args.span_plot_samples),
        )
        print(f"  PNGs: {len(pngs)} files")

        pdf_path = sub / f"beam_smoke_{name}.pdf"
        _write_pdf(model, res, loads, pdf_path)
        print(f"  PDF: {pdf_path}")

    print(f"\nAll done. Base directory: {base}")


if __name__ == "__main__":
    main()
