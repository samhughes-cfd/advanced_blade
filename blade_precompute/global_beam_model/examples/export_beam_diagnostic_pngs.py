"""Write global_beam_model diagnostic PNGs (resultants, strains, deformation, convergence, etc.).

Uses the same layout as precompute ``write_beam_model_pngs`` (see
``blade_precompute.orchestration.precompute.vis``), including
``beam_iteration_history.png``, ``beam_resultants_nodal.png``,
``beam_strains_nodal.png``, ``beam_section_stress_nodal.png``, ``beam_warping.png``,
and other section plots when the solve carries section recovery fields.

From the repo root::

    python -m blade_precompute.global_beam_model.examples.export_beam_diagnostic_pngs

writes under ``examples/output/beam_diagnostic_pngs`` by default.
"""

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
    DEFAULT_PNG_DIR,
    solve_beam_examples_case,
)
from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import print_convergence_summary
from blade_precompute.orchestration.precompute.vis import write_beam_model_pngs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Solve a beam smoke case and export matplotlib PNGs like precompute global_beam_model stage."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PNG_DIR,
        help=f"Directory for PNG files (default: {DEFAULT_PNG_DIR}).",
    )
    p.add_argument(
        "--fixture",
        choices=("synthetic", "gbt"),
        default="synthetic",
        help="synthetic: synthetic_tapered_blade; gbt: blade JSON + tabulated K6/K7.",
    )
    p.add_argument(
        "--blade-spec",
        type=Path,
        default=DEFAULT_BLADE_SPEC,
        help="Blade JSON for --fixture gbt.",
    )
    p.add_argument("--n-nodes", type=int, default=17, help="Beam nodes (gbt only).")
    p.add_argument("--load-vy", type=float, default=350.0, help="Uniform q_y [N/m] on elements.")
    p.add_argument("--max-iter", type=int, default=110)
    p.add_argument("--n-load-steps", type=int, default=72)
    p.add_argument(
        "--span-plot-samples",
        type=int,
        default=400,
        help="Uniform span samples for warping / distributed-load abscissa (default 400).",
    )
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, loads, res, _ = solve_beam_examples_case(
        args.fixture,
        blade_spec=args.blade_spec,
        n_nodes=int(args.n_nodes),
        load_vy=float(args.load_vy),
        max_iter=int(args.max_iter),
        n_load_steps=int(args.n_load_steps),
        verbose=False,
    )
    print_convergence_summary(res, model=model)

    paths = write_beam_model_pngs(
        out_dir,
        model,
        res,
        loads,
        span_plot_samples=int(args.span_plot_samples),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
