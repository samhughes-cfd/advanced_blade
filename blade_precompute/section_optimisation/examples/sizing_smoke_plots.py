from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blade_precompute.section_optimisation import BladeOptimizer
from blade_precompute.section_optimisation.__main__ import _repo_root, _smoke_problem_builtin, _smoke_problem_from_spec


def main() -> None:
    p = argparse.ArgumentParser(description="Run the section_optimisation smoke sizing case and save plots to PDF.")
    p.add_argument(
        "--blade-spec",
        type=Path,
        default=None,
        help="Optional blade geometry spec JSON. Default: example_blade_10.json or example_blade.json if present, else built-in geometry.",
    )
    p.add_argument("--optimise", action="store_true", help="Run SLSQP optimisation after evaluation.")
    p.add_argument("--maxiter", type=int, default=120, help="SLSQP max iterations when --optimise is set.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "design_smoke.pdf",
        help="Output PDF path (default: examples/output/design_smoke.pdf).",
    )
    args = p.parse_args()

    blade_spec_path = args.blade_spec
    if blade_spec_path is None:
        for name in ("example_blade_10.json", "example_blade.json"):
            candidate = _repo_root() / name
            if candidate.is_file():
                blade_spec_path = candidate
                break

    if blade_spec_path is not None:
        sizing, dv0 = _smoke_problem_from_spec(blade_spec_path)
    else:
        sizing, dv0 = _smoke_problem_builtin()

    ev = sizing.evaluate(dv0)
    res = None
    if args.optimise:
        opt = BladeOptimizer(sizing.problem, options={"maxiter": args.maxiter, "ftol": 1e-5, "disp": False})
        res = opt.run(dv0)

    from blade_precompute.section_optimisation.interface import plot as dplot

    z = np.asarray(sizing.problem.blade_geometry.z_stations, dtype=np.float64)
    figs = []
    fig, _ = dplot.plot_design_vector_vs_span(z, dv0, title="Initial design vector")
    figs.append(fig)
    if res is not None:
        fig, _ = dplot.plot_design_vector_vs_span(z, res.dv_opt, dv_compare=dv0, title="Optimised vs initial thickness")
        figs.append(fig)
        fig, _ = dplot.plot_optimisation_history(res)
        figs.append(fig)

    outp = args.out.resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(outp) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)

    print(f"Saved: {outp}  (mass={ev.mass:.4f} kg, max_Hashin={ev.max_fi_hashin:.4f}, max_VM={ev.max_fi_vm:.4f})")


if __name__ == "__main__":
    main()

