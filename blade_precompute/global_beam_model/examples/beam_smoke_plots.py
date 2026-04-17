from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while _p.name != "blade_precompute" and _p.parent != _p:
    _p = _p.parent
if _p.name == "blade_precompute":
    sys.path.insert(0, str(_p.parent))

import blade_precompute.beam_model as bm
from blade_precompute.beam_model.__main__ import _smoke_model


def main() -> None:
    p = argparse.ArgumentParser(description="Run the beam_model smoke case and save plots to PDF.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "beam_smoke.pdf",
        help="Output PDF path (default: examples/output/beam_smoke.pdf).",
    )
    p.add_argument("--max-iter", type=int, default=70, help="Newton iteration cap.")
    p.add_argument("--n-load-steps", type=int, default=18, help="Number of load steps.")
    args = p.parse_args()

    model = _smoke_model()
    n = model.n_nodes
    q_line = np.zeros((len(model.elements), 3), dtype=np.float64)
    q_line[:, 1] = 350.0
    loads = bm.BeamLoads(
        nodal_F=np.zeros((n, 3)),
        nodal_M=np.zeros((n, 3)),
        distributed_q=q_line,
        bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
    )
    opts = bm.SolverOptions(
        max_iter=args.max_iter,
        tol_res=5e-2,
        tol_res_rel=5e-3,
        tol_du=1e-7,
        n_gauss=2,
        n_load_steps=args.n_load_steps,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res = bm.BeamAnalysis(model).solve_static(loads, options=opts)

    from blade_precompute.beam_model.interface import plot as bmplot

    figs = []
    for fn in [
        bmplot.plot_centerline_ref_def,
        bmplot.plot_spanwise_resultants,
        bmplot.plot_spanwise_strains,
        bmplot.plot_spanwise_resultants_nodal,
        bmplot.plot_spanwise_strains_nodal,
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
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(outp) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)

    print(f"Saved: {outp}  (converged={res.converged}, n_iter={res.n_iterations})")


if __name__ == "__main__":
    main()

