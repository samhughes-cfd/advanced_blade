"""CLI: minimal nonlinear static beam solve → :class:`~global_beam_model.core.types.BeamSolveResult`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.postprocess import sample_resultants_at_z
from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import (
    default_beam_loads,
    default_solver_options_for_synthetic_tapered,
    print_convergence_summary,
    smoke_model as smoke_model_synthetic_tapered,
)


def _gbt_model(blade_spec_path: Path, n_beam_nodes: int) -> bm.BeamModel:
    """Build a BeamModel from a blade spec using tabulated synthetic section stiffnesses."""
    from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
    from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
    from blade_precompute.section_optimisation.api import BladeDesignProblem

    bg = BladeDesignProblem.load_geometry(blade_spec_path)
    z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    n = int(z.shape[0])
    K6_template = np.diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6]).astype(np.float64)
    K7_template = np.zeros((7, 7), dtype=np.float64)
    K7_template[:6, :6] = K6_template
    K7_template[6, 6] = 1e4
    K6 = np.stack([K6_template.copy() for _ in range(n)], axis=0)
    K7 = np.stack([K7_template.copy() for _ in range(n)], axis=0)
    stations = stations_from_arrays(z, K6, K7)
    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        chord=np.asarray(bg.chord, dtype=np.float64),
        twist=np.asarray(bg.twist, dtype=np.float64),
        airfoil_profiles=list(bg.airfoil_profiles),
        web_positions=np.asarray(bg.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg.subcomponent_materials),
        chi0=None,
    )
    return bm.BeamModel.from_blade_geometry(geom, n_beam_nodes, stations, span_axis=2)


def _summarise(res: bm.BeamSolveResult, model: bm.BeamModel, *, print_spanwise: bool) -> None:
    tip = res.nodal_positions[-1] - model.X_ref[-1]
    print("BeamSolveResult:")
    print(f"  converged={res.converged}  n_iterations={res.n_iterations}")
    print(f"  |res|={res.residual_norm:.3e}")
    print(f"  resultants shape: {res.resultants.shape}")
    print(f"  tip displacement [m]: {tip}")
    print(f"  max |warping| [m²]: {float(np.max(np.abs(res.nodal_warping))):.6e}")
    if print_spanwise and res.z_stations_out is not None and res.z_stations_out.size > 0:
        z0 = float(model.X_ref[0, model.span_axis])
        z1 = float(model.X_ref[-1, model.span_axis])
        zq = np.linspace(z0, z1, 9)
        R_lines = sample_resultants_at_z(zq, res.z_stations_out, res.resultants)
        print("\nSpanwise resultants [N, Vy, Vz, My, Mz, T, B] (sample z):")
        for k, zz in enumerate(zq):
            R = R_lines[k]
            print(
                f"  z={zz:5.2f} m   N={R[0]:10.1f}  Vy={R[1]:10.1f}  Vz={R[2]:10.1f}  "
                f"My={R[3]:10.1f}  Mz={R[4]:10.1f}  T={R[5]:10.1f}  B={R[6]:10.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal beam static solve (canonical: BeamSolveResult).")
    parser.add_argument(
        "--synthetic-tapered",
        action="store_true",
        help=(
            "Use examples/synthetic_tapered_blade.py stiffness law and convergence-tuned solver "
            "(deterministic regression case); ignores --blade-spec for model build."
        ),
    )
    parser.add_argument(
        "--blade-spec",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "example_blade.json",
        help="Blade geometry spec JSON (default: repo-root/example_blade.json). Ignored with --synthetic-tapered.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=17,
        help="Number of beam nodes (default: 17).",
    )
    parser.add_argument(
        "--load-vy",
        type=float,
        default=350.0,
        help="Uniform lateral distributed load [N/m] along the span (default: 350.0).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=70,
        help="Newton iteration cap (default tuned for the spec-driven demo case).",
    )
    parser.add_argument(
        "--n-load-steps",
        type=int,
        default=18,
        help="Load increments for the distributed lateral load.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib figures (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="If set, save a multi-page PDF of beam plots to this path instead of showing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration solver diagnostics from the engine.",
    )
    parser.add_argument(
        "--print-spanwise",
        action="store_true",
        dest="print_spanwise",
        help="After the solve, print sampled spanwise resultant lines [N, Vy, Vz, My, Mz, T, B].",
    )
    args = parser.parse_args()
    if args.synthetic_tapered:
        model = smoke_model_synthetic_tapered()
        loads = default_beam_loads(model, q_y_Npm=float(args.load_vy))
        opts = default_solver_options_for_synthetic_tapered(
            max_iter=int(args.max_iter),
            n_load_steps=int(args.n_load_steps),
            verbose=bool(args.verbose),
        )
    else:
        model = _gbt_model(args.blade_spec, args.n_nodes)
        n = model.n_nodes
        q_line = np.zeros((len(model.elements), 3), dtype=np.float64)
        q_line[:, 1] = args.load_vy
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
            full_fd_hessian=False,
            spin_stabilization=1e-5,
            warping_stabilization=1e-3,
            verbose=bool(args.verbose),
        )
    res = bm.BeamAnalysis(model).solve_static(loads, options=opts)
    print_convergence_summary(res, model=model)
    _summarise(res, model, print_spanwise=bool(args.print_spanwise))

    if args.plot or args.plot_out is not None:
        from blade_precompute.global_beam_model.interface import plot as bmplot

        figs = []
        fig, _ = bmplot.plot_centerline_ref_def(model, res)
        figs.append(fig)
        fig, _ = bmplot.plot_spanwise_resultants(res)
        figs.append(fig)
        fig, _ = bmplot.plot_spanwise_strains(res)
        figs.append(fig)
        fig, _ = bmplot.plot_nodal_warping(model, res)
        figs.append(fig)
        fig, _ = bmplot.plot_iteration_history(res)
        figs.append(fig)
        fig, _ = bmplot.plot_reactions(res)
        figs.append(fig)
        fig, _ = bmplot.plot_distributed_loads(model, loads)
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
            print(f"Saved beam plots to {outp}")
        else:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
