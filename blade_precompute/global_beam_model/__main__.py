"""CLI: minimal nonlinear static beam solve → :class:`~global_beam_model.core.types.BeamSolveResult`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.postprocess import sample_resultants_at_z


def _gbt_model(yaml_path: Path, n_beam_nodes: int) -> bm.BeamModel:
    """Build a BeamModel from a blade YAML using the full GBT stiffness pipeline."""
    from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
    from blade_precompute.orchestration.gbt_beam_stations import beam_section_stations_from_gbt
    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.core.types import DesignVector
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder

    bg = BladeDesignProblem.load_geometry(yaml_path)
    n = int(bg.z_stations.shape[0])
    dv = DesignVector(
        t_skin=np.full(n, 0.012, dtype=np.float64),
        t_cap=np.full(n, 0.050, dtype=np.float64),
        t_web=np.full(n, 0.015, dtype=np.float64),
    )
    section_defs = SectionBuilder.build(dv, bg)
    z = np.array([float(sd.station_z) for sd in section_defs], dtype=np.float64)
    stations, _ = beam_section_stations_from_gbt(z, tuple(section_defs), bg, n_beam_nodes)
    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        tau0=np.asarray(bg.tau0, dtype=np.float64),
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
        "--yaml",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "example_blade.yaml",
        help="Blade geometry YAML (default: repo-root/example_blade.yaml).",
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
        help="Newton iteration cap (default tuned for the GBT/YAML-driven case).",
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
    model = _gbt_model(args.yaml, args.n_nodes)
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
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=bool(args.verbose),
    )
    res = bm.BeamAnalysis(model).solve_static(loads, options=opts)
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
        fig, _ = bmplot.plot_spanwise_resultants_nodal(res)
        figs.append(fig)
        fig, _ = bmplot.plot_spanwise_strains_nodal(res)
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
