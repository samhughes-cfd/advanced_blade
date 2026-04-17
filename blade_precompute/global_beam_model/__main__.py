"""CLI: minimal nonlinear static beam solve → :class:`~global_beam_model.core.types.BeamSolveResult`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
from blade_precompute.global_beam_model.engine.postprocess import sample_resultants_at_z


def _tapered_K7(z_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z0, z1 = float(z_nodes[0]), float(z_nodes[-1])
    n = z_nodes.shape[0]
    mats6 = np.zeros((n, 6, 6), dtype=np.float64)
    mats7 = np.zeros((n, 7, 7), dtype=np.float64)
    for i, z in enumerate(z_nodes):
        t = (z - z0) / (z1 - z0 + 1e-30)
        EA = 8.0e9 * (1.0 - 0.55 * t)
        EIy = 5.0e6 * (1.0 - 0.45 * t)
        EIz = 12.0e6 * (1.0 - 0.40 * t)
        GJ = 4.0e6 * (1.0 - 0.35 * t)
        kAy = 5.0e5 * (1.0 - 0.2 * t)
        kAz = 5.0e5 * (1.0 - 0.2 * t)
        Kww = 8.0e5 * (1.0 - 0.25 * t)
        mats6[i, 0, 0] = EA
        mats6[i, 1, 1] = EIy
        mats6[i, 2, 2] = EIz
        mats6[i, 3, 3] = GJ
        mats6[i, 4, 4] = kAy
        mats6[i, 5, 5] = kAz
        mats7[i, :6, :6] = mats6[i]
        mats7[i, 6, 6] = Kww
    return mats6, mats7


def _smoke_model() -> bm.BeamModel:
    L = 12.0
    n_st = 5
    z_st = np.linspace(0.0, L, n_st)
    x_pre = 0.02 * (z_st / L) ** 2
    r_ref = np.stack([x_pre, np.zeros_like(z_st), z_st], axis=1)
    kappa0 = np.zeros((n_st, 3), dtype=np.float64)
    for k in range(1, n_st - 1):
        zm, z0, z2 = z_st[k], z_st[k - 1], z_st[k + 1]
        xm = x_pre[k]
        x0 = x_pre[k - 1]
        x2 = x_pre[k + 1]
        d2x = ((x2 - xm) / (z2 - zm) - (xm - x0) / (zm - z0)) / (0.5 * (z2 - z0))
        kappa0[k, 1] = float(-d2x)
    geom = BladeGeometry(
        z_stations=z_st,
        r_ref=r_ref,
        kappa0=kappa0,
        tau0=np.zeros(n_st),
        chord=np.ones(n_st) * 0.5,
        twist=np.zeros(n_st),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2)),
        subcomponent_materials={},
        chi0=np.zeros(n_st),
    )
    n_nodes = 17
    K6s, K7s = _tapered_K7(z_st)
    stations = stations_from_arrays(z_st, K6s, K7s)
    return bm.BeamModel.from_blade_geometry(geom, n_nodes, stations, span_axis=2)


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
        "--max-iter",
        type=int,
        default=70,
        help="Newton iteration cap (default tuned for the built-in smoke case).",
    )
    parser.add_argument(
        "--n-load-steps",
        type=int,
        default=18,
        help="Load increments for the built-in distributed lateral load.",
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
