"""CLI: blade sizing smoke — :class:`~design_optimisation.core.types.DesignEvaluation` or optimisation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blade_precompute.design_optimisation import (
    BladeDesignProblem,
    BladeOptimizer,
    DesignProblem,
    DesignVector,
    ExtremeLoads,
    OptimBladeGeometry,
)
from blade_precompute.design_optimisation.core.types import DesignEvaluation, OptimizationObjective, OptimisationResult
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _objective_from_cli(s: str) -> OptimizationObjective:
    key = (s or "").strip().lower().replace("_", "-")
    if key in ("min-mass", "minmass"):
        return "min_mass"
    if key in ("max-specific-stiffness", "max-stiffness-mass", "specific-stiffness"):
        return "max_specific_stiffness"
    raise ValueError(f"Unknown objective {s!r}; use min-mass or max-specific-stiffness.")


def _ply_gfrp(t_ply: float) -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp",
        E1=42e9,
        E2=12e9,
        G12=4.5e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=t_ply,
        Xt=900e6,
        Xc=650e6,
        Yt=65e6,
        Yc=120e6,
        S12=75e6,
        Zt=45e6,
        S13=40e6,
        S23=40e6,
    )


def _ply_cfrp(t_ply: float) -> OrthotropicPly:
    return OrthotropicPly(
        name="cfrp",
        E1=135e9,
        E2=9e9,
        G12=4.8e9,
        nu12=0.28,
        rho=1600.0,
        t_ply=t_ply,
        Xt=1800e6,
        Xc=1200e6,
        Yt=60e6,
        Yc=220e6,
        S12=90e6,
        Zt=55e6,
        S13=45e6,
        S23=45e6,
    )


def _smoke_problem_from_yaml(yaml_path: Path) -> tuple[BladeDesignProblem, DesignVector]:
    bg = BladeDesignProblem.load_geometry(yaml_path)
    z = np.asarray(bg.z_stations, dtype=np.float64)
    L = float(z[-1]) if z.size else 1.0
    scale = z / max(L, 1e-12)
    loads = ExtremeLoads(
        z_stations=z,
        N=np.zeros_like(z),
        Vy=np.zeros_like(z),
        Vz=np.zeros_like(z),
        My=2.18 * scale,
        Mz=np.zeros_like(z),
        T=np.zeros_like(z),
        B=None,
    )
    problem = DesignProblem(
        blade_geometry=bg,
        extreme_loads=loads,
        solver=None,
        objective=objective,
        ks_rho=35.0,
        enable_tier3_delam=False,
        n_workers=1,
    )
    dv0 = DesignVector(
        t_skin=np.full(z.shape[0], 0.012),
        t_cap=np.full(z.shape[0], 0.050),
        t_web=np.full(z.shape[0], 0.015),
    )
    return BladeDesignProblem(problem), dv0


def _smoke_problem_builtin(
    *, objective: OptimizationObjective = "min_mass"
) -> tuple[BladeDesignProblem, DesignVector]:
    z = np.linspace(0.0, 8.0, 5, dtype=np.float64)
    L = float(z[-1])
    r_ref = np.zeros((z.shape[0], 3), dtype=np.float64)
    r_ref[:, 2] = z
    r_ref[:, 1] = 0.015 * (z / L) ** 2
    kappa0 = np.zeros((z.shape[0], 3), dtype=np.float64)
    tau0 = np.zeros_like(z)
    chord = np.linspace(2.0, 1.2, z.shape[0], dtype=np.float64)
    twist = np.zeros_like(z)
    web_positions = np.array([-0.32, 0.32], dtype=np.float64)
    t0 = 0.0002
    lam_skin = LaminateDefinition(
        plies=[
            (_ply_gfrp(t0), 0.0),
            (_ply_gfrp(t0), 45.0),
            (_ply_gfrp(t0), -45.0),
            (_ply_gfrp(t0), 90.0),
        ],
        shear_lag_correction=True,
    )
    lam_cap = LaminateDefinition(plies=[(_ply_cfrp(t0), 0.0)] * 8, shear_lag_correction=True)
    lam_web = LaminateDefinition(
        plies=[
            (_ply_gfrp(t0), 45.0),
            (_ply_gfrp(t0), -45.0),
            (_ply_gfrp(t0), 45.0),
            (_ply_gfrp(t0), -45.0),
        ],
        shear_lag_correction=True,
    )
    al = IsotropicMaterial(
        name="al6082",
        E=70e9,
        nu=0.33,
        rho=2700.0,
        sigma_allow=260e6,
    )
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=kappa0,
        tau0=tau0,
        chord=chord,
        twist=twist,
        airfoil_profiles=[],
        web_positions=web_positions,
        subcomponent_materials={
            "skin": lam_skin,
            "cap_ps": lam_cap,
            "web": lam_web,
            "leading_edge_insert": al,
        },
        thickness_role={"leading_edge_insert": "fixed"},
        cap_shear_lag_width=None,
        box_height_frac=0.11,
    )
    scale = z / max(L, 1e-12)
    loads = ExtremeLoads(
        z_stations=z,
        N=np.zeros_like(z),
        Vy=np.zeros_like(z),
        Vz=np.zeros_like(z),
        My=2.18 * scale,
        Mz=np.zeros_like(z),
        T=np.zeros_like(z),
        B=None,
    )
    problem = DesignProblem(
        blade_geometry=bg,
        extreme_loads=loads,
        solver=None,
        objective=objective,
        ks_rho=35.0,
        enable_tier3_delam=False,
        n_workers=1,
    )
    dv0 = DesignVector(
        t_skin=np.full(z.shape[0], 0.012),
        t_cap=np.full(z.shape[0], 0.050),
        t_web=np.full(z.shape[0], 0.015),
    )
    return BladeDesignProblem(problem), dv0


def _print_evaluation(ev: DesignEvaluation) -> None:
    print("DesignEvaluation:")
    print(f"  mass [kg]: {ev.mass:.6g}")
    print(f"  stiffness_metric (int. trace K7): {ev.stiffness_metric:.6g}  S/m: {ev.stiffness_metric / max(ev.mass, 1e-300):.6g}")
    print(f"  max_fi_tw: {ev.max_fi_tw:.6g}  max_fi_vm: {ev.max_fi_vm:.6g}")
    if ev.fi_delam is not None:
        print(f"  max_fi_delam: {ev.max_fi_delam}")


def _print_optimisation(res: OptimisationResult) -> None:
    print("OptimisationResult:")
    print(f"  success={res.success}  message={res.message!r}  n_iter={res.n_iter}")
    print(f"  dv_opt t_skin mean [m]: {float(np.mean(res.dv_opt.t_skin)):.6g}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate (or optimise) a blade sizing problem; canonical: DesignEvaluation or OptimisationResult."
    )
    p.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help=f"Blade geometry YAML (default: example_blade_10.yaml or example_blade.yaml under {_repo_root()} if present, else built-in geometry).",
    )
    p.add_argument(
        "--optimise",
        action="store_true",
        help="Run BladeOptimizer after the initial evaluation.",
    )
    p.add_argument("--maxiter", type=int, default=120, help="SLSQP max iterations when --optimise is set.")
    p.add_argument("--plot", action="store_true", help="Show matplotlib figures (requires matplotlib).")
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Save design plots to a multi-page PDF at this path instead of showing.",
    )
    p.add_argument(
        "--objective",
        type=str,
        default="min-mass",
        help="Optimization objective when using --optimise: min-mass (default) or max-specific-stiffness (maximize integrated trace(K7)/mass via log objective).",
    )
    args = p.parse_args()
    objective = _objective_from_cli(args.objective)
    yaml_path = args.yaml
    if yaml_path is None:
        for name in ("example_blade_10.yaml", "example_blade.yaml"):
            candidate = _repo_root() / name
            if candidate.is_file():
                yaml_path = candidate
                break
    if yaml_path is not None:
        sizing, dv0 = _smoke_problem_from_yaml(yaml_path, objective=objective)
    else:
        sizing, dv0 = _smoke_problem_builtin(objective=objective)
    ev = sizing.evaluate(dv0)
    _print_evaluation(ev)
    res = None
    if args.optimise:
        opt = BladeOptimizer(sizing.problem, options={"maxiter": args.maxiter, "ftol": 1e-5, "disp": False})
        res = opt.run(dv0)
        _print_optimisation(res)

    if args.plot or args.plot_out is not None:
        from blade_precompute.design_optimisation.interface import plot as dplot

        z = np.asarray(sizing.problem.blade_geometry.z_stations, dtype=np.float64)
        figs = []
        fig, _ = dplot.plot_design_vector_vs_span(z, dv0, title="Initial design vector")
        figs.append(fig)
        if res is not None:
            fig, _ = dplot.plot_design_vector_vs_span(
                z, res.dv_opt, dv_compare=dv0, title="Optimised vs initial thickness"
            )
            figs.append(fig)
            fig, _ = dplot.plot_optimisation_history(res)
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
            print(f"Saved design plots to {outp}")
        else:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
