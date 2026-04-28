"""Dev helper: beam convergence smoke (uses cached section NPZ).

This is intentionally under tools/scratch and is not part of the package API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.core.types import BeamLoads, BoundaryCondition, SolverOptions
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
from blade_precompute.orchestration.precompute import (
    LinspaceSpec,
    job_span_z_m,
    linspace_from_spec,
    load_inputs,
    resample_blade_geometry_to_z,
    resample_precompute_inputs,
)
from blade_precompute.section_optimisation.api import BladeDesignProblem

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "outputs" / "20260424_001415" / "section_properties" / "section_solve_stations.npz"


def main() -> None:
    data = np.load(NPZ)
    z_sec = np.asarray(data["z_stations"], dtype=np.float64).ravel()
    K6 = np.asarray(data["K6_stack"], dtype=np.float64)
    K7 = np.asarray(data["K7_stack"], dtype=np.float64)
    stations = stations_from_arrays(z_sec, K6, K7)

    blade_spec = ROOT / "example_blade_10.json"
    bg = BladeDesignProblem.load_geometry(blade_spec.resolve())
    z_struct = bg.z_stations.ravel()
    sspec = LinspaceSpec(
        z_min=float(z_struct[0]),
        z_max=float(z_struct[-1]),
        n=int(z_struct.shape[0]),
    )
    bg_struct = resample_blade_geometry_to_z(bg, linspace_from_spec(sspec))

    geom = BladeGeometry(
        z_stations=np.asarray(bg_struct.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg_struct.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg_struct.kappa0, dtype=np.float64),
        chord=np.asarray(bg_struct.chord, dtype=np.float64),
        twist=np.asarray(bg_struct.twist, dtype=np.float64),
        airfoil_profiles=list(bg_struct.airfoil_profiles),
        web_positions=np.asarray(bg_struct.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg_struct.subcomponent_materials),
        chi0=None,
    )

    inp = load_inputs(ROOT / "data_library")
    zg0, zg1 = job_span_z_m(inp)
    gspec = LinspaceSpec(z_min=zg0, z_max=zg1, n=int(inp.span_r_z_m.size))
    inp_geom = resample_precompute_inputs(inp, linspace_from_spec(gspec))

    analysis = BeamAnalysis.from_blade_geometry(geom, 50, stations, span_axis=2)
    model = analysis.model
    n_nodes, n_elem = model.n_nodes, len(model.elements)
    z_mid = np.asarray([el.z_mid for el in model.elements], dtype=np.float64)
    qy = np.interp(z_mid, inp_geom.loads_r_z_m, inp_geom.q_y_Npm)
    qz = np.interp(z_mid, inp_geom.loads_r_z_m, inp_geom.q_z_Npm)
    mx = np.interp(z_mid, inp_geom.loads_r_z_m, inp_geom.m_x_Nmpm)
    distributed_q = np.zeros((n_elem, 3), dtype=np.float64)
    distributed_q[:, 1] = qy
    distributed_q[:, 2] = qz
    loads = BeamLoads(
        nodal_F=np.zeros((n_nodes, 3), dtype=np.float64),
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        distributed_q=distributed_q,
        distributed_mz=np.asarray(mx, dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )
    opts = SolverOptions(
        max_iter=120,
        tol_res=5e-2,
        tol_res_rel=5e-3,
        tol_du=1e-7,
        n_gauss=2,
        n_load_steps=180,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        relax_factor=0.85,
        verbose=False,
    )
    res = analysis.solve_static(loads, options=opts)
    print("precompute_default_opts", res.converged, res.residual_norm, res.n_iterations)


if __name__ == "__main__":
    main()
