"""Load precompute inputs and build orchestration context."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blade_precompute.orchestration import (
    PrecomputeOrchestrationContext,
    load_component_materials_json,
    resolve_system_type,
    validate_component_indices,
)

from blade_precompute.orchestration.precompute.containers import PrecomputeInputs
from blade_precompute.orchestration.precompute.grid import require_columns


def load_inputs(data_dir: Path) -> PrecomputeInputs:
    from data_library.plot_inputs import read_columnar_dat

    data_dir = data_dir.resolve()
    span_path = (data_dir / "blade_spanwise_distribution.dat").resolve()
    loads_path = (data_dir / "extreme_load_distribution.dat").resolve()
    if not span_path.is_file():
        raise FileNotFoundError(span_path)
    if not loads_path.is_file():
        raise FileNotFoundError(loads_path)

    span_names, span_data = read_columnar_dat(span_path)
    span_cols = {n: span_data[:, i] for i, n in enumerate(span_names)}
    require_columns(
        span_cols,
        ("r_z_m", "chord_m", "twist_deg", "naca_m", "naca_p", "naca_xx"),
        path=span_path,
    )

    load_names, load_data = read_columnar_dat(loads_path)
    load_cols = {n: load_data[:, i] for i, n in enumerate(load_names)}
    require_columns(load_cols, ("r_z_m", "q_y_Npm", "q_z_Npm", "m_x_Nmpm"), path=loads_path)

    return PrecomputeInputs(
        spanwise_path=span_path,
        extreme_loads_path=loads_path,
        span_r_z_m=np.asarray(span_cols["r_z_m"], dtype=np.float64),
        chord_m=np.asarray(span_cols["chord_m"], dtype=np.float64),
        twist_deg=np.asarray(span_cols["twist_deg"], dtype=np.float64),
        naca_m=np.asarray(span_cols["naca_m"], dtype=np.float64),
        naca_p=np.asarray(span_cols["naca_p"], dtype=np.float64),
        naca_xx=np.asarray(span_cols["naca_xx"], dtype=np.float64),
        loads_r_z_m=np.asarray(load_cols["r_z_m"], dtype=np.float64),
        q_y_Npm=np.asarray(load_cols["q_y_Npm"], dtype=np.float64),
        q_z_Npm=np.asarray(load_cols["q_z_Npm"], dtype=np.float64),
        m_x_Nmpm=np.asarray(load_cols["m_x_Nmpm"], dtype=np.float64),
    )


def resolve_component_materials_path(data_dir: Path, explicit: Path | None) -> Path:
    data_dir = data_dir.resolve()
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    cand = (data_dir / "component_materials.json").resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        f"No --component-materials provided and default file missing: {cand}. "
        "Pass --component-materials path/to.json (see data_library/component_materials.json)."
    )


def build_precompute_orchestration_context(
    *,
    data_dir: Path,
    blade_yaml: Path,
    system_type_key: str,
    component_materials_path: Path | None,
) -> PrecomputeOrchestrationContext:
    mat_path = resolve_component_materials_path(data_dir, component_materials_path)
    cmap = load_component_materials_json(mat_path)
    validate_component_indices(blade_yaml.resolve(), cmap)
    layout = resolve_system_type(system_type_key)
    return PrecomputeOrchestrationContext(
        system_type_key=str(system_type_key).strip(),
        layout=layout,
        component_materials=cmap,
    )
