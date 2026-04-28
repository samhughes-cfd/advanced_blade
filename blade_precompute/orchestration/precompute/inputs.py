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


_SPAN_EXPECTED_UNITS: dict[str, str] = {
    "spanwise_pos": "m",
    "radial_pos": "m",
    "norm_radial_pos": "-",
    "norm_spanwise_pos": "-",
    "chord_dist": "m",
    "twist_dist": "deg",
    "naca_series": "-",
    "naca_m": "-",
    "naca_p": "-",
    "naca_xx": "-",
    "kappa0_x": "1/m",
    "kappa0_y": "1/m",
    "kappa0_z": "1/m",
}

_LOADS_EXPECTED_UNITS: dict[str, str] = {
    "spanwise_pos": "m",
    "radial_pos": "m",
    "q_y_Npm": "N/m",
    "q_z_Npm": "N/m",
    "m_x_Nmpm": "N*m/m",
}


def _assert_dat_units(
    col_names: list[str],
    col_units: list[str],
    expected: dict[str, str],
    *,
    path: Path,
) -> None:
    """Raise ``ValueError`` listing every column whose declared unit does not match ``expected``."""
    from data_library.plot_inputs import _canonicalise_unit

    mismatches: list[str] = []
    unit_map = dict(zip(col_names, col_units))
    for col, want in expected.items():
        got = unit_map.get(col)
        if got is None:
            continue  # column-presence validated separately by require_columns
        if _canonicalise_unit(got) != _canonicalise_unit(want):
            mismatches.append(f"  {col!r}: file declares {got!r}, expected {want!r}")
    if mismatches:
        raise ValueError(
            f"{path}: unit mismatch(es) in '# units:' row:\n" + "\n".join(mismatches)
            + "\nCheck DAT_STYLE.md for the canonical unit grammar."
        )


def load_inputs(data_dir: Path) -> PrecomputeInputs:
    from data_library.plot_inputs import read_columnar_dat_with_units

    data_dir = data_dir.resolve()
    span_path = (data_dir / "blade_spanwise_distribution.dat").resolve()
    loads_path = (data_dir / "extreme_load_distribution.dat").resolve()
    if not span_path.is_file():
        raise FileNotFoundError(span_path)
    if not loads_path.is_file():
        raise FileNotFoundError(loads_path)

    span_names, span_units, span_data = read_columnar_dat_with_units(span_path)
    _assert_dat_units(span_names, span_units, _SPAN_EXPECTED_UNITS, path=span_path)
    span_cols = {n: span_data[:, i] for i, n in enumerate(span_names)}
    require_columns(
        span_cols,
        (
            "spanwise_pos",
            "radial_pos",
            "norm_radial_pos",
            "norm_spanwise_pos",
            "chord_dist",
            "twist_dist",
            "kappa0_x",
            "kappa0_y",
            "kappa0_z",
            "naca_series",
            "naca_m",
            "naca_p",
            "naca_xx",
        ),
        path=span_path,
    )
    n_span = int(np.asarray(span_cols["spanwise_pos"], dtype=np.float64).size)
    naca_series = np.asarray(np.round(span_cols["naca_series"]), dtype=np.int64).ravel()
    if naca_series.shape[0] != n_span:
        raise ValueError(f"naca_series length {naca_series.shape[0]} != span rows {n_span} in {span_path}")

    load_names, load_units, load_data = read_columnar_dat_with_units(loads_path)
    _assert_dat_units(load_names, load_units, _LOADS_EXPECTED_UNITS, path=loads_path)
    load_cols = {n: load_data[:, i] for i, n in enumerate(load_names)}
    require_columns(
        load_cols,
        ("spanwise_pos", "radial_pos", "q_y_Npm", "q_z_Npm", "m_x_Nmpm"),
        path=loads_path,
    )

    return PrecomputeInputs(
        spanwise_path=span_path,
        extreme_loads_path=loads_path,
        span_r_z_m=np.asarray(span_cols["spanwise_pos"], dtype=np.float64),
        radial_r_m=np.asarray(span_cols["radial_pos"], dtype=np.float64),
        chord_m=np.asarray(span_cols["chord_dist"], dtype=np.float64),
        twist_deg=np.asarray(span_cols["twist_dist"], dtype=np.float64),
        kappa0_x=np.asarray(span_cols["kappa0_x"], dtype=np.float64),
        kappa0_y=np.asarray(span_cols["kappa0_y"], dtype=np.float64),
        kappa0_z=np.asarray(span_cols["kappa0_z"], dtype=np.float64),
        naca_m=np.asarray(span_cols["naca_m"], dtype=np.float64),
        naca_p=np.asarray(span_cols["naca_p"], dtype=np.float64),
        naca_xx=np.asarray(span_cols["naca_xx"], dtype=np.float64),
        naca_series=naca_series,
        loads_r_z_m=np.asarray(load_cols["spanwise_pos"], dtype=np.float64),
        q_y_Npm=np.asarray(load_cols["q_y_Npm"], dtype=np.float64),
        q_z_Npm=np.asarray(load_cols["q_z_Npm"], dtype=np.float64),
        m_x_Nmpm=np.asarray(load_cols["m_x_Nmpm"], dtype=np.float64),
    )


def resolve_component_materials_path(data_dir: Path, explicit: Path | None) -> Path | None:
    """Return the resolved path to ``component_materials.json``, or ``None`` if absent.

    When an explicit path is provided it must exist (``FileNotFoundError`` on miss).
    When no explicit path is given the default ``data_dir/component_materials.json`` is
    tried; if absent ``None`` is returned (the caller synthesises a stub map from the
    material library instead).
    """
    data_dir = data_dir.resolve()
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    cand = (data_dir / "component_materials.json").resolve()
    if cand.is_file():
        return cand
    return None


def build_precompute_orchestration_context(
    *,
    data_dir: Path,
    blade_yaml: Path | None,
    system_type_key: str,
    component_materials_path: Path | None,
    skip_component_index_validation: bool = False,
    component_materials_override: "ComponentMaterialsMap | None" = None,
) -> PrecomputeOrchestrationContext:
    """``blade_yaml`` may be ``None`` when precompute uses spanwise + material-library blade inputs only.

    If ``component_materials.json`` is absent and no ``component_materials_override`` is supplied,
    a stub map (skin=0, spar_cap=0, shear_web=0) is used so the metadata field is populated.
    Pass ``component_materials_override`` to propagate the actual logical map derived from
    ``SUBCOMPONENT_MATERIAL_IDS``.
    """
    from blade_precompute.orchestration.component_materials import ComponentMaterialsMap

    mat_path = resolve_component_materials_path(data_dir, component_materials_path)
    if component_materials_override is not None:
        cmap = component_materials_override
    elif mat_path is not None:
        cmap = load_component_materials_json(mat_path)
    else:
        cmap = ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0)
    if not skip_component_index_validation and blade_yaml is not None and mat_path is not None:
        validate_component_indices(Path(blade_yaml).resolve(), cmap)
    layout = resolve_system_type(system_type_key)
    return PrecomputeOrchestrationContext(
        system_type_key=str(system_type_key).strip(),
        layout=layout,
        component_materials=cmap,
    )
