"""Build :class:`SectionStation` list for the global beam from GBT modal reduction."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.core.types import K7Array, SectionStation
from blade_precompute.global_beam_model.k7_interpolation import K7Interpolator
from blade_precompute.global_beam_model.section_property_interpolator import (
    SectionPropertyInterpolator,
    section_stiffness_array_from_sequence,
)
from blade_precompute.section_beam_model.gbt import (
    CrossSectionModalAnalysis,
    DEFAULT_BEAM_EXPORT_MODE_LABELS,
    SectionLoads,
    select_modes,
    truncation_report,
)
from blade_precompute.section_beam_model.gbt.section_stiffness_export import (
    SectionStiffness,
    gbt_to_beam_stiffness,
    gbt_to_k7,
    section_stiffness_to_k6,
    section_stiffness_to_station,
)
from blade_precompute.section_buckling.interface.precompute import section_definition_to_gbt_cross_section


def beam_section_stations_from_gbt(
    station_z: NDArray[np.float64],
    section_definitions: tuple[Any, ...],
    bg: Any,
    n_beam_nodes: int,
    *,
    n_modes_floor: int = 24,
    n_modes_multiplier: int = 4,
) -> tuple[List[SectionStation], list[str]]:
    """
    Per structural station: GBT modal analysis → classical :class:`SectionStiffness`,
    PCHIP onto beam-node span coordinates, then ``SectionStation`` rows.

    n_modes_floor: minimum GBT modes computed per cross-section regardless of mesh
        density. Default 24 ensures at least the 4 classical export modes plus
        several distortional modes are resolved.
    n_modes_multiplier: target n_cross = multiplier × n_beam_nodes. Increase for
        finer meshes where higher distortional modes may couple to beam DOFs.
        Default 4.
    """
    z_src = np.asarray(station_z, dtype=np.float64).ravel()
    n_s = int(z_src.shape[0])
    if len(section_definitions) != n_s:
        raise ValueError("section_definitions count must match station_z length.")

    stiff_list: list[SectionStiffness] = []
    k7_src: list[NDArray[np.float64]] = []
    reports: list[str] = []
    n_cross = max(n_modes_floor, n_modes_multiplier * int(n_beam_nodes))
    for sd in section_definitions:
        cs = section_definition_to_gbt_cross_section(sd)
        loads = SectionLoads(N=-1.0)
        full = CrossSectionModalAnalysis(cs, loads).run(n_modes=n_cross)
        sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
        reports.append(truncation_report(full, sel))
        st = gbt_to_beam_stiffness(full, sel, section=cs)
        stiff_list.append(st)
        k6_i = section_stiffness_to_k6(st, EIyz=st.EIyz)
        k7_src.append(gbt_to_k7(full, k6_i, full_vlasov=True))

    arr = section_stiffness_array_from_sequence(z_src, stiff_list)
    zs = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_node = np.linspace(float(zs[0]), float(zs[-1]), int(n_beam_nodes), dtype=np.float64)
    interp = SectionPropertyInterpolator(z_src, arr)
    arr_n = interp.interpolate(z_node, allow_extrapolation=False)

    k7_stack = np.stack(k7_src, axis=0)
    k7_arr = K7Array(s=z_src, entries=k7_stack)
    k7_node = K7Interpolator(k7_arr).interpolate(z_node, allow_extrapolation=False)

    stations: list[SectionStation] = []
    for i in range(int(n_beam_nodes)):
        st = SectionStiffness(
            EA=float(arr_n.EA[i]),
            EI_x=float(arr_n.EI_x[i]),
            EI_y=float(arr_n.EI_y[i]),
            GJ=float(arr_n.GJ[i]),
            GA_x=float(arr_n.GA_x[i]),
            GA_y=float(arr_n.GA_y[i]),
            EIyz=float(arr_n.EIyz[i]),
        )
        k7_i = np.asarray(k7_node.entries[i], dtype=np.float64).copy()
        stations.append(section_stiffness_to_station(float(arr_n.s[i]), st, K7=k7_i))
    return stations, reports
