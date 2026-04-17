"""Build :class:`SectionStation` list for the global beam from GBT modal reduction."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.core.types import SectionStation
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


def _interp_k7_at_z(
    z: float,
    z_src: NDArray[np.float64],
    mats: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Piecewise-linear ``(7, 7)`` stiffness between structural stations."""
    zs = np.asarray(z_src, dtype=np.float64).ravel()
    if zs.size == 0:
        raise ValueError("z_src must be non-empty.")
    if z <= float(zs[0]):
        return np.asarray(mats[0], dtype=np.float64).copy()
    if z >= float(zs[-1]):
        return np.asarray(mats[-1], dtype=np.float64).copy()
    j = int(np.searchsorted(zs, z, side="right"))
    z0, z1 = float(zs[j - 1]), float(zs[j])
    a = (z - z0) / (z1 - z0)
    return ((1.0 - a) * np.asarray(mats[j - 1], dtype=np.float64) + a * np.asarray(mats[j], dtype=np.float64)).astype(
        np.float64, copy=False
    )


def beam_section_stations_from_gbt(
    station_z: NDArray[np.float64],
    section_definitions: tuple[Any, ...],
    bg: Any,
    n_beam_nodes: int,
) -> tuple[List[SectionStation], list[str]]:
    """
    Per structural station: GBT modal analysis → classical :class:`SectionStiffness`,
    PCHIP onto beam-node span coordinates, then ``SectionStation`` rows.
    """
    z_src = np.asarray(station_z, dtype=np.float64).ravel()
    n_s = int(z_src.shape[0])
    if len(section_definitions) != n_s:
        raise ValueError("section_definitions count must match station_z length.")

    stiff_list: list[SectionStiffness] = []
    k7_src: list[NDArray[np.float64]] = []
    reports: list[str] = []
    n_cross = max(24, 4 * int(n_beam_nodes))
    for sd in section_definitions:
        cs = section_definition_to_gbt_cross_section(sd)
        loads = SectionLoads(N=-1.0)
        full = CrossSectionModalAnalysis(cs, loads).run(n_modes=n_cross)
        sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
        reports.append(truncation_report(full, sel))
        st = gbt_to_beam_stiffness(full, sel, section=cs)
        stiff_list.append(st)
        k6_i = section_stiffness_to_k6(st, EIyz=st.EIyz)
        k7_src.append(gbt_to_k7(full, k6_i))

    arr = section_stiffness_array_from_sequence(z_src, stiff_list)
    zs = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_node = np.linspace(float(zs[0]), float(zs[-1]), int(n_beam_nodes), dtype=np.float64)
    interp = SectionPropertyInterpolator(z_src, arr)
    arr_n = interp.interpolate(z_node, allow_extrapolation=False)

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
        k7_i = _interp_k7_at_z(float(arr_n.s[i]), z_src, k7_src)
        stations.append(section_stiffness_to_station(float(arr_n.s[i]), st, K7=k7_i))
    return stations, reports
