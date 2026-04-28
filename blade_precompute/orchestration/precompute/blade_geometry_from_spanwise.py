"""Build :class:`OptimBladeGeometry` from resampled :class:`PrecomputeInputs`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute.containers import PrecomputeInputs
from blade_precompute.orchestration.precompute.material_library import (
    MaterialRow,
    subcomponent_box_materials_from_csv,
)
from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF


def build_optim_blade_geometry_from_spanwise(
    inp: PrecomputeInputs,
    *,
    mat_table: Mapping[int, MaterialRow],
    logical: Mapping[str, int],
    web_positions: Sequence[float] | NDArray[np.float64] | None = None,
    box_height_frac: float | None = None,
    cap_shear_lag_width: float | None = None,
    layup_skin: list[float] | None = None,
    layup_cap: list[float] | None = None,
    layup_web: list[float] | None = None,
    naca_airfoil_n_points: int = 200,
    system_layout: "Any | None" = None,
) -> OptimBladeGeometry:
    """
    Assemble a structural blade definition matching the example blade layout (``skin`` / ``cap_ps`` / ``web``,
    NACA from spanwise columns, ``r_ref`` from ``(0, radial - radial[root], z)``).

    ``web_positions`` and ``box_height_frac`` are derived automatically when not supplied:

    * ``web_positions``: derived from ``system_layout.web_chord_fracs`` (converted from
      0–1 chord-fraction to the ``OptimBladeGeometry`` half-chord-normalised convention
      ``pos = frac - 0.5`` so that ``-0.5 = LE, +0.5 = TE``).  Requires ``system_layout``
      with exactly two web fractions when omitted.
    * ``box_height_frac``: derived from the mean NACA max-thickness ratio across all
      spanwise stations (``mean(naca_xx) / 100``).  This is physically consistent with the
      airfoil geometry rather than an independently prescribed constant.
    """
    z = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    n = int(z.size)
    if n < 1:
        raise ValueError("PrecomputeInputs must have at least one spanwise station.")
    for name, a in {
        "chord_m": inp.chord_m,
        "twist_deg": inp.twist_deg,
        "naca_m": inp.naca_m,
        "naca_p": inp.naca_p,
        "naca_xx": inp.naca_xx,
        "naca_series": inp.naca_series,
    }.items():
        if int(np.asarray(a, dtype=np.float64).ravel().shape[0]) != n:
            raise ValueError(f"Length mismatch: {name} vs span_r_z_m")

    if layup_skin is None:
        layup_skin = [0.0, 45.0, -45.0, 90.0]
    if layup_cap is None:
        layup_cap = [0.0, 0.0]
    if layup_web is None:
        layup_web = [45.0, -45.0]

    subcomponent_materials, thickness_role = subcomponent_box_materials_from_csv(
        mat_table,
        logical,
        layup_skin=layup_skin,
        layup_cap=layup_cap,
        layup_web=layup_web,
    )
    if "core" in logical:
        warnings.warn(
            "logical['core'] is currently ignored by build_optim_blade_geometry_from_spanwise; "
            "only skin/spar_cap/shear_web are used for box subcomponents.",
            stacklevel=2,
        )
    rr = np.asarray(inp.radial_r_m, dtype=np.float64).ravel()
    if rr.size != n:
        raise ValueError("radial_r_m length must match span_r_z_m.")
    y0 = float(rr[0]) if np.all(np.isfinite(rr)) else 0.0
    yl = (rr - y0) if np.all(np.isfinite(rr)) else np.zeros(n, dtype=np.float64)
    r_ref = np.column_stack(
        (np.zeros(n, dtype=np.float64), yl, z)
    )
    kappa0 = np.column_stack(
        (
            np.asarray(inp.kappa0_x, dtype=np.float64).ravel(),
            np.asarray(inp.kappa0_y, dtype=np.float64).ravel(),
            np.asarray(inp.kappa0_z, dtype=np.float64).ravel(),
        )
    )
    chord = np.asarray(inp.chord_m, dtype=np.float64).ravel()
    twist = np.asarray(inp.twist_deg, dtype=np.float64).ravel()

    airfoil_profiles: list[object] = []
    n_pts = int(naca_airfoil_n_points)
    for i in range(n):
        series = int(np.clip(int(round(float(inp.naca_series[i]))), 4, 6))
        airfoil_profiles.append(
            AirfoilSDF.from_naca_series(
                series,
                float(inp.naca_m[i]),
                float(inp.naca_p[i]),
                float(inp.naca_xx[i]),
                n_points=n_pts,
                chord=float(chord[i]),
                closed_te=True,
            )
        )
    if web_positions is None:
        if system_layout is None or not hasattr(system_layout, "web_chord_fracs"):
            raise ValueError(
                "web_positions not supplied and no system_layout provided to derive them from. "
                "Pass system_layout=orch.layout or an explicit web_positions sequence."
            )
        fracs = tuple(system_layout.web_chord_fracs)
        if len(fracs) != 2:
            raise ValueError(
                f"system_layout.web_chord_fracs has {len(fracs)} entries; "
                "build_optim_blade_geometry_from_spanwise currently requires exactly 2 webs."
            )
        wpos = np.asarray([f - 0.5 for f in fracs], dtype=np.float64)
    else:
        wpos = np.asarray(web_positions, dtype=np.float64).ravel()
        if wpos.size != 2:
            raise ValueError("web_positions must be two abscissae (e.g. -0.32, 0.32).")

    if box_height_frac is None:
        naca_xx_arr = np.asarray(inp.naca_xx, dtype=np.float64).ravel()
        box_height_frac = float(np.mean(naca_xx_arr) / 100.0)

    return OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=kappa0,
        chord=chord,
        twist=twist,
        airfoil_profiles=airfoil_profiles,
        web_positions=wpos,
        subcomponent_materials=subcomponent_materials,
        thickness_role=thickness_role,
        cap_shear_lag_width=cap_shear_lag_width,
        box_height_frac=float(box_height_frac),
        radial_r_m=rr,
    )