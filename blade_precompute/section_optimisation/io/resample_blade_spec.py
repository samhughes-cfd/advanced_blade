"""Resample spanwise blade arrays in a blade mapping spec to a new station count."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def resample_blade_spec(
    raw: Mapping[str, Any],
    *,
    n_stations: int,
) -> dict[str, Any]:
    """
    Build a new document with ``blade.z_stations`` (and parallel arrays) on a uniform grid.

    Numeric fields are linearly interpolated in ``z``; ``airfoil_profiles`` use the profile
    from the nearest source station. ``ply_library`` and ``blade.subcomponents`` are copied
    unchanged.
    """
    if n_stations < 1:
        raise ValueError("n_stations must be >= 1.")
    blade = raw.get("blade")
    if not isinstance(blade, dict):
        raise ValueError("Spec must contain a mapping 'blade:'.")

    z_src = np.asarray(blade["z_stations"], dtype=np.float64).ravel()
    if z_src.size < 2 and n_stations > 1:
        raise ValueError("Need at least two source z_stations to interpolate.")
    z_min = float(z_src[0])
    z_max = float(z_src[-1])
    if n_stations == 1:
        z_new = np.array([z_min], dtype=np.float64)
    else:
        z_new = np.linspace(z_min, z_max, n_stations)

    def interp_1d(values: Any) -> NDArray[np.float64]:
        a = np.asarray(values, dtype=np.float64).ravel()
        if a.shape[0] != z_src.shape[0]:
            raise ValueError("Spanwise array length must match z_stations.")
        return np.interp(z_new, z_src, a)

    r_ref = np.asarray(blade["r_ref"], dtype=np.float64)
    if r_ref.ndim != 2 or r_ref.shape[1] != 3:
        raise ValueError("r_ref must have shape (n_stations, 3).")
    new_r = np.column_stack(
        [interp_1d(r_ref[:, j]) for j in range(3)],
    )
    new_r[:, 2] = z_new

    kappa0 = np.asarray(blade["kappa0"], dtype=np.float64)
    if kappa0.ndim != 2 or kappa0.shape[1] != 3:
        raise ValueError("kappa0 must have shape (n_stations, 3).")
    new_kappa = np.column_stack([interp_1d(kappa0[:, j]) for j in range(3)])

    new_chord = interp_1d(blade["chord"])
    new_twist = interp_1d(blade["twist"])

    airfoils_src = list(blade.get("airfoil_profiles", []))
    if len(airfoils_src) != z_src.shape[0]:
        raise ValueError("airfoil_profiles length must match z_stations.")
    new_airfoils: list[str] = []
    for zn in z_new:
        j = int(np.argmin(np.abs(z_src - float(zn))))
        new_airfoils.append(str(airfoils_src[j]))

    blade_out = dict(blade)
    blade_out["z_stations"] = [float(x) for x in z_new]
    blade_out["r_ref"] = new_r.tolist()
    blade_out["kappa0"] = new_kappa.tolist()
    blade_out["chord"] = [float(x) for x in new_chord]
    blade_out["twist"] = [float(x) for x in new_twist]
    blade_out["airfoil_profiles"] = new_airfoils

    out: dict[str, Any] = {}
    if "ply_library" in raw:
        out["ply_library"] = raw["ply_library"]
    out["blade"] = blade_out
    return out
