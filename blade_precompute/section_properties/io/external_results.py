"""
Adapters for externally homogenised section properties.

This allows loading precomputed section stiffness/results (for example from
BECAS/VABS or shell homogenisation workflows) into the existing
``SectionSolverProtocol`` flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.types import SectionSolveResult, SectionSolverProtocol


def _arr(d: Mapping[str, Any], key: str, shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    v = d.get(key)
    if v is None:
        return np.full(shape, default, dtype=np.float64)
    a = np.asarray(v, dtype=np.float64)
    return a.reshape(shape)


def section_result_from_mapping(data: Mapping[str, Any]) -> SectionSolveResult:
    """
    Build :class:`SectionSolveResult` from a plain mapping.

    Required keys:
    - ``K6`` (6x6)
    - ``K7`` (7x7)

    Optional keys default to zeros/empty arrays where practical.
    """
    K6 = _arr(data, "K6", (6, 6))
    K7 = _arr(data, "K7", (7, 7))
    M6 = _arr(data, "M6", (6, 6))
    n_comp = int(data.get("n_composite", 0))
    n_iso = int(data.get("n_isotropic", 0))
    n_ply_max = int(data.get("n_ply_max", 1))
    return SectionSolveResult(
        K7=K7,
        K6=K6,
        M6=M6,
        warping_function=np.asarray(data.get("warping_function", [0.0]), dtype=np.float64).ravel(),
        K_ww=float(data.get("K_ww", K7[6, 6])),
        k_w=np.asarray(data.get("k_w", np.zeros(6)), dtype=np.float64).reshape(6),
        composite_resultant_basis=_arr(data, "composite_resultant_basis", (n_comp, 7, 6), 0.0),
        isotropic_resultant_basis=_arr(data, "isotropic_resultant_basis", (n_iso, 7, 3), 0.0),
        composite_subcomp_names=list(data.get("composite_subcomp_names", [])),
        isotropic_subcomp_names=list(data.get("isotropic_subcomp_names", [])),
        ABD_inv=_arr(data, "ABD_inv", (n_comp, 3, 3), 0.0),
        Q_bar=_arr(data, "Q_bar", (n_comp, n_ply_max, 3, 3), 0.0),
        T_ply=_arr(data, "T_ply", (n_comp, n_ply_max, 3, 3), 0.0),
        z_ply=_arr(data, "z_ply", (n_comp, n_ply_max + 1), 0.0),
        iso_thickness=np.asarray(data.get("iso_thickness", np.zeros(n_iso)), dtype=np.float64).reshape(n_iso),
        iso_C=_arr(data, "iso_C", (n_iso, 3, 3), 0.0),
        iso_sigma_allow=np.asarray(data.get("iso_sigma_allow", np.zeros(n_iso)), dtype=np.float64).reshape(n_iso),
        Zt=_arr(data, "Zt", (n_comp, n_ply_max), 0.0),
        S13=_arr(data, "S13", (n_comp, n_ply_max), 0.0),
        S23=_arr(data, "S23", (n_comp, n_ply_max), 0.0),
        area=float(data.get("area", 0.0)),
        mass_per_length=float(data.get("mass_per_length", 0.0)),
        shear_center=np.asarray(data.get("shear_center", [0.0, 0.0]), dtype=np.float64).reshape(2),
        mass_center=np.asarray(data.get("mass_center", [0.0, 0.0]), dtype=np.float64).reshape(2),
        elastic_center=np.asarray(data.get("elastic_center", [0.0, 0.0]), dtype=np.float64).reshape(2),
        E_omega_basis=np.asarray(data["E_omega_basis"], dtype=np.float64)
        if "E_omega_basis" in data
        else None,
    )


@dataclass
class ExternalSectionResultSolver(SectionSolverProtocol):
    """
    ``SectionSolverProtocol`` adapter backed by precomputed station results.

    ``station_results`` maps station ``z`` to a plain mapping accepted by
    :func:`section_result_from_mapping`.
    """

    station_results: dict[float, Mapping[str, Any]]

    @classmethod
    def from_npz(cls, path: str | Path) -> "ExternalSectionResultSolver":
        """
        Build solver from an ``.npz`` bundle.

        The bundle must include:
        - ``z_stations``: (n,)
        - ``K6_stack``: (n,6,6)
        - ``K7_stack``: (n,7,7)
        Optional arrays include ``M6_stack``, centres, area, mass_per_length.
        """
        p = Path(path)
        raw = np.load(p, allow_pickle=False)
        z = np.asarray(raw["z_stations"], dtype=np.float64).ravel()
        K6 = np.asarray(raw["K6_stack"], dtype=np.float64)
        K7 = np.asarray(raw["K7_stack"], dtype=np.float64)
        if z.shape[0] != K6.shape[0] or z.shape[0] != K7.shape[0]:
            raise ValueError("z_stations, K6_stack, K7_stack must share first dimension.")
        out: dict[float, Mapping[str, Any]] = {}
        for i in range(z.shape[0]):
            item: dict[str, Any] = {"K6": K6[i], "K7": K7[i]}
            if "M6_stack" in raw:
                item["M6"] = np.asarray(raw["M6_stack"], dtype=np.float64)[i]
            if "shear_center" in raw:
                item["shear_center"] = np.asarray(raw["shear_center"], dtype=np.float64)[i]
            if "mass_center" in raw:
                item["mass_center"] = np.asarray(raw["mass_center"], dtype=np.float64)[i]
            if "elastic_center" in raw:
                item["elastic_center"] = np.asarray(raw["elastic_center"], dtype=np.float64)[i]
            if "area" in raw:
                item["area"] = float(np.asarray(raw["area"], dtype=np.float64)[i])
            if "mass_per_length" in raw:
                item["mass_per_length"] = float(np.asarray(raw["mass_per_length"], dtype=np.float64)[i])
            out[float(z[i])] = item
        return cls(station_results=out)

    def _lookup(self, station_z: float) -> Mapping[str, Any]:
        if not self.station_results:
            raise ValueError("ExternalSectionResultSolver has no station results.")
        keys = np.asarray(sorted(self.station_results.keys()), dtype=np.float64)
        idx = int(np.argmin(np.abs(keys - float(station_z))))
        return self.station_results[float(keys[idx])]

    def solve_one(self, section_def: object) -> SectionSolveResult:
        station_z = float(getattr(section_def, "station_z", 0.0))
        return section_result_from_mapping(self._lookup(station_z))

    def solve(self, section_defs: list[object]) -> list[SectionSolveResult]:
        return [self.solve_one(s) for s in section_defs]
