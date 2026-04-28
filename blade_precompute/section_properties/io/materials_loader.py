"""Shared helpers for orthotropic plies and laminate definitions."""

from __future__ import annotations

from typing import Any, List, Mapping

from ..engine.laminate import LaminateDefinition
from ..engine.materials import OrthotropicPly


def orthotropic_ply_from_dict(name: str, d: Mapping[str, Any]) -> OrthotropicPly:
    return OrthotropicPly(
        name=name,
        E1=float(d["E1"]),
        E2=float(d["E2"]),
        G12=float(d["G12"]),
        nu12=float(d["nu12"]),
        rho=float(d.get("rho", 1600.0)),
        t_ply=float(d["t_ply"]),
        Xt=float(d["Xt"]),
        Xc=float(d["Xc"]),
        Yt=float(d["Yt"]),
        Yc=float(d["Yc"]),
        S12=float(d["S12"]),
        Zt=float(d["Zt"]),
        S13=float(d["S13"]),
        S23=float(d["S23"]),
    )


def laminate_from_mapping_spec(
    spec: Mapping[str, Any],
    ply_lib: Mapping[str, Mapping[str, Any]],
    mat_key: str,
) -> LaminateDefinition:
    """Build :class:`LaminateDefinition` from ``layup`` and ``ply_type`` mapping fields."""
    layup: List[float] = [float(a) for a in spec["layup"]]
    shear = bool(spec.get("shear_lag_correction", True))
    pname = str(spec["ply_type"])
    if pname not in ply_lib:
        raise KeyError(f"Unknown ply_type '{pname}' for material '{mat_key}'")
    ply = orthotropic_ply_from_dict(pname, ply_lib[pname])
    plies = [(ply, float(ang)) for ang in layup]
    return LaminateDefinition(plies=plies, shear_lag_correction=shear)
