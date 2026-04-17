"""Analytical blade mass from strip geometry and material densities."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_properties.engine.elements import build_strip_fe_data
from blade_precompute.section_properties.engine.mesh import build_line_mesh
from blade_precompute.section_properties.engine.section_properties import mass_per_length

from .section_builder import SectionBuilder
from ..core.types import DesignVector, OptimBladeGeometry


def mass_objective(dv: DesignVector, blade_geometry: OptimBladeGeometry) -> float:
    """
    Integrate spanwise ``mass_per_length`` (trapezoidal rule on ``z_stations``).

    Uses ``LaminateDefinition.equivalent_density()`` for composites and
    ``IsotropicMaterial.rho`` for metals, consistent with ``section_model``.
    """
    sections = SectionBuilder.build(dv, blade_geometry)
    z = np.asarray(blade_geometry.z_stations, dtype=np.float64)
    mu = np.zeros(z.shape[0], dtype=np.float64)
    for i, sec in enumerate(sections):
        mesh = build_line_mesh(sec)
        fe = build_strip_fe_data(sec, mesh)
        mu[i] = mass_per_length(sec, fe)
    if z.size == 1:
        return float(mu[0] * 1.0)
    dz = np.diff(z)
    return float(np.sum(0.5 * (mu[:-1] + mu[1:]) * dz))
