"""Public façade for geometrically exact beam static analysis (Tier A)."""

from __future__ import annotations

from ..engine.blade_geometry import BladeGeometry, beam_model_from_blade_geometry
from ..engine.solver import solve_static
from ..core.types import BeamLoads, BeamModel, BeamSolveResult, SectionStation, SolverOptions


class BeamAnalysis:
    """Holds a :class:`BeamModel` and exposes ``solve_static`` (Tier A)."""

    def __init__(self, model: BeamModel) -> None:
        self.model = model

    @classmethod
    def from_blade_geometry(
        cls,
        geometry: BladeGeometry,
        n_nodes: int,
        section_stations: list[SectionStation],
        *,
        span_axis: int = 2,
    ) -> BeamAnalysis:
        """Build from :class:`~global_beam_model.engine.blade_geometry.BladeGeometry` (see :mod:`global_beam_model.core.tier_paths`)."""
        m = beam_model_from_blade_geometry(
            geometry, n_nodes, section_stations, span_axis=span_axis
        )
        return cls(m)

    def solve_static(
        self,
        loads: BeamLoads,
        *,
        options: SolverOptions | None = None,
    ) -> BeamSolveResult:
        """Delegate to :func:`~global_beam_model.engine.solver.solve_static`."""
        opt = options if options is not None else SolverOptions()
        return solve_static(self.model, loads, options=opt)
