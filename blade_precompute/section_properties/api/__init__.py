"""Public façade for section YAML load + solve (consistent verb vocabulary)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.types import SectionSolveResult, SectionSolverProtocol
from ..engine.elements import build_strip_fe_data
from ..engine.geometry import SectionDefinition
from ..engine.implicit_section_geometry import GeometryConstraintSpec, build_section_from_constraints
from ..engine.mesh import build_line_mesh, subcomponents_by_type
from ..engine.solver import MidsurfaceSectionSolver
from ..io.yaml_loader import load_section_from_yaml


@dataclass
class AnalysisConfig:
    """
    Optional post-processing after a midsurface section solve.

    Core stiffness always uses :class:`MidsurfaceSectionSolver` (or a custom
    :class:`SectionSolverProtocol`). Interlaminar and panel checks rebuild the
    strip mesh locally so they stay consistent with ``fe``/``mesh`` conventions.
    """

    run_interlaminar: bool = False
    run_panel_buckling: bool = False
    frame_spacing_m: float = 0.5
    verbose: bool = False
    merge_tolerance: float = 1e-6
    #: Optional ``(6,)`` vector ``[N, My, Mz, T, Vy, Vz]`` for CLPT-based panel stress peaks (``K6`` order).
    reference_sectional_forces: NDArray[np.float64] | None = None
    interlaminar_vy: float = 0.0
    interlaminar_vz: float = 1.0
    use_strip_interlaminar_equilibrium: bool = False


def _composite_representative_edges(section: SectionDefinition, fe) -> tuple[list[int], list]:
    """One midsurface edge index and laminate per composite subcomponent."""
    from ..engine.laminate import LaminateDefinition

    comp_si, _ = subcomponents_by_type(section)
    edge_indices: list[int] = []
    lams: list = []
    for si in comp_si:
        e0 = next(e for e in range(fe.n_edges) if int(fe.subcomp_idx[e]) == si)
        edge_indices.append(e0)
        mat = section.subcomponents[si].material
        assert isinstance(mat, LaminateDefinition)
        lams.append(mat)
    return edge_indices, lams


class SectionAnalysis:
    """One entry object for ``load`` → ``solve`` on midsurface sections."""

    def __init__(
        self,
        solver: SectionSolverProtocol | None = None,
        *,
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        self._solver: SectionSolverProtocol = solver or MidsurfaceSectionSolver()
        self._config = config or AnalysisConfig()

    @staticmethod
    def load(path: str | Path) -> SectionDefinition:
        """Parse YAML into :class:`~section_model.engine.geometry.SectionDefinition`."""
        return load_section_from_yaml(path)

    def solve(
        self,
        section: SectionDefinition,
        *,
        panel_frame_spacing_m: float | None = None,
        panel_reference_forces_6: NDArray[np.float64] | None = None,
    ) -> SectionSolveResult:
        """Run the configured section solver and optional post-processing."""
        result = self._solver.solve_one(section)

        if self._config.verbose:
            self._print_coupling_notice(result)

        mesh = None
        fe = None
        if self._config.run_interlaminar or self._config.run_panel_buckling:
            mesh = build_line_mesh(section, self._config.merge_tolerance)
            fe = build_strip_fe_data(section, mesh)

        if self._config.run_interlaminar:
            if fe is None or mesh is None:
                raise RuntimeError("interlaminar post-processing requires strip mesh data.")
            result = self._attach_interlaminar(result, section, fe, mesh)

        if self._config.run_panel_buckling:
            if fe is None:
                raise RuntimeError("panel buckling post-processing requires strip mesh data.")
            result = self._attach_buckling(
                result,
                section,
                fe,
                frame_spacing_m=panel_frame_spacing_m,
                reference_forces_6=panel_reference_forces_6,
            )

        return result

    def _print_coupling_notice(self, result: SectionSolveResult) -> None:
        k = result.K6
        denom = (abs(k[1, 1]) * abs(k[3, 3])) ** 0.5
        if denom > 1e-30:
            eta = abs(k[1, 3]) / denom
            if eta > 0.01:
                print(
                    f"section_properties: bend–twist coupling eta = {eta:.4f} "
                    f"(|K[My,T]| / sqrt(K[My,My]*K[T,T]))"
                )
        if abs(result.k_z - 5.0 / 6.0) > 0.05:
            print(
                f"section_properties: energy-consistent k_z = {result.k_z:.4f} "
                f"(isotropic default 5/6 = {5.0 / 6.0:.4f})"
            )

    def _attach_interlaminar(
        self,
        result: SectionSolveResult,
        section: SectionDefinition,
        fe,
        mesh,
    ) -> SectionSolveResult:
        from ..engine.interlaminar_recovery import recover_interlaminar
        from ..engine.strip_shear_equilibrium import recover_interlaminar_strip_equilibrium

        comp_edges, lams = _composite_representative_edges(section, fe)
        if not lams:
            return result

        eiy = float(result.K6[1, 1])
        eiz = float(result.K6[2, 2])
        if self._config.use_strip_interlaminar_equilibrium:
            il_result, _ = recover_interlaminar_strip_equilibrium(
                comp_edges,
                lams,
                mesh,
                fe,
                section,
                result,
                vy=float(self._config.interlaminar_vy),
                vz=float(self._config.interlaminar_vz),
                eiy=eiy,
                eiz=eiz,
            )
        else:
            il_result = recover_interlaminar(
                comp_edge_indices=comp_edges,
                lams=lams,
                vy=float(self._config.interlaminar_vy),
                vz=float(self._config.interlaminar_vz),
                eiy=eiy,
                eiz=eiz,
            )
        result.interlaminar = il_result

        if self._config.verbose and il_result.IFI_global > 0.5:
            print(
                f"section_properties: IFI_global = {il_result.IFI_global:.3f} "
                f"(edge {il_result.critical_edge}, z = {il_result.critical_z * 1e3:.2f} mm)"
            )
        return result

    def _attach_buckling(
        self,
        result: SectionSolveResult,
        section: SectionDefinition,
        fe,
        *,
        frame_spacing_m: float | None,
        reference_forces_6: NDArray[np.float64] | None,
    ) -> SectionSolveResult:
        from ..engine.panel_buckling import assess_panel_buckling_section, composite_edge_panel_stresses_from_reference

        comp_edges, lams = _composite_representative_edges(section, fe)
        if not lams:
            return result

        n = len(lams)
        fs = float(self._config.frame_spacing_m if frame_spacing_m is None else frame_spacing_m)
        fref = reference_forces_6
        if fref is None:
            fref = self._config.reference_sectional_forces
        if fref is not None:
            peaks = composite_edge_panel_stresses_from_reference(result, fref)
            sigma_zz = np.array([peaks[i, 0] for i in range(n)], dtype=np.float64)
            tau = np.array([peaks[i, 1] for i in range(n)], dtype=np.float64)
            sigma_yy = np.array([peaks[i, 2] for i in range(n)], dtype=np.float64)
        else:
            sigma_zz = np.zeros(n, dtype=np.float64)
            tau = np.zeros(n, dtype=np.float64)
            sigma_yy = np.zeros(n, dtype=np.float64)

        bk = assess_panel_buckling_section(
            fe=fe,
            comp_edge_indices=comp_edges,
            lams=lams,
            sigma_zz=sigma_zz,
            tau=tau,
            frame_spacing_m=fs,
            sigma_yy=sigma_yy,
        )
        result.panel_buckling = bk

        if self._config.verbose and bk.n_buckled > 0:
            print(
                f"section_properties: local panel buckling — {bk.n_buckled} panel(s), "
                f"BI_max = {bk.BI_max:.3f} (edge {bk.critical_edge})"
            )
        return result

    @staticmethod
    def from_constraints(spec: GeometryConstraintSpec) -> SectionDefinition:
        """Build a constrained implicit-geometry section in S frame."""
        return build_section_from_constraints(spec).section

    def load_and_solve(self, path: str | Path) -> SectionSolveResult:
        """``load`` then ``solve``."""
        return self.solve(self.load(path))

    def solve_constraints(self, spec: GeometryConstraintSpec) -> SectionSolveResult:
        """Build constrained section then solve with configured solver."""
        return self.solve(self.from_constraints(spec))
