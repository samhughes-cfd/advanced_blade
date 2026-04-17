"""Parallel midsurface section solves for dirty station indices."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from blade_precompute.section_properties.engine.geometry import SectionDefinition
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver
from blade_precompute.section_properties.core.types import SectionSolveResult


def _solve_one_static(sd: SectionDefinition) -> SectionSolveResult:
    """Top-level picklable entry point for ``ProcessPoolExecutor`` (Windows spawn)."""
    return MidsurfaceSectionSolver().solve_one(sd)


def solve_dirty_stations(
    section_defs: list[SectionDefinition],
    dirty_indices: list[int],
    n_workers: int = 4,
) -> dict[int, SectionSolveResult]:
    if not dirty_indices:
        return {}
    if n_workers <= 1 or len(dirty_indices) == 1:
        return {i: _solve_one_static(section_defs[i]) for i in dirty_indices}
    out: dict[int, SectionSolveResult] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {i: ex.submit(_solve_one_static, section_defs[i]) for i in dirty_indices}
        for i, fut in futures.items():
            out[i] = fut.result()
    return out
