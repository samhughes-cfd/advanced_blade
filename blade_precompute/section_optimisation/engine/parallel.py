"""Parallel midsurface section solves for dirty station indices."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    *,
    on_done: Callable[[int, SectionSolveResult], None] | None = None,
) -> dict[int, SectionSolveResult]:
    if not dirty_indices:
        return {}
    n = len(dirty_indices)
    if n_workers <= 1 or n == 1:
        out: dict[int, SectionSolveResult] = {}
        for i in dirty_indices:
            r = _solve_one_static(section_defs[i])
            out[i] = r
            if on_done is not None:
                on_done(i, r)
        return out
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_solve_one_static, section_defs[i]): i for i in dirty_indices}
        out: dict[int, SectionSolveResult] = {}
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            out[i] = res
            if on_done is not None:
                on_done(i, res)
    return out
