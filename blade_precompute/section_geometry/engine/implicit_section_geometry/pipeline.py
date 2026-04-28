"""High-level build/evaluate pipeline for implicit section geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .builder import MultiCellSection
from ..eval_cache import SectionEvalCache
from .grid import SDFGrid
from .medial import MedialAxisExtractor
from ...interface.export import SectionPropertiesReport


@dataclass
class ImplicitSectionBuildResult:
    """Bundled outputs from a one-shot section geometry run."""

    section: MultiCellSection
    grid: SDFGrid
    properties: SectionPropertiesReport
    midlines: Dict[str, List]


def build_section_pipeline(section: MultiCellSection, grid: SDFGrid) -> ImplicitSectionBuildResult:
    """Run properties and medial extraction for a section on a grid."""
    eval_cache = SectionEvalCache()
    properties = SectionPropertiesReport(section, grid, eval_cache=eval_cache)
    midlines = MedialAxisExtractor(grid).extract_for_section(section, eval_cache=eval_cache)
    return ImplicitSectionBuildResult(
        section=section,
        grid=grid,
        properties=properties,
        midlines=midlines,
    )


__all__ = ["ImplicitSectionBuildResult", "build_section_pipeline"]
