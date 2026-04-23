"""YAML and external-result adapters."""

from .external_results import ExternalSectionResultSolver, section_result_from_mapping
from .section_solve_bundle import save_section_solve_stations_bundle
from .yaml_loader import load_section_from_yaml

__all__ = [
    "ExternalSectionResultSolver",
    "load_section_from_yaml",
    "save_section_solve_stations_bundle",
    "section_result_from_mapping",
]
