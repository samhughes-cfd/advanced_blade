"""YAML and external-result adapters."""

from .external_results import ExternalSectionResultSolver, section_result_from_mapping
from .yaml_loader import load_section_from_yaml

__all__ = ["load_section_from_yaml", "section_result_from_mapping", "ExternalSectionResultSolver"]
