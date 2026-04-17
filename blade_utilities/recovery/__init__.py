"""
Fused beam stress/strain recovery (tensor cache + runtime operators + I/O).

This package consolidates what previously lived under ``stress_recovery`` and
``recovery_operators``. Those top-level modules remain as thin import shims.

Dependency policy (pragmatic): recovery builds on section homogenisation outputs
from ``blade_precompute.section_properties`` (``SectionSolveResult``, CLPT bases,
failure criteria, interlaminar helpers). ``blade_precompute`` beam workflows import
recovery here; recovery does not import ``global_beam_model``.
"""

from __future__ import annotations

from .api import RecoveryCacheBuilder
from .core.cache_types import (
    RecoveryCacheProtocol,
    RecoveryCacheStorage,
    RecoveryEvaluatorProtocol,
)
from .core.transforms import plane_stress_voigt_from_R
from .io.persistence import load_cache, save_cache
from .operators.apply import (
    apply_interlaminar_transfer,
    apply_section_stress_operator,
    apply_span_derivative,
    apply_strain_operator,
)
from .operators.builder import build_recovery_operator_bundle
from .operators.types import RecoveryOperatorBundle, RecoveryOperatorBundleProtocol
from .tensor_cache.builder import build_recovery_cache
from .tensor_cache.cache import RecoveryCache

__all__ = [
    "RecoveryCache",
    "RecoveryCacheBuilder",
    "RecoveryCacheProtocol",
    "RecoveryCacheStorage",
    "RecoveryEvaluatorProtocol",
    "RecoveryOperatorBundle",
    "RecoveryOperatorBundleProtocol",
    "apply_interlaminar_transfer",
    "apply_section_stress_operator",
    "apply_span_derivative",
    "apply_strain_operator",
    "build_recovery_cache",
    "build_recovery_operator_bundle",
    "load_cache",
    "plane_stress_voigt_from_R",
    "save_cache",
]
