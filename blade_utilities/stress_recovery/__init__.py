"""Precomputed fused recovery operators for composite and isotropic subcomponents.

Run: ``python -m recovery_cache``
"""

from .api import RecoveryCacheBuilder
from .core.types import RecoveryCacheProtocol, RecoveryCacheStorage, RecoveryEvaluatorProtocol
from .engine.builder import build_recovery_cache, plane_stress_voigt_from_R
from .engine.cache import RecoveryCache
from .io.persistence import load_cache, save_cache

__all__ = [
    "RecoveryCache",
    "RecoveryCacheBuilder",
    "RecoveryCacheProtocol",
    "RecoveryEvaluatorProtocol",
    "RecoveryCacheStorage",
    "build_recovery_cache",
    "load_cache",
    "plane_stress_voigt_from_R",
    "save_cache",
]
