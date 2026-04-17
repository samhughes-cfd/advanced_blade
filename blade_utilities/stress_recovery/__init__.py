"""
Precomputed fused recovery operators (shim).

Canonical imports: ``blade_utilities.recovery``. This package re-exports the same
public API for backward compatibility.
"""

from blade_utilities.recovery import (
    RecoveryCache,
    RecoveryCacheBuilder,
    RecoveryCacheProtocol,
    RecoveryCacheStorage,
    RecoveryEvaluatorProtocol,
    build_recovery_cache,
    load_cache,
    plane_stress_voigt_from_R,
    save_cache,
)

__all__ = [
    "RecoveryCache",
    "RecoveryCacheBuilder",
    "RecoveryCacheProtocol",
    "RecoveryCacheStorage",
    "RecoveryEvaluatorProtocol",
    "build_recovery_cache",
    "load_cache",
    "plane_stress_voigt_from_R",
    "save_cache",
]
