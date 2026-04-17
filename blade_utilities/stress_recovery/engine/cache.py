"""Shim: see :mod:`blade_utilities.recovery.tensor_cache.cache`."""

from blade_utilities.recovery.tensor_cache.cache import (
    CACHE_VERSION,
    NPZ_VERSION_KEY,
    RecoveryCache,
)

_NPZ_VERSION_KEY = NPZ_VERSION_KEY
_CACHE_VERSION = CACHE_VERSION

__all__ = ["RecoveryCache", "NPZ_VERSION_KEY", "CACHE_VERSION", "_NPZ_VERSION_KEY", "_CACHE_VERSION"]
