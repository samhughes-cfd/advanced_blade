"""NPZ serialisation for :class:`~blade_utilities.recovery.tensor_cache.cache.RecoveryCache`."""

from __future__ import annotations

import numpy as np

from blade_utilities.recovery.tensor_cache.cache import NPZ_VERSION_KEY, RecoveryCache


def save_cache(cache: RecoveryCache, path: str) -> None:
    np.savez_compressed(path, **cache.to_dict())


def load_cache(path: str) -> RecoveryCache:
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)
        d = {k: np.asarray(z[k]) for k in keys}
    d.pop(NPZ_VERSION_KEY, None)
    return RecoveryCache.from_dict(d)
