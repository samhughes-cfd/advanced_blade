"""NPZ serialisation for :class:`~recovery_cache.engine.cache.RecoveryCache`."""

from __future__ import annotations

import numpy as np

from ..engine.cache import RecoveryCache, _NPZ_VERSION_KEY


def save_cache(cache: RecoveryCache, path: str) -> None:
    np.savez_compressed(path, **cache.to_dict())


def load_cache(path: str) -> RecoveryCache:
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)
        d = {k: np.asarray(z[k]) for k in keys}
    d.pop(_NPZ_VERSION_KEY, None)
    return RecoveryCache.from_dict(d)
