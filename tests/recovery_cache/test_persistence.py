"""Recovery cache persistence regressions."""

from __future__ import annotations

import numpy as np

from blade_utilities.recovery import RecoveryCache, load_cache, save_cache


def _minimal_cache() -> RecoveryCache:
    return RecoveryCache(
        L_rec=np.zeros((1, 1, 1, 3, 7), dtype=np.float64),
        L_iso=np.zeros((1, 1, 3, 7), dtype=np.float64),
        Xt=np.ones((1, 1), dtype=np.float64),
        Xc=np.ones((1, 1), dtype=np.float64),
        Yt=np.ones((1, 1), dtype=np.float64),
        Yc=np.ones((1, 1), dtype=np.float64),
        S12=np.ones((1, 1), dtype=np.float64),
        sigma_allow_iso=np.ones(1, dtype=np.float64),
        Zt=np.ones((1, 1), dtype=np.float64),
        S13=np.ones((1, 1), dtype=np.float64),
        S23=np.ones((1, 1), dtype=np.float64),
        spanwise_dz=np.ones(1, dtype=np.float64),
        z_stations=np.zeros(1, dtype=np.float64),
        z_ply_ref=np.zeros((1, 1), dtype=np.float64),
        composite_subcomp_idx=[0],
        isotropic_subcomp_idx=[0],
        composite_subcomp_names=["skin"],
        isotropic_subcomp_names=["web"],
        ply_count=np.ones((1, 1), dtype=np.int32),
        K7=np.eye(7, dtype=np.float64)[None, :, :],
        K6=np.eye(6, dtype=np.float64)[None, :, :],
        M6=np.eye(6, dtype=np.float64)[None, :, :],
        shear_center=np.zeros((1, 2), dtype=np.float64),
        mass_center=np.zeros((1, 2), dtype=np.float64),
    )


def test_save_then_load_cache_preserves_versioned_npz(tmp_path):
    path = tmp_path / "cache.npz"
    cache = _minimal_cache()

    save_cache(cache, str(path))
    loaded = load_cache(str(path))

    np.testing.assert_allclose(loaded.L_rec, cache.L_rec)
    assert loaded.composite_subcomp_names == ["skin"]
