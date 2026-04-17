"""Station-level cache helpers."""

from __future__ import annotations

from ..core.types import DesignVector, StationCache


def init_station_caches(n_station: int) -> list[StationCache]:
    """Caches start dirty with zero thickness stamps (always stale until first solve)."""
    return [
        StationCache(t_skin=-1.0, t_cap=-1.0, t_web=-1.0, result=None, dirty=True)
        for _ in range(n_station)
    ]


def dirty_indices(dv: DesignVector, caches: list[StationCache], tol: float = 1e-9) -> list[int]:
    return [i for i in range(len(caches)) if caches[i].is_stale(dv, i, tol)]
