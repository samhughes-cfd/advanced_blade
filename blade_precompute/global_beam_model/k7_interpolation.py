"""PCHIP spanwise interpolation of tabulated ``K7`` / ``K6`` (no GBT imports; avoids import cycles)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from .core.types import K7Array


def _symmetrize_k7_batch(entries: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrise ``(n, 7, 7)`` matrices."""
    e = np.asarray(entries, dtype=np.float64)
    return 0.5 * (e + np.transpose(e, (0, 2, 1)))


class K6Interpolator:
    """
    PCHIP interpolation of ``(n_stations, 6, 6)`` stiffness matrices onto arbitrary span coords.

    Each upper-triangular entry ``(i, j)``, ``i <= j``, is interpolated independently;
    the result is symmetrised.  Mirrors :class:`K7Interpolator` for the 6×6 case.
    """

    def __init__(self, s: NDArray[np.float64], k6_stack: NDArray[np.float64]) -> None:
        s = np.asarray(s, dtype=np.float64).ravel()
        ent = np.asarray(k6_stack, dtype=np.float64)
        if ent.ndim == 2:
            ent = ent[None, ...]
        n = int(s.size)
        if n < 1 or ent.shape[0] != n or ent.shape[1:] != (6, 6):
            raise ValueError(f"K6Interpolator: s has {n} pts but k6_stack shape={ent.shape}.")
        self._s = s
        self._single: NDArray[np.float64] | None = None
        self._interp: dict[tuple[int, int], PchipInterpolator] | None = None
        if n == 1:
            self._single = ent[0].copy()
        else:
            self._interp = {}
            for i in range(6):
                for j in range(i, 6):
                    y = ent[:, i, j].astype(np.float64, copy=False)
                    self._interp[(i, j)] = PchipInterpolator(s, y, extrapolate=False)

    def interpolate(
        self,
        query_points: NDArray[np.float64] | Sequence[float],
        *,
        allow_extrapolation: bool = True,
    ) -> NDArray[np.float64]:
        """Return ``(n_query, 6, 6)`` PCHIP-interpolated matrices."""
        zq = np.asarray(query_points, dtype=np.float64).ravel()
        s0, s1 = float(self._s[0]), float(self._s[-1])
        zq_eval = np.clip(zq, s0, s1) if allow_extrapolation else zq
        nq = int(zq_eval.size)
        out = np.zeros((nq, 6, 6), dtype=np.float64)
        if self._interp is None:
            assert self._single is not None
            out[:] = self._single
            return out
        for i in range(6):
            for j in range(i, 6):
                vals = np.asarray(self._interp[(i, j)](zq_eval), dtype=np.float64).ravel()
                out[:, i, j] = vals
                if i != j:
                    out[:, j, i] = vals
        return 0.5 * (out + np.transpose(out, (0, 2, 1)))


class K7Interpolator:
    """
    PCHIP interpolation of :class:`K7Array` onto arbitrary span coordinates.

    Each upper-triangular entry ``(i, j)``, ``i <= j``, is interpolated independently;
    the result is symmetrised.
    """

    def __init__(self, k7_array: K7Array) -> None:
        s = np.asarray(k7_array.s, dtype=np.float64).ravel()
        ent = np.asarray(k7_array.entries, dtype=np.float64)
        n = int(s.size)
        if n < 1:
            raise ValueError("K7Array must contain at least one station.")
        self._s = s
        self._single: NDArray[np.float64] | None
        self._interp: dict[tuple[int, int], PchipInterpolator] | None
        if n == 1:
            self._single = ent[0].copy()
            self._interp = None
        else:
            self._single = None
            self._interp = {}
            for i in range(7):
                for j in range(i, 7):
                    y = ent[:, i, j].astype(np.float64, copy=False)
                    self._interp[(i, j)] = PchipInterpolator(s, y, extrapolate=False)

    def interpolate(
        self,
        query_points: NDArray[np.float64] | Sequence[float],
        *,
        allow_extrapolation: bool = False,
    ) -> K7Array:
        zq = np.asarray(query_points, dtype=np.float64).ravel()
        s0, s1 = float(self._s[0]), float(self._s[-1])
        if not allow_extrapolation and zq.size > 0:
            if float(zq.min()) < s0 - 1e-9 or float(zq.max()) > s1 + 1e-9:
                raise ValueError(
                    f"query_points must lie within [{s0}, {s1}] (set allow_extrapolation=True to clamp)."
                )
        zq_eval = np.clip(zq, s0, s1) if allow_extrapolation else zq
        nq = int(zq_eval.size)
        out = np.zeros((nq, 7, 7), dtype=np.float64)

        if self._interp is None:
            assert self._single is not None
            out[:] = self._single
            return K7Array(s=zq.copy(), entries=out)

        for i in range(7):
            for j in range(i, 7):
                ip = self._interp[(i, j)]
                try:
                    vals = np.asarray(ip(zq_eval), dtype=np.float64).ravel()
                except ValueError as exc:
                    raise ValueError(
                        "K7 interpolation failed; check query_points within station range."
                    ) from exc
                out[:, i, j] = vals
                if i != j:
                    out[:, j, i] = vals

        out = _symmetrize_k7_batch(out)
        return K7Array(s=zq.copy(), entries=out)
