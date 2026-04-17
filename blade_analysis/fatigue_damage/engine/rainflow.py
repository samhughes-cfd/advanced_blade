"""
Rainflow counting on ply stress and isotropic von Mises histories.

Rainflow is applied only to scalar stress signals — never to beam resultants.
Composite and isotropic paths are separate vectorised layouts after per-signal extraction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import rainflow

from blade_utilities.recovery import RecoveryCache

from ..core.loads import StressHistory
from .stress_range import ply_stress_component, von_mises_plane_stress
from ..core.types import RainflowBins


def _cycles_to_bins(
    ranges: NDArray[np.float64],
    means: NDArray[np.float64],
    counts: NDArray[np.float64],
    range_edges: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate rainflow cycles into fixed stress-range bins (mean is count-weighted)."""
    n_bin = int(range_edges.shape[0]) - 1
    if ranges.size == 0:
        z = np.zeros(n_bin, dtype=np.float64)
        return 0.5 * (range_edges[:-1] + range_edges[1:]), z, z

    idx = np.digitize(ranges, range_edges) - 1
    idx = np.clip(idx, 0, n_bin - 1)
    w = np.asarray(counts, dtype=np.float64)
    sum_w = np.bincount(idx, weights=w, minlength=n_bin).astype(np.float64)
    sum_wm = np.bincount(idx, weights=w * np.asarray(means, dtype=np.float64), minlength=n_bin).astype(np.float64)
    centers = 0.5 * (range_edges[:-1] + range_edges[1:])
    mean_bin = np.divide(
        sum_wm,
        sum_w,
        out=np.zeros_like(sum_w),
        where=sum_w > 0.0,
    )
    return centers.astype(np.float64), mean_bin, sum_w


def _range_edges_from_series(series_list: list[NDArray[np.float64]], n_range_bins: int) -> NDArray[np.float64]:
    """Shared linear bin edges ``[0, S_max]`` from peak-to-peak of provided 1D series."""
    ptps = [float(np.ptp(np.asarray(s, dtype=np.float64))) for s in series_list if s.size > 1]
    s_max = max(ptps) if ptps else 1.0
    s_max = max(s_max, 1e-12)
    return np.linspace(0.0, s_max, n_range_bins + 1, dtype=np.float64)


def _extract_cycles_arrays(y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rs: list[float] = []
    ms: list[float] = []
    cs: list[float] = []
    for rng, mean, count, _i0, _i1 in rainflow.extract_cycles(np.asarray(y, dtype=np.float64)):
        rs.append(float(rng))
        ms.append(float(mean))
        cs.append(float(count))
    if not rs:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(rs, dtype=np.float64),
        np.asarray(ms, dtype=np.float64),
        np.asarray(cs, dtype=np.float64),
    )


def count_cycles_ply_stresses(
    stress_history: StressHistory,
    component: int = 0,
    n_range_bins: int = 128,
    range_edges: NDArray[np.float64] | None = None,
) -> RainflowBins:
    """
    Rainflow on ply stress component histories (default ``σ11``).

    Never rainflow on beam resultants — only on scalar ply stress ``σ(t)``.
    Composite path only; isotropic block is separate in :func:`count_cycles_iso_vm`.

    Output shape ``(n_bin, n_s, n_comp_sub, n_ply)``. Isotropic bins are zeros
    (use :func:`count_cycles_iso_vm` separately).
    """
    sig = ply_stress_component(stress_history.sigma_composite, component)
    n_t, n_s, n_cp, n_ply = sig.shape
    series_for_edges: list[NDArray[np.float64]] = []
    if range_edges is None:
        for s in range(n_s):
            for p in range(n_cp):
                for k in range(n_ply):
                    y = sig[:, s, p, k]
                    if y.size > 1:
                        series_for_edges.append(y)
        edges = _range_edges_from_series(series_for_edges, n_range_bins)
    else:
        edges = np.asarray(range_edges, dtype=np.float64)

    ranges_c = np.zeros((edges.shape[0] - 1, n_s, n_cp, n_ply), dtype=np.float64)
    means_c = np.zeros_like(ranges_c)
    counts_c = np.zeros_like(ranges_c)

    for s in range(n_s):
        for p in range(n_cp):
            for k in range(n_ply):
                y = sig[:, s, p, k]
                r, m, c = _extract_cycles_arrays(y)
                rb, mb, cb = _cycles_to_bins(r, m, c, edges)
                ranges_c[:, s, p, k] = rb
                means_c[:, s, p, k] = mb
                counts_c[:, s, p, k] = cb

    z0 = np.zeros((edges.shape[0] - 1, n_s, stress_history.sigma_isotropic.shape[2]), dtype=np.float64)
    return RainflowBins(
        ranges_comp=ranges_c,
        means_comp=means_c,
        counts_comp=counts_c,
        ranges_iso=z0,
        means_iso=z0.copy(),
        counts_iso=z0.copy(),
    )


def count_cycles_iso_vm(
    stress_history: StressHistory,
    n_range_bins: int = 128,
    range_edges: NDArray[np.float64] | None = None,
) -> RainflowBins:
    """
    Rainflow on von Mises stress from isotropic membrane Voigt histories.

    Isotropic scalar driver ``σ_VM(t)`` only — separate vector layout from composite plies.

    Composite bins are zeros (use :func:`count_cycles_ply_stresses` separately).
    """
    sig_iso = stress_history.sigma_isotropic
    s11, s22, t12 = sig_iso[..., 0], sig_iso[..., 1], sig_iso[..., 2]
    sigma_vm = von_mises_plane_stress(s11, s22, t12)
    n_t, n_s, n_ip = sigma_vm.shape

    series_for_edges: list[NDArray[np.float64]] = []
    if range_edges is None:
        for si in range(n_s):
            for p in range(n_ip):
                y = sigma_vm[:, si, p]
                if y.size > 1:
                    series_for_edges.append(y)
        edges = _range_edges_from_series(series_for_edges, n_range_bins)
    else:
        edges = np.asarray(range_edges, dtype=np.float64)

    ranges_i = np.zeros((edges.shape[0] - 1, n_s, n_ip), dtype=np.float64)
    means_i = np.zeros_like(ranges_i)
    counts_i = np.zeros_like(ranges_i)

    for si in range(n_s):
        for p in range(n_ip):
            y = sigma_vm[:, si, p]
            r, m, c = _extract_cycles_arrays(y)
            rb, mb, cb = _cycles_to_bins(r, m, c, edges)
            ranges_i[:, si, p] = rb
            means_i[:, si, p] = mb
            counts_i[:, si, p] = cb

    n_bin = edges.shape[0] - 1
    _, n_cp, n_ply, _ = stress_history.sigma_composite.shape[1:]
    z0 = np.zeros((n_bin, n_s, n_cp, n_ply), dtype=np.float64)
    return RainflowBins(
        ranges_comp=z0,
        means_comp=z0.copy(),
        counts_comp=z0.copy(),
        ranges_iso=ranges_i,
        means_iso=means_i,
        counts_iso=counts_i,
    )


def merge_rainflow_bins(comp: RainflowBins, iso: RainflowBins) -> RainflowBins:
    """Merge composite-only and isotropic-only partial bins (same ``n_bin``, ``n_s``)."""
    return RainflowBins(
        ranges_comp=comp.ranges_comp,
        means_comp=comp.means_comp,
        counts_comp=comp.counts_comp,
        ranges_iso=iso.ranges_iso,
        means_iso=iso.means_iso,
        counts_iso=iso.counts_iso,
    )


class IncrementalRainflowAccumulator:
    """
    Accumulate scalar fatigue drivers across time chunks, then rainflow once.

    Avoids holding full ``(n_t, n_s, n_cp, n_ply, 3)`` stress tensors; still
    stores ``σ_driver(t, …)`` with ``O(n_t × n_signals)`` memory.
    """

    def __init__(
        self,
        cache: RecoveryCache,
        n_t: int,
        n_range_bins: int = 128,
        stress_component: int = 0,
    ) -> None:
        self._cache = cache
        self._n_range_bins = int(n_range_bins)
        self._stress_component = int(stress_component)
        n_s, n_cp, n_ply, _, _ = cache.L_rec.shape
        n_ip = cache.L_iso.shape[1]
        self._n_s = n_s
        self._n_cp = n_cp
        self._n_ply = n_ply
        self._n_ip = n_ip
        self._n_t = int(n_t)
        self._sigma11 = np.zeros((n_t, n_s, n_cp, n_ply), dtype=np.float64)
        self._sigma_vm = np.zeros((n_t, n_s, n_ip), dtype=np.float64)
        self._t0 = 0

    def update(self, stress_chunk: StressHistory) -> None:
        nt = int(stress_chunk.sigma_composite.shape[0])
        t1 = self._t0 + nt
        if t1 > self._n_t:
            raise ValueError("stress_chunk exceeds preallocated n_t for IncrementalRainflowAccumulator.")
        sl = slice(self._t0, t1)
        self._sigma11[sl] = ply_stress_component(stress_chunk.sigma_composite, self._stress_component)
        s = stress_chunk.sigma_isotropic
        self._sigma_vm[sl] = von_mises_plane_stress(s[..., 0], s[..., 1], s[..., 2])
        self._t0 = t1

    def finalise(self) -> RainflowBins:
        """Run rainflow on completed scalar histories and return binned cycles."""
        n_bin = self._n_range_bins
        n_s, n_cp, n_ply, n_ip = self._n_s, self._n_cp, self._n_ply, self._n_ip

        series_for_edges: list[NDArray[np.float64]] = []
        for s in range(n_s):
            for p in range(n_cp):
                for k in range(n_ply):
                    y = self._sigma11[:, s, p, k]
                    if y.size > 1:
                        series_for_edges.append(y)
        for s in range(n_s):
            for p in range(n_ip):
                y = self._sigma_vm[:, s, p]
                if y.size > 1:
                    series_for_edges.append(y)
        edges = _range_edges_from_series(series_for_edges, self._n_range_bins)

        ranges_c = np.zeros((n_bin, n_s, n_cp, n_ply), dtype=np.float64)
        means_c = np.zeros_like(ranges_c)
        counts_c = np.zeros_like(ranges_c)
        ranges_i = np.zeros((n_bin, n_s, n_ip), dtype=np.float64)
        means_i = np.zeros_like(ranges_i)
        counts_i = np.zeros_like(ranges_i)

        for si in range(n_s):
            for p in range(n_cp):
                for k in range(n_ply):
                    r, m, c = _extract_cycles_arrays(self._sigma11[:, si, p, k])
                    rb, mb, cb = _cycles_to_bins(r, m, c, edges)
                    ranges_c[:, si, p, k] = rb
                    means_c[:, si, p, k] = mb
                    counts_c[:, si, p, k] = cb

        for si in range(n_s):
            for p in range(n_ip):
                r, m, c = _extract_cycles_arrays(self._sigma_vm[:, si, p])
                rb, mb, cb = _cycles_to_bins(r, m, c, edges)
                ranges_i[:, si, p] = rb
                means_i[:, si, p] = mb
                counts_i[:, si, p] = cb

        return RainflowBins(
            ranges_comp=ranges_c,
            means_comp=means_c,
            counts_comp=counts_c,
            ranges_iso=ranges_i,
            means_iso=means_i,
            counts_iso=counts_i,
        )
