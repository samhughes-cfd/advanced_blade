"""PCHIP spanwise interpolation of classical section stiffnesses."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from blade_precompute.section_beam_model.gbt.section_stiffness_export import SectionStiffness

from .core.types import SectionStiffnessArray


def section_stiffness_array_from_sequence(
    s: NDArray[np.float64] | Sequence[float],
    items: Sequence[SectionStiffness],
) -> SectionStiffnessArray:
    """Build :class:`SectionStiffnessArray` from parallel ``SectionStiffness`` values."""
    s_arr = np.asarray(s, dtype=np.float64).ravel()
    if len(items) != int(s_arr.size):
        raise ValueError("items length must match s length.")
    return SectionStiffnessArray(
        s=s_arr,
        EA=np.array([x.EA for x in items], dtype=np.float64),
        EI_x=np.array([x.EI_x for x in items], dtype=np.float64),
        EI_y=np.array([x.EI_y for x in items], dtype=np.float64),
        GJ=np.array([x.GJ for x in items], dtype=np.float64),
        GA_x=np.array([x.GA_x for x in items], dtype=np.float64),
        GA_y=np.array([x.GA_y for x in items], dtype=np.float64),
    )


class SectionPropertyInterpolator:
    """
    Interpolate :class:`SectionStiffnessArray` onto arbitrary span coordinates using
    independent PCHIP curves per scalar component.
    """

    def __init__(
        self,
        stations: NDArray[np.float64] | Sequence[float],
        stiffness_array: SectionStiffnessArray,
    ) -> None:
        st = np.asarray(stations, dtype=np.float64).ravel()
        if st.shape[0] != stiffness_array.s.shape[0]:
            raise ValueError("stations length must match stiffness_array.s length.")
        if not np.allclose(st, stiffness_array.s):
            raise ValueError("stations must match stiffness_array.s (same values, increasing).")
        self._s = st
        self._array = stiffness_array
        self._interp: dict[str, PchipInterpolator] = {}
        for name in ("EA", "EI_x", "EI_y", "GJ", "GA_x", "GA_y"):
            y = np.asarray(getattr(stiffness_array, name), dtype=np.float64).ravel()
            self._interp[name] = PchipInterpolator(st, y, extrapolate=False)

    def interpolate(
        self,
        query_points: NDArray[np.float64] | Sequence[float],
        *,
        allow_extrapolation: bool = False,
    ) -> SectionStiffnessArray:
        zq = np.asarray(query_points, dtype=np.float64).ravel()
        s0, s1 = float(self._s[0]), float(self._s[-1])
        if not allow_extrapolation and zq.size > 0:
            if float(zq.min()) < s0 - 1e-9 or float(zq.max()) > s1 + 1e-9:
                raise ValueError(
                    f"query_points must lie within [{s0}, {s1}] (set allow_extrapolation=True to clamp)."
                )
        if allow_extrapolation:
            zq_eval = np.clip(zq, s0, s1)
        else:
            zq_eval = zq
        cols = {}
        for name, ip in self._interp.items():
            try:
                cols[name] = ip(zq_eval)
            except ValueError as exc:
                raise ValueError(
                    f"Interpolation failed for {name}; check query_points within station range."
                ) from exc
        return SectionStiffnessArray(s=zq.copy(), **cols)

    def plot_distribution(self, component: str) -> None:
        """Plot a stiffness component vs span (requires matplotlib)."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_distribution.") from exc
        if component not in self._interp:
            raise ValueError(f"Unknown component {component!r}.")
        fig, ax = plt.subplots(figsize=(7, 4))
        s_fine = np.linspace(float(self._s[0]), float(self._s[-1]), 200)
        ax.plot(self._s, getattr(self._array, component), "o", label="stations")
        ax.plot(s_fine, self._interp[component](s_fine), "-", label="PCHIP")
        ax.set_xlabel("s [m]")
        ax.set_ylabel(component)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()
