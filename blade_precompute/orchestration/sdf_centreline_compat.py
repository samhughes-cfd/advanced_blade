"""Compatibility notes for member / medial pipelines that consume grid-sampled ``phi``.

:class:`~blade_precompute.section_beam_model.gbt.member.MemberBucklingAnalysis` (GBT) operates on
modal strips and pre-buckling stresses — it does **not** import ``AirfoilSDF`` directly.
Section-scale SDFs still feed **centreline / medial** extraction in
``section_geometry`` via :meth:`SDFGrid.grad_magnitude` and friends; composite sections
should expose callables with the same ``(x, y) -> ndarray`` broadcasting rules as
``AirfoilSDF.__call__``.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SectionPhiCallable(Protocol):
    """Any section-level SDF usable on a numpy grid (including CSG-composed regions)."""

    def __call__(self, x: Any, y: Any) -> Any: ...


def describe_sdf_for_centreline(phi: SectionPhiCallable) -> str:
    return (
        "Centreline / medial routines expect a proper distance field on the "
        "evaluation grid (see AirfoilSDF optional Eikonal re-normalisation). "
        "Boolean CSG minima can create non-smooth ridges — validate |∇φ| before "
        "ridge following."
    )


def assert_grid_phi_finite(phi_grid: NDArray[np.floating]) -> None:
    """Debug helper: fail fast if SDF sampling produced non-finite values."""
    a = np.asarray(phi_grid, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("SDF grid contains non-finite values; centreline extraction is unsafe.")
