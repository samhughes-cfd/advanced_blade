from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


class SDFField(Protocol):
    """Signed-distance field in 2D section coordinates [y, z]."""

    def eval(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate phi on points shaped (n, 2)."""


@dataclass(frozen=True)
class StationFrame2D:
    """
    Station frame transform between S and B section frames.

    S frame: chord-based, twist-agnostic.
    B frame: edge/flap frame, twist-aware.
    Positive twist rotates S coordinates to B with right-handed convention.
    """

    twist_rad: float

    def rotation_s_to_b(self) -> NDArray[np.float64]:
        c = float(np.cos(self.twist_rad))
        s = float(np.sin(self.twist_rad))
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def rotation_b_to_s(self) -> NDArray[np.float64]:
        r = self.rotation_s_to_b()
        return r.T.copy()

    def points_s_to_b(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64) @ self.rotation_s_to_b().T

    def points_b_to_s(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64) @ self.rotation_b_to_s().T

    def direction_b_to_s(self, direction_b: NDArray[np.float64]) -> NDArray[np.float64]:
        vec = np.asarray(direction_b, dtype=np.float64).reshape(2)
        return self.rotation_b_to_s() @ vec


@dataclass(frozen=True)
class FrameTaggedPolyline:
    name: str
    points: NDArray[np.float64]
    frame: str  # "S" or "B"


@dataclass(frozen=True)
class GeometryConstraintSpec:
    """
    Blade-section constraints for implicit geometry construction.
    """

    skin_outer_boundary_s: NDArray[np.float64]
    skin_thickness: float
    web_width: float
    web_stations_s: tuple[float, float]
    spar_cap_width: float
    spar_cap_thickness: float
    twist_rad: float
    station_z: float
    materials: dict[str, object]
    thickness_field: Callable[[float], float] | None = None
    n_samples: int = 256


@dataclass
class MedialAxisDiagnostics:
    branch_count: int
    spur_count: int
    disconnected_components: int
    thickness_residual_rms: float
    self_intersections: int
    notes: list[str] = field(default_factory=list)


@dataclass
class MidlineExtractionResult:
    midsurface_coords_s: NDArray[np.float64]
    strip_width_m: float
    diagnostics: MedialAxisDiagnostics

