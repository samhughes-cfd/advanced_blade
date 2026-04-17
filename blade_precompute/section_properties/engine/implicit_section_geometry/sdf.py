from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .types import SDFField


def _point_segment_distance(points: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    ab = b - a
    ap = points - a[None, :]
    den = float(np.dot(ab, ab))
    if den < 1e-30:
        return np.linalg.norm(ap, axis=1)
    t = np.clip((ap @ ab) / den, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1)


def _points_in_polygon(points: NDArray[np.float64], poly: NDArray[np.float64]) -> NDArray[np.bool_]:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    for i in range(poly.shape[0]):
        cond = ((y1[i] > y) != (y2[i] > y)) & (x < (x2[i] - x1[i]) * (y - y1[i]) / (y2[i] - y1[i] + 1e-30) + x1[i])
        inside ^= cond
    return inside


@dataclass(frozen=True)
class PolygonSDF(SDFField):
    polygon: NDArray[np.float64]

    def __post_init__(self) -> None:
        poly = np.asarray(self.polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise ValueError("polygon must be shape (n,2), n>=3.")
        object.__setattr__(self, "polygon", poly)

    def eval(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(points, dtype=np.float64)
        d = np.full(p.shape[0], np.inf, dtype=np.float64)
        for i in range(self.polygon.shape[0]):
            a = self.polygon[i]
            b = self.polygon[(i + 1) % self.polygon.shape[0]]
            d = np.minimum(d, _point_segment_distance(p, a, b))
        inside = _points_in_polygon(p, self.polygon)
        return np.where(inside, -d, d)


@dataclass(frozen=True)
class BoxSDF(SDFField):
    center: tuple[float, float]
    half_size: tuple[float, float]

    def eval(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)[None, :]
        q = np.abs(p) - np.asarray(self.half_size, dtype=np.float64)[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        return outside + inside


@dataclass(frozen=True)
class CircleSDF(SDFField):
    center: tuple[float, float]
    radius: float

    def eval(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(points, dtype=np.float64)
        c = np.asarray(self.center, dtype=np.float64)
        return np.linalg.norm(p - c[None, :], axis=1) - float(self.radius)


@dataclass(frozen=True)
class UnionSDF(SDFField):
    fields: tuple[SDFField, ...]

    def eval(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        vals = [f.eval(points) for f in self.fields]
        return np.min(np.stack(vals, axis=0), axis=0)


@dataclass(frozen=True)
class IntersectionSDF(SDFField):
    fields: tuple[SDFField, ...]

    def eval(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        vals = [f.eval(points) for f in self.fields]
        return np.max(np.stack(vals, axis=0), axis=0)


def sdf_union(fields: Iterable[SDFField]) -> SDFField:
    return UnionSDF(tuple(fields))


def sdf_intersection(fields: Iterable[SDFField]) -> SDFField:
    return IntersectionSDF(tuple(fields))

