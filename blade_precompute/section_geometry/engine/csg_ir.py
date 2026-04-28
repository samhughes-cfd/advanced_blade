"""Compiled CSG expression graph for grid-based SDF evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from ..geometry.primitives import (
    sdf_box,
    sdf_capsule,
    sdf_circle,
    sdf_half_plane,
    sdf_oriented_box,
    sdf_polygon,
    sdf_rounded_box,
)


@dataclass(frozen=True)
class Expr:
    """Base expression type."""


@dataclass(frozen=True)
class Circle(Expr):
    cx: float = 0.0
    cy: float = 0.0
    r: float = 1.0


@dataclass(frozen=True)
class Box(Expr):
    cx: float = 0.0
    cy: float = 0.0
    half_w: float = 1.0
    half_h: float = 0.5


@dataclass(frozen=True)
class RoundedBox(Expr):
    cx: float = 0.0
    cy: float = 0.0
    half_w: float = 1.0
    half_h: float = 0.5
    r: float = 0.1


@dataclass(frozen=True)
class OrientedBox(Expr):
    cx: float = 0.0
    cy: float = 0.0
    half_w: float = 1.0
    half_h: float = 0.5
    angle: float = 0.0


@dataclass(frozen=True)
class Capsule(Expr):
    ax: float
    ay: float
    bx: float
    by: float
    r: float


@dataclass(frozen=True)
class HalfPlane(Expr):
    nx: float = 0.0
    ny: float = 1.0
    d: float = 0.0


@dataclass(frozen=True)
class Polygon(Expr):
    vertices: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class CallableField(Expr):
    """Opaque callable leaf keyed by a stable token."""

    key: str
    fn: Any = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class Union(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Intersect(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Subtract(Expr):
    base: Expr
    cutter: Expr


@dataclass(frozen=True)
class Offset(Expr):
    base: Expr
    amount: float


@dataclass(frozen=True)
class Rotate(Expr):
    base: Expr
    angle: float
    cx: float = 0.0
    cy: float = 0.0


def eval_expr(expr: Expr, x, y):
    """Evaluate an expression on arbitrary coordinate arrays."""
    if isinstance(expr, Circle):
        return sdf_circle(x, y, cx=expr.cx, cy=expr.cy, r=expr.r)
    if isinstance(expr, Box):
        return sdf_box(x, y, cx=expr.cx, cy=expr.cy, half_w=expr.half_w, half_h=expr.half_h)
    if isinstance(expr, RoundedBox):
        return sdf_rounded_box(
            x, y, cx=expr.cx, cy=expr.cy, half_w=expr.half_w, half_h=expr.half_h, r=expr.r
        )
    if isinstance(expr, OrientedBox):
        return sdf_oriented_box(
            x, y, cx=expr.cx, cy=expr.cy, half_w=expr.half_w, half_h=expr.half_h, angle=expr.angle
        )
    if isinstance(expr, Capsule):
        return sdf_capsule(x, y, ax=expr.ax, ay=expr.ay, bx=expr.bx, by=expr.by, r=expr.r)
    if isinstance(expr, HalfPlane):
        return sdf_half_plane(x, y, nx=expr.nx, ny=expr.ny, d=expr.d)
    if isinstance(expr, Polygon):
        verts = np.asarray(expr.vertices, dtype=float)
        return sdf_polygon(x, y, verts)
    if isinstance(expr, CallableField):
        return np.asarray(expr.fn(x, y), dtype=float)
    if isinstance(expr, Union):
        return np.minimum(eval_expr(expr.left, x, y), eval_expr(expr.right, x, y))
    if isinstance(expr, Intersect):
        return np.maximum(eval_expr(expr.left, x, y), eval_expr(expr.right, x, y))
    if isinstance(expr, Subtract):
        return np.maximum(eval_expr(expr.base, x, y), -eval_expr(expr.cutter, x, y))
    if isinstance(expr, Offset):
        return eval_expr(expr.base, x, y) - float(expr.amount)
    if isinstance(expr, Rotate):
        cos_a = np.cos(-float(expr.angle))
        sin_a = np.sin(-float(expr.angle))
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        xr = xa - float(expr.cx)
        yr = ya - float(expr.cy)
        xp = cos_a * xr - sin_a * yr + float(expr.cx)
        yp = sin_a * xr + cos_a * yr + float(expr.cy)
        return eval_expr(expr.base, xp, yp)
    raise TypeError(f"Unsupported Expr node: {type(expr).__name__}")


def compile_expr(expr: Expr, grid, cache: Dict[Expr, np.ndarray] | None = None) -> np.ndarray:
    """Compile/evaluate expression on a grid with subtree memoization."""
    memo = {} if cache is None else cache
    hit = memo.get(expr)
    if hit is not None:
        return hit

    if isinstance(expr, Union):
        out = np.minimum(compile_expr(expr.left, grid, memo), compile_expr(expr.right, grid, memo))
    elif isinstance(expr, Intersect):
        out = np.maximum(compile_expr(expr.left, grid, memo), compile_expr(expr.right, grid, memo))
    elif isinstance(expr, Subtract):
        out = np.maximum(compile_expr(expr.base, grid, memo), -compile_expr(expr.cutter, grid, memo))
    elif isinstance(expr, Offset):
        out = compile_expr(expr.base, grid, memo) - float(expr.amount)
    elif isinstance(expr, Rotate):
        out = np.asarray(eval_expr(expr, grid.X, grid.Y), dtype=float)
    else:
        out = np.asarray(eval_expr(expr, grid.X, grid.Y), dtype=float)
    memo[expr] = out
    return out


__all__ = [
    "Expr",
    "Circle",
    "Box",
    "RoundedBox",
    "OrientedBox",
    "Capsule",
    "HalfPlane",
    "Polygon",
    "CallableField",
    "Union",
    "Intersect",
    "Subtract",
    "Offset",
    "Rotate",
    "eval_expr",
    "compile_expr",
]

