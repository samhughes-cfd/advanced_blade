"""Regression tests for compiled CSG expression graphs."""

import numpy as np

from blade_precompute.section_geometry.engine.csg_ir import (
    Box,
    Circle,
    Intersect,
    Offset,
    Subtract,
    Union,
    compile_expr,
)
from blade_precompute.section_geometry.geometry.csg import intersect, offset, subtract, union
from blade_precompute.section_geometry.geometry.grid import SDFGrid
from blade_precompute.section_geometry.geometry.primitives import sdf_box, sdf_circle
from blade_precompute.section_geometry.engine.implicit_section_geometry import AirfoilSDF, LEInsert, TEInsert
from blade_precompute.section_geometry.sections.subcomponents import (
    ContinuousSparCap,
    SandwichCore,
    ShearWeb,
    SparCap,
)


def test_compile_expr_matches_callable_csg():
    grid = SDFGrid.from_bbox(-2.0, 2.0, -2.0, 2.0, nx=120, ny=100)
    expr = Subtract(
        base=Union(Circle(cx=-0.2, cy=0.0, r=0.8), Box(cx=0.5, cy=0.0, half_w=0.5, half_h=0.4)),
        cutter=Offset(Circle(cx=0.0, cy=0.0, r=0.5), amount=-0.08),
    )
    phi_expr = compile_expr(expr, grid)

    f = subtract(
        union(
            lambda x, y: sdf_circle(x, y, cx=-0.2, cy=0.0, r=0.8),
            lambda x, y: sdf_box(x, y, cx=0.5, cy=0.0, half_w=0.5, half_h=0.4),
        ),
        offset(lambda x, y: sdf_circle(x, y, cx=0.0, cy=0.0, r=0.5), -0.08),
    )
    phi_ref = grid.eval(f)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_compile_expr_subtree_memoization_reuses_cache():
    grid = SDFGrid.from_bbox(-1.5, 1.5, -1.0, 1.0, nx=90, ny=70)
    leaf = Circle(cx=0.0, cy=0.0, r=0.6)
    expr = Intersect(Union(leaf, Box(cx=0.3, cy=0.0, half_w=0.4, half_h=0.3)), leaf)
    memo = {}
    phi = compile_expr(expr, grid, cache=memo)
    assert isinstance(phi, np.ndarray)
    assert leaf in memo
    assert expr in memo


def test_te_insert_expr_matches_callable():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    comp = TEInsert(af, skin_thickness=0.003, x_start=0.72)
    assert comp._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(comp._expr, grid)
    phi_ref = grid.eval(comp)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_le_insert_expr_matches_callable():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    comp = LEInsert(af, skin_thickness=0.003, x_end=0.12, le_x=0.0, le_y=0.0, radius=0.12)
    assert comp._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(comp._expr, grid)
    phi_ref = grid.eval(comp)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_shear_web_expr_matches_callable_chord_normal():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    comp = ShearWeb(
        af,
        skin_thickness=0.003,
        x_top=0.25,
        y_top=0.06,
        x_bot=0.25,
        y_bot=-0.04,
        thickness=0.004,
        alignment="chord_normal",
        twist_angle=0.0,
    )
    assert comp._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(comp._expr, grid)
    phi_ref = grid.eval(comp)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_shear_web_expr_matches_callable_flapwise():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    comp = ShearWeb(
        af,
        skin_thickness=0.003,
        x_top=0.25,
        y_top=0.06,
        x_bot=0.25,
        y_bot=-0.04,
        thickness=0.004,
        alignment="flapwise",
        twist_angle=0.2,
    )
    assert comp._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(comp._expr, grid)
    phi_ref = grid.eval(comp)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_sandwich_core_expr_matches_callable_chord_clips():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    core = SandwichCore(
        af,
        skin_thickness=0.003,
        exclusion_sdfs=[],
        x_start=0.2,
        x_end=0.7,
    )
    assert core._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(core._expr, grid)
    phi_ref = grid.eval(core)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_sandwich_core_expr_matches_callable_with_exclusion():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    web = ShearWeb(
        af,
        skin_thickness=0.003,
        x_top=0.3,
        y_top=0.05,
        x_bot=0.3,
        y_bot=-0.05,
        thickness=0.004,
    )
    core = SandwichCore(
        af,
        skin_thickness=0.003,
        exclusion_sdfs=[web],
        x_start=0.2,
        x_end=0.7,
    )
    assert core._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(core._expr, grid)
    phi_ref = grid.eval(core)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_spar_cap_expr_matches_callable():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    comp = SparCap(
        af,
        skin_thickness=0.003,
        x_start=0.2,
        x_end=0.6,
        cap_height=0.012,
        surface="upper",
    )
    assert comp._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(comp._expr, grid)
    phi_ref = grid.eval(comp)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)


def test_continuous_spar_cap_expr_matches_callable_flapwise():
    af = AirfoilSDF.from_naca("2412", n_points=180, chord=1.0)
    comp = ContinuousSparCap(
        af,
        skin_thickness=0.003,
        x_start=0.2,
        x_end=0.6,
        cap_height=0.012,
        surface="lower",
        twist_angle=0.15,
    )
    assert comp._expr is not None
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=160, ny=90)
    phi_expr = compile_expr(comp._expr, grid)
    phi_ref = grid.eval(comp)
    np.testing.assert_allclose(phi_expr, phi_ref, atol=1e-12, rtol=0.0)

