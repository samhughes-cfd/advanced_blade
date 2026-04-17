"""
tests.test_transforms
=====================
Unit tests for geometry.transforms.
"""

import numpy as np
import pytest
from blade_precompute.section_geometry.geometry.primitives import sdf_box, sdf_circle
from blade_precompute.section_geometry.geometry.transforms import (
    rotate_field, translate_field, scale_field,
    mirror_field_x, mirror_field_y,
    SDFFrame, rotate_points, flapwise_aligned_web_angle,
)


def circle(x, y): return sdf_circle(x, y, r=1.0)
def box(x, y):    return sdf_box(x, y, half_w=1.0, half_h=0.5)


class TestRotateField:
    def test_rotation_preserves_distances(self):
        """Rotating a circle SDF by any angle should leave all distances unchanged."""
        rotated = rotate_field(circle, np.radians(45))
        x = np.array([1.5, -0.5, 0.0])
        y = np.array([0.0,  0.0, 1.5])
        np.testing.assert_allclose(
            circle(x, y), rotated(x, y), atol=1e-10
        )

    def test_rotation_moves_geometry(self):
        """Rotating a box by 90° should move points between inside/outside."""
        # Point at (0.7, 0) is inside the box (half_w=1, half_h=0.5)
        assert box(0.7, 0.0) < 0.0
        # After 90° rotation the box becomes (half_h=0.5 wide, half_w=1 tall)
        # Same point (0.7, 0) is now outside
        rotated_box = rotate_field(box, np.radians(90))
        assert rotated_box(0.7, 0.0) > 0.0

    def test_rotation_about_nonzero_centre(self):
        """Rotating about (1, 0) moves a circle centred at (1, 0) onto itself."""
        c = lambda x, y: sdf_circle(x, y, cx=1.0, cy=0.0, r=0.5)
        rotated = rotate_field(c, np.radians(90), cx=1.0, cy=0.0)
        # Points on the original boundary should still be on the boundary
        theta = np.linspace(0, 2 * np.pi, 20)
        bx = 1.0 + 0.5 * np.cos(theta)
        by = 0.0 + 0.5 * np.sin(theta)
        np.testing.assert_allclose(rotated(bx, by), 0.0, atol=1e-10)

    def test_rotation_inverse(self):
        """Rotating by +θ then −θ should return the original field."""
        f_fwd = rotate_field(box, np.radians(37))
        f_inv = rotate_field(f_fwd, np.radians(-37))
        x = np.linspace(-1.5, 1.5, 30)
        y = np.linspace(-1.0, 1.0, 20)
        X, Y = np.meshgrid(x, y)
        np.testing.assert_allclose(box(X, Y), f_inv(X, Y), atol=1e-10)

    def test_360_degree_invariance(self):
        """Full rotation should return identity."""
        f_full = rotate_field(box, np.radians(360))
        x = np.array([0.3, -0.8, 1.2])
        y = np.array([0.1,  0.4, 0.0])
        np.testing.assert_allclose(box(x, y), f_full(x, y), atol=1e-10)


class TestTranslateField:
    def test_translate_moves_circle(self):
        """Translate circle by (2, 1): centre should now be at (2, 1)."""
        c_moved = translate_field(circle, 2.0, 1.0)
        # Inside the moved circle
        assert c_moved(2.0, 1.0) < 0.0
        # Original centre should now be outside
        assert c_moved(0.0, 0.0) > 0.0

    def test_translate_preserves_distances(self):
        """Distances from the boundary should be unchanged after translate."""
        c_moved = translate_field(circle, 5.0, -3.0)
        # Point 2 units from moved centre → phi = 1
        np.testing.assert_allclose(c_moved(7.0, -3.0), 1.0, atol=1e-10)

    def test_translate_inverse(self):
        f1 = translate_field(box, 1.0, 2.0)
        f2 = translate_field(f1, -1.0, -2.0)
        x = np.array([0.0, 0.5, -0.3])
        y = np.array([0.0, 0.2,  0.4])
        np.testing.assert_allclose(box(x, y), f2(x, y), atol=1e-10)


class TestMirror:
    def test_mirror_x_symmetric_field(self):
        """Mirror of a symmetric circle should be identical."""
        m = mirror_field_x(circle)
        x = np.array([0.5, -0.3, 1.2])
        y = np.array([0.7,  0.1, 0.0])
        np.testing.assert_allclose(circle(x, y), m(x, y), atol=1e-10)

    def test_mirror_x_asymmetric_field(self):
        """Mirror_x of a box flips sign of y-offset correctly."""
        # Box: half_w=1, half_h=0.5. Point (0, 0.3) is inside → phi < 0.
        # mirror_x sends (0, 0.3) → evaluate at (0, -0.3) which is also inside.
        assert mirror_field_x(box)(0.0, 0.3) < 0.0

    def test_mirror_y(self):
        # Translate circle to (1, 0). Mirror_y sends it to (-1, 0).
        c = lambda x, y: sdf_circle(x, y, cx=1.0, r=0.5)
        m = mirror_field_y(c)
        assert m(-1.0, 0.0) < 0.0   # centre of mirrored circle
        assert m(1.0, 0.0) > 0.0    # original centre now outside


class TestSDFFrame:
    def test_identity_frame(self):
        frame = SDFFrame()
        wrapped = frame.apply(circle)
        x = np.array([1.5, 0.0, -0.5])
        y = np.array([0.0, 0.5,  0.3])
        np.testing.assert_allclose(circle(x, y), wrapped(x, y), atol=1e-10)

    def test_translate_then_rotate(self):
        # Translate circle to (1, 0), then rotate 90° CCW about origin.
        # New centre: (1,0) rotated 90° CCW = (0, 1).
        frame = SDFFrame().translate(1.0, 0.0).rotate(np.radians(90))
        wrapped = frame.apply(circle)
        # Centre should now be at (0, 1)
        assert wrapped(0.0, 1.0) < 0.0
        # Point well outside the moved circle
        assert wrapped(3.0, 0.0) > 0.0

    def test_inverse_frame(self):
        frame = SDFFrame().translate(2.0, -1.0).rotate(np.radians(30))
        inv   = frame.inverse()
        combined = inv.apply(frame.apply(circle))
        x = np.array([0.5, -0.3, 1.1])
        y = np.array([0.2,  0.7, 0.0])
        np.testing.assert_allclose(circle(x, y), combined(x, y), atol=1e-8)

    def test_compose(self):
        f1 = SDFFrame().translate(1.0, 0.0)
        f2 = SDFFrame().translate(0.0, 1.0)
        f12 = f1.compose(f2)
        # Equivalent to translate(1, 0) then translate(0, 1) = translate(1, 1)
        wrapped = f12.apply(circle)
        # Centre should be at (1, 1)
        assert wrapped(1.0, 1.0) < 0.0
        assert wrapped(0.0, 0.0) > 0.0


class TestHelpers:
    def test_rotate_points_90(self):
        xr, yr = rotate_points(np.array([1.0]), np.array([0.0]), np.radians(90))
        np.testing.assert_allclose(xr, 0.0, atol=1e-10)
        np.testing.assert_allclose(yr, 1.0, atol=1e-10)

    def test_flapwise_angle(self):
        # No twist → no correction needed
        assert flapwise_aligned_web_angle(0.0) == 0.0
        # 15° twist → web counter-rotated by -15°
        np.testing.assert_allclose(
            flapwise_aligned_web_angle(np.radians(15)),
            -np.radians(15), atol=1e-10
        )


class TestScaleFieldValidation:
    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            scale_field(circle, sx=0.0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            scale_field(circle, sx=1.0, sy=-1.0)

    def test_non_finite_scale_raises(self):
        with pytest.raises(ValueError, match="finite"):
            scale_field(circle, sx=np.inf)

    def test_valid_scale_evaluates(self):
        scaled = scale_field(circle, sx=2.0)
        assert np.isfinite(scaled(0.0, 0.0))
