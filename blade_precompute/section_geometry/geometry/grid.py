"""
geometry.grid
=============
Structured background grid for SDF evaluation and post-processing.

SDFGrid
-------
Wraps a 2-D Cartesian grid and provides:
  - Vectorised SDF evaluation for any callable
  - Gradient computation (∇φ via central finite differences)
  - Gradient magnitude |∇φ|  (used by medial-axis extraction)
  - Zero-level-set extraction as a polyline (via matplotlib contour)
  - Masked interior / exterior arrays
  - 2-D quadrature (area, centroid, moments of inertia) for a region
    defined by phi < 0

Usage
-----
    from geometry.grid import SDFGrid

    grid = SDFGrid.from_bbox(-0.05, 1.05, -0.15, 0.15, nx=512, ny=200)
    phi  = grid.eval(airfoil_sdf)
    grad = grid.gradient(phi)
    gm   = grid.grad_magnitude(phi)
"""

import numpy as np


class SDFGrid:
    """Structured 2-D grid for SDF operations.

    Parameters
    ----------
    X, Y : ndarray, shape (ny, nx)
        Meshgrid coordinate arrays (from np.meshgrid with indexing='xy').
    """

    def __init__(self, X, Y):
        if X.shape != Y.shape:
            raise ValueError("X and Y must have identical shape.")
        self.X  = X
        self.Y  = Y
        self.nx = X.shape[1]
        self.ny = X.shape[0]
        self.dx = float(X[0, 1] - X[0, 0]) if self.nx > 1 else 1.0
        self.dy = float(Y[1, 0] - Y[0, 0]) if self.ny > 1 else 1.0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bbox(cls, x_min, x_max, y_min, y_max, nx=400, ny=200):
        """Create a grid covering a bounding box.

        Parameters
        ----------
        x_min, x_max, y_min, y_max : float
            Spatial extents.
        nx, ny : int
            Number of grid points in each direction.
        """
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        return cls(X, Y)

    @classmethod
    def from_airfoil(cls, airfoil_sdf, padding=0.1, nx=400, ny=200):
        """Automatically size the grid to wrap an AirfoilSDF with padding."""
        verts = airfoil_sdf.vertices
        x_min = verts[:, 0].min() - padding
        x_max = verts[:, 0].max() + padding
        y_min = verts[:, 1].min() - padding
        y_max = verts[:, 1].max() + padding
        return cls.from_bbox(x_min, x_max, y_min, y_max, nx=nx, ny=ny)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def eval(self, sdf_callable):
        """Evaluate an SDF callable on the full grid.

        Parameters
        ----------
        sdf_callable : callable (x, y) → ndarray

        Returns
        -------
        phi : ndarray, shape (ny, nx)
        """
        return np.asarray(sdf_callable(self.X, self.Y), dtype=float)

    def gradient(self, phi):
        """Central-difference gradient of a grid-sampled field.

        Parameters
        ----------
        phi : ndarray, shape (ny, nx)

        Returns
        -------
        gx, gy : ndarray, shape (ny, nx)
            Partial derivatives ∂φ/∂x and ∂φ/∂y.
        """
        gx = np.gradient(phi, self.dx, axis=1)
        gy = np.gradient(phi, self.dy, axis=0)
        return gx, gy

    def grad_magnitude(self, phi):
        """Return |∇φ| on the grid.

        For a true SDF |∇φ| = 1 everywhere.  Deviations indicate:
          - |∇φ| < 1 − ε  →  possible medial-axis singularity
          - |∇φ| > 1 + ε  →  numerical artefact / CSG seam
        """
        gx, gy = self.gradient(phi)
        return np.sqrt(gx**2 + gy**2)

    def eikonal_error(self, phi):
        """Pointwise Eikonal residual  | |∇φ| − 1 |."""
        return np.abs(self.grad_magnitude(phi) - 1.0)

    # ------------------------------------------------------------------
    # Level-set extraction
    # ------------------------------------------------------------------

    def zero_contour(self, phi):
        """Extract the zero-level-set as a list of (N, 2) polyline arrays.

        Requires matplotlib (for contour-finding only; no plot shown).

        Returns
        -------
        segments : list of ndarray, shape (N, 2)
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cs = ax.contour(self.X, self.Y, phi, levels=[0.0])
        segments = []
        # Matplotlib ≥3.8: QuadContourSet exposes get_paths(); .collections was removed.
        for path in cs.get_paths():
            v = path.vertices
            segments.append(v.copy())
        plt.close(fig)
        return segments

    def level_set(self, phi, level=0.0):
        """Extract an arbitrary level-set contour."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cs = ax.contour(self.X, self.Y, phi, levels=[level])
        segments = []
        for path in cs.get_paths():
            segments.append(path.vertices.copy())
        plt.close(fig)
        return segments

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def interior_mask(self, phi):
        """Boolean mask: True where phi < 0 (inside the shape)."""
        return phi < 0.0

    def boundary_mask(self, phi, tol=None):
        """Boolean mask near the zero-level-set.

        Parameters
        ----------
        tol : float, optional
            Half-band tolerance.  Defaults to max(dx, dy).
        """
        tol = tol or max(self.dx, self.dy)
        return np.abs(phi) < tol

    # ------------------------------------------------------------------
    # Quadrature
    # ------------------------------------------------------------------

    def area(self, phi):
        """Compute the enclosed area (phi < 0) by cell-counting quadrature.

        Returns
        -------
        float
        """
        mask = phi < 0.0
        return float(mask.sum()) * self.dx * self.dy

    def centroid(self, phi):
        """Centroid of the region phi < 0.

        Returns
        -------
        cx, cy : float
        """
        mask = (phi < 0.0).astype(float)
        A  = mask.sum()
        if A == 0:
            return float("nan"), float("nan")
        cx = (mask * self.X).sum() / A
        cy = (mask * self.Y).sum() / A
        return float(cx), float(cy)

    def second_moments(self, phi):
        """Second moments of area (Ixx, Iyy, Ixy) about the centroid.

        Returns
        -------
        Ixx, Iyy, Ixy : float
            Ixx = ∫∫ y² dA,  Iyy = ∫∫ x² dA,  Ixy = ∫∫ xy dA
        """
        cx, cy = self.centroid(phi)
        mask   = (phi < 0.0).astype(float)
        dA     = self.dx * self.dy
        xr = self.X - cx
        yr = self.Y - cy
        Ixx = float((mask * yr**2).sum() * dA)
        Iyy = float((mask * xr**2).sum() * dA)
        Ixy = float((mask * xr * yr).sum() * dA)
        return Ixx, Iyy, Ixy

    def section_properties(self, phi):
        """Return a dict of section properties for the region phi < 0.

        Keys: area, cx, cy, Ixx, Iyy, Ixy, r_gyr_x, r_gyr_y
        """
        A  = self.area(phi)
        cx, cy = self.centroid(phi)
        Ixx, Iyy, Ixy = self.second_moments(phi)
        return {
            "area":    A,
            "cx":      cx,
            "cy":      cy,
            "Ixx":     Ixx,
            "Iyy":     Iyy,
            "Ixy":     Ixy,
            "r_gyr_x": float(np.sqrt(Ixx / A)) if A > 0 else float("nan"),
            "r_gyr_y": float(np.sqrt(Iyy / A)) if A > 0 else float("nan"),
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"SDFGrid(nx={self.nx}, ny={self.ny}, "
            f"dx={self.dx:.4g}, dy={self.dy:.4g}, "
            f"x=[{self.X.min():.3g},{self.X.max():.3g}], "
            f"y=[{self.Y.min():.3g},{self.Y.max():.3g}])"
        )
