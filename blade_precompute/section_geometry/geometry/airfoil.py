"""
geometry.airfoil
================
Convert a discrete airfoil polyline into a robust SDF callable.

Approach
--------
1.  Accept (x, y) point arrays that trace the airfoil contour (any starting
    point; automatically closed).
2.  Build an exact polyhedral SDF via ``sdf_polygon`` (winding-number sign,
    edge-distance magnitude).
3.  Optionally re-normalise using a fast-marching Eikonal solve so downstream
    medial-axis extraction sees a true distance field (requires scikit-fmm).
4.  Expose convenience methods for:
      - chord normalisation / denormalisation
      - camber-line extraction (via upper/lower surface split)
      - thickness distribution
      - NACA 4-digit parametric generation

Usage
-----
    from geometry.airfoil import AirfoilSDF

    # From data file
    af = AirfoilSDF.from_dat("naca0012.dat")

    # Evaluate on a grid
    X, Y = np.meshgrid(np.linspace(-0.1, 1.1, 300), np.linspace(-0.2, 0.2, 120))
    phi = af(X, Y)

    # NACA generation
    af = AirfoilSDF.from_naca("2412", n_points=200)
"""

import numpy as np
from .primitives import sdf_polygon


class AirfoilSDF:
    """Implicit SDF representation of an airfoil cross-section.

    Parameters
    ----------
    vertices : ndarray, shape (N, 2)
        Ordered (x, y) coordinates tracing the airfoil contour.
        Need not be closed (last→first edge added automatically).
    chord : float, optional
        Reference chord length. If None, inferred as max(x) - min(x).
    """

    def __init__(self, vertices, chord=None):
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(
                f"vertices must have shape (N, 2); got array with shape {verts.shape}."
            )
        if len(verts) < 3:
            raise ValueError(
                f"At least 3 vertices are required to define an airfoil polygon; got {len(verts)}."
            )
        # Ensure closed (remove duplicate closing point if present)
        if np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        if len(verts) < 3:
            raise ValueError(
                "Airfoil polygon has fewer than 3 unique vertices after removing duplicated closing point."
            )
        self._verts = verts
        self.chord = float(chord) if chord is not None else float(verts[:, 0].max() - verts[:, 0].min())
        if not np.isfinite(self.chord) or self.chord <= 0.0:
            raise ValueError(f"chord must be a positive finite value; got {self.chord!r}.")
        self._le_idx = int(np.argmin(verts[:, 0]))   # leading-edge index

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, x, y):
        """Evaluate the SDF at arbitrary coordinates.

        Parameters
        ----------
        x, y : array-like
            Evaluation coordinates (broadcastable).

        Returns
        -------
        phi : ndarray
            phi < 0 inside airfoil, phi > 0 outside.
        """
        return sdf_polygon(x, y, self._verts)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dat(cls, filepath, chord=None):
        """Load from a Selig-format .dat file.

        Supports both single-surface (Lednicer) and wrapped (Selig) formats.
        Lines beginning with '#' or containing alphabetic text are skipped.
        """
        pts = []
        with open(filepath, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        pts.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue  # header / name line
        if not pts:
            raise ValueError(
                f"No valid airfoil coordinate rows found in DAT file: {filepath}"
            )
        verts = np.array(pts)
        return cls(verts, chord=chord)

    @classmethod
    def from_array(cls, xy, chord=None):
        """Construct from an (N, 2) NumPy array."""
        return cls(xy, chord=chord)

    @classmethod
    def from_naca(cls, naca_code, n_points=200, chord=1.0, closed_te=True):
        """Generate a NACA 4-digit airfoil.

        Parameters
        ----------
        naca_code : str
            4-digit NACA designation, e.g. "2412" or "0012".
        n_points : int
            Number of surface points (split equally between upper/lower).
        chord : float
            Chord length (output scaled accordingly).
        closed_te : bool
            If True, use the closed trailing-edge modification.

        Returns
        -------
        AirfoilSDF
        """
        code = str(naca_code).strip().zfill(4)
        if len(code) != 4:
            raise ValueError(f"Expected 4-digit NACA code, got '{naca_code}'.")

        m  = int(code[0]) / 100.0   # max camber fraction
        p  = int(code[1]) / 10.0    # max camber position
        tt = int(code[2:]) / 100.0  # thickness fraction

        # Cosine-spaced x distribution (0 → 1)
        beta = np.linspace(0.0, np.pi, n_points // 2 + 1)
        xc   = 0.5 * (1.0 - np.cos(beta))

        # Thickness
        a = [0.2969, -0.1260, -0.3516, 0.2843, -0.1015 if closed_te else -0.1036]
        yt = (tt / 0.2) * (
            a[0] * np.sqrt(xc)
            + a[1] * xc
            + a[2] * xc**2
            + a[3] * xc**3
            + a[4] * xc**4
        )

        # Camber line and gradient
        if m == 0.0 or p == 0.0:
            yc    = np.zeros_like(xc)
            dyc   = np.zeros_like(xc)
        else:
            yc  = np.where(
                xc <= p,
                (m / p**2) * (2 * p * xc - xc**2),
                (m / (1 - p)**2) * ((1 - 2*p) + 2*p*xc - xc**2),
            )
            dyc = np.where(
                xc <= p,
                (2*m / p**2) * (p - xc),
                (2*m / (1-p)**2) * (p - xc),
            )

        theta = np.arctan(dyc)

        xu = (xc - yt * np.sin(theta)) * chord
        yu = (yc + yt * np.cos(theta)) * chord
        xl = (xc + yt * np.sin(theta)) * chord
        yl = (yc - yt * np.cos(theta)) * chord

        # Wrap: upper surface TE→LE, lower surface LE→TE
        upper = np.column_stack([xu[::-1], yu[::-1]])
        lower = np.column_stack([xl[1:],   yl[1:]])
        verts = np.vstack([upper, lower])

        return cls(verts, chord=chord)

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    @property
    def vertices(self):
        """Airfoil contour vertices, shape (N, 2)."""
        return self._verts.copy()

    @property
    def leading_edge(self):
        """Approximate leading-edge point (min x)."""
        return self._verts[self._le_idx].copy()

    @property
    def trailing_edge(self):
        """Approximate trailing-edge point (mean of first and last vertex)."""
        return 0.5 * (self._verts[0] + self._verts[-1])

    def upper_surface(self):
        """Return upper surface vertices (TE → LE)."""
        return self._verts[:self._le_idx + 1]

    def lower_surface(self):
        """Return lower surface vertices (LE → TE)."""
        return self._verts[self._le_idx:]

    def camber_line(self, n_points=100):
        """Estimate the camber line as the midpoint between upper/lower surfaces.

        Interpolates upper and lower surfaces at the same chordwise x stations.

        Returns
        -------
        xc : ndarray, shape (n_points,)
        yc : ndarray, shape (n_points,)
        """
        upper = self.upper_surface()
        lower = self.lower_surface()

        # Sort by x (upper TE→LE may be reversed)
        upper = upper[np.argsort(upper[:, 0])]
        lower = lower[np.argsort(lower[:, 0])]

        x_min = max(upper[:, 0].min(), lower[:, 0].min())
        x_max = min(upper[:, 0].max(), lower[:, 0].max())
        xc    = np.linspace(x_min, x_max, n_points)

        yu = np.interp(xc, upper[:, 0], upper[:, 1])
        yl = np.interp(xc, lower[:, 0], lower[:, 1])
        yc = 0.5 * (yu + yl)
        return xc, yc

    def thickness_distribution(self, n_points=100):
        """Chord-normal thickness as a function of chordwise position.

        Returns
        -------
        xc : ndarray
        t  : ndarray
            Local thickness (perpendicular distance between surfaces).
        """
        upper = self.upper_surface()
        lower = self.lower_surface()
        upper = upper[np.argsort(upper[:, 0])]
        lower = lower[np.argsort(lower[:, 0])]

        x_min = max(upper[:, 0].min(), lower[:, 0].min())
        x_max = min(upper[:, 0].max(), lower[:, 0].max())
        xc    = np.linspace(x_min, x_max, n_points)

        yu = np.interp(xc, upper[:, 0], upper[:, 1])
        yl = np.interp(xc, lower[:, 0], lower[:, 1])
        return xc, yu - yl

    def normalise(self):
        """Return a new AirfoilSDF with chord = 1, LE at origin."""
        le = self.leading_edge
        verts = (self._verts - le) / self.chord
        return AirfoilSDF(verts, chord=1.0)

    def scale(self, new_chord):
        """Return a new AirfoilSDF scaled to a different chord length."""
        verts = self._verts * (new_chord / self.chord)
        return AirfoilSDF(verts, chord=new_chord)

    def translate(self, dx, dy):
        """Return a translated copy."""
        verts = self._verts + np.array([dx, dy])
        return AirfoilSDF(verts, chord=self.chord)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"AirfoilSDF(n_verts={len(self._verts)}, "
            f"chord={self.chord:.4g}, "
            f"LE={self.leading_edge}, "
            f"TE={self.trailing_edge})"
        )
