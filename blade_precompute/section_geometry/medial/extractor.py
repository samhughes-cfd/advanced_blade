"""
medial.extractor
================
MedialAxisExtractor: SDF-based medial axis / midline recovery.

The extractor operates entirely on pre-evaluated SDF arrays (ndarray, shape
(ny, nx)) together with the corresponding SDFGrid for coordinate mapping.
"""

import numpy as np
from scipy import ndimage

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    import skfmm
    _HAS_SKFMM = True
except ImportError:
    _HAS_SKFMM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redistance(phi, dx, dy):
    """Re-normalise a drifted SDF using fast-marching (scikit-fmm).

    Raises ImportError if scikit-fmm is unavailable.
    """
    if not _HAS_SKFMM:
        raise ImportError(
            "scikit-fmm is required for Eikonal redistancing. "
            "Install with: pip install scikit-fmm"
        )
    return skfmm.distance(phi, dx=[dy, dx])


def _grad_magnitude(phi, dx, dy):
    """|∇φ| via central differences."""
    gx = np.gradient(phi, dx, axis=1)
    gy = np.gradient(phi, dy, axis=0)
    return np.sqrt(gx**2 + gy**2)


def _pixels_to_coords(rows, cols, grid):
    """Convert (row, col) pixel indices to physical (x, y) coordinates."""
    x = grid.X[0, 0] + cols * grid.dx
    y = grid.Y[0, 0] + rows * grid.dy
    return x, y


def _order_skeleton_points(rows, cols):
    """Attempt to order skeleton points as a polyline using nearest-neighbour.

    Works well for simply-connected, roughly 1-D structures (thin walls,
    webs, caps).  For branched structures, returns the longest branch.
    """
    if len(rows) == 0:
        return np.array([]), np.array([])

    pts = np.column_stack([cols.astype(float), rows.astype(float)])
    n   = len(pts)

    # Start from the leftmost point
    start = int(np.argmin(pts[:, 0]))
    ordered = [start]
    remaining = set(range(n))
    remaining.discard(start)
    current = start

    while remaining:
        dists = np.linalg.norm(pts[list(remaining)] - pts[current], axis=1)
        nearest_idx = np.argmin(dists)
        nearest = list(remaining)[nearest_idx]
        # Break if the nearest neighbour is too far (branch gap)
        if dists[nearest_idx] > 3.0:  # pixels
            break
        ordered.append(nearest)
        remaining.discard(nearest)
        current = nearest

    ordered = np.array(ordered)
    return rows[ordered], cols[ordered]


def _prune_short_branches(skel_mask, min_branch_pixels=5):
    """Remove isolated short branches from a binary skeleton.

    Iteratively removes endpoints (pixels with only one neighbour) until
    all remaining branches exceed the minimum length.
    """
    skel = skel_mask.astype(bool).copy()
    kernel = np.ones((3, 3), dtype=int)

    for _ in range(min_branch_pixels):
        # Count neighbours for each skeleton pixel
        neighbour_count = ndimage.convolve(skel.astype(int), kernel, mode="constant") - skel.astype(int)
        # Endpoints: skeleton pixels with exactly 1 neighbour
        endpoints = skel & (neighbour_count == 1)
        skel[endpoints] = False

    return skel


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MedialAxisExtractor:
    """Extracts the medial axis (midline/midsurface) from an SDF field.

    Parameters
    ----------
    grid : SDFGrid
        The background grid.
    grad_threshold : float
        |∇φ| threshold below which a point is flagged as a medial candidate.
        Default 0.95  (5% below the Eikonal ideal of 1.0).
    redistance : bool
        Whether to apply Eikonal redistancing before extraction
        (recommended for fields built from repeated CSG ops).
        Requires scikit-fmm.
    min_branch_pixels : int
        Skeleton branches shorter than this (in pixels) are pruned.
        Default 10.
    """

    def __init__(self, grid, grad_threshold=0.95,
                 redistance=False, min_branch_pixels=10):
        self.grid             = grid
        self.grad_threshold   = float(grad_threshold)
        self.redistance       = bool(redistance)
        self.min_branch_pixels = int(min_branch_pixels)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, phi, return_skeleton_mask=False):
        """Extract the medial axis of the region phi < 0.

        Parameters
        ----------
        phi : ndarray, shape (ny, nx)
            Pre-evaluated SDF field.
        return_skeleton_mask : bool
            If True, also return the raw binary skeleton array.

        Returns
        -------
        polylines : list of ndarray, shape (N, 2)
            Ordered (x, y) coordinates of each medial branch in physical units.
        skeleton : ndarray (bool), shape (ny, nx)   [only if return_skeleton_mask]
            Binary skeleton on the grid.
        """
        dx, dy = self.grid.dx, self.grid.dy

        # Optional redistancing
        if self.redistance:
            phi = _redistance(phi, dx, dy)

        # --- Step 1: medial candidate mask ---
        gm            = _grad_magnitude(phi, dx, dy)
        interior_mask = phi < 0.0
        medial_cand   = interior_mask & (gm < self.grad_threshold)

        # --- Step 2: morphological thinning to 1-px skeleton ---
        if _HAS_SKIMAGE:
            skeleton = skeletonize(medial_cand)
        else:
            # Fallback: distance-transform peak finding
            skeleton = self._fallback_skeleton(phi, interior_mask)

        # --- Step 3: prune short branches ---
        skeleton = _prune_short_branches(skeleton, self.min_branch_pixels)

        # --- Step 4: connected components → separate polylines ---
        labelled, n_labels = ndimage.label(skeleton)
        polylines = []
        for label_id in range(1, n_labels + 1):
            component = labelled == label_id
            rows, cols = np.where(component)
            rows_ord, cols_ord = _order_skeleton_points(rows, cols)
            if len(rows_ord) < 2:
                continue
            x, y = _pixels_to_coords(rows_ord, cols_ord, self.grid)
            polylines.append(np.column_stack([x, y]))

        if return_skeleton_mask:
            return polylines, skeleton
        return polylines

    def extract_for_section(self, section_geometry):
        """Extract medial axes for all subcomponents of a BladeSectionGeometry.

        Evaluates each subcomponent's SDF on the grid and runs extraction.

        Parameters
        ----------
        section_geometry : BladeSectionGeometry

        Returns
        -------
        results : dict
            {label: list of (N, 2) ndarray polylines}
        """
        results = {}
        for label in section_geometry:
            phi = self.grid.eval(section_geometry[label])
            polylines = self.extract(phi)
            results[label] = polylines
        return results

    # ------------------------------------------------------------------
    # Fallback skeleton (no scikit-image)
    # ------------------------------------------------------------------

    def _fallback_skeleton(self, phi, interior_mask):
        """Distance-transform skeleton via local maxima of -phi inside region.

        Less accurate than Lee's thinning but has no extra dependencies.
        """
        # Inside region: -phi gives distance to boundary (positive inside)
        dist = np.where(interior_mask, -phi, 0.0)
        # Local maxima of the distance transform = centres of inscribed circles
        # Detect via maximum filter
        footprint = np.ones((5, 5), dtype=bool)
        local_max  = ndimage.maximum_filter(dist, footprint=footprint) == dist
        skeleton   = interior_mask & local_max & (dist > 0)
        return skeleton

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def midline_length(self, polyline):
        """Arc-length of a polyline in physical units."""
        if len(polyline) < 2:
            return 0.0
        diffs = np.diff(polyline, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def midline_curvature(self, polyline):
        """Approximate curvature κ at each interior point of a polyline.

        Returns
        -------
        s : ndarray
            Arc-length stations.
        kappa : ndarray
            Signed curvature at each point.
        """
        if len(polyline) < 3:
            return np.array([0.0]), np.array([0.0])

        dx = np.gradient(polyline[:, 0])
        dy = np.gradient(polyline[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-30)**1.5
        ds    = np.sqrt(dx**2 + dy**2)
        s     = np.concatenate([[0.0], np.cumsum(ds[:-1])])
        return s, kappa


# ---------------------------------------------------------------------------
# One-shot convenience function
# ---------------------------------------------------------------------------

def extract_midline(sdf_callable, grid, grad_threshold=0.95,
                    redistance=False, min_branch_pixels=10):
    """Extract the medial axis of a single SDF region.

    Parameters
    ----------
    sdf_callable : callable (x, y) → ndarray
    grid : SDFGrid
    grad_threshold, redistance, min_branch_pixels : see MedialAxisExtractor

    Returns
    -------
    polylines : list of ndarray, shape (N, 2)
    """
    extractor = MedialAxisExtractor(
        grid,
        grad_threshold=grad_threshold,
        redistance=redistance,
        min_branch_pixels=min_branch_pixels,
    )
    phi = grid.eval(sdf_callable)
    return extractor.extract(phi)
