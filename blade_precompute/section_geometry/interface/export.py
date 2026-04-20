"""
interface.export
================
Data export utilities for section geometry and medial axes.
"""

import json
import csv
import numpy as np


class SectionPropertiesReport:
    """Compute and collect section properties for all subcomponents.

    Parameters
    ----------
    section_geometry : BladeSectionGeometry
    grid : SDFGrid

    Usage
    -----
        report = SectionPropertiesReport(bsg, grid)
        df     = report.to_dataframe()
        report.to_json("section_props.json")
    """

    def __init__(self, section_geometry, grid):
        self._bsg  = section_geometry
        self._grid = grid
        self._data = {}
        self._compute()

    def _compute(self):
        grid = self._grid
        for label in self._bsg:
            phi = grid.eval(self._bsg[label])
            props = grid.section_properties(phi)
            # Also store midline bounding box
            props["label"] = label
            self._data[label] = props

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def __getitem__(self, label):
        return self._data[label]

    def __iter__(self):
        return iter(self._data)

    def summary(self):
        """Return a formatted string summary."""
        lines = ["=" * 64, f"{'Section Properties':^64}", "=" * 64]
        for label, props in self._data.items():
            lines.append(f"\n  [{label}]")
            lines.append(f"    Area   : {props['area']:.6g}")
            lines.append(f"    Centroid: ({props['cx']:.6g}, {props['cy']:.6g})")
            lines.append(f"    Ixx    : {props['Ixx']:.6g}")
            lines.append(f"    Iyy    : {props['Iyy']:.6g}")
            lines.append(f"    Ixy    : {props['Ixy']:.6g}")
        lines.append("\n" + "=" * 64)
        return "\n".join(lines)

    def to_dict(self):
        """Return all properties as a plain dict."""
        return dict(self._data)

    def to_json(self, filepath, *, job_meta=None):
        """Serialise to JSON file.

        Parameters
        ----------
        filepath : str or Path
            Output path.
        job_meta : dict, optional
            Merged under top-level key ``job_meta`` (orchestration provenance).
        """
        # Convert numpy types to Python natives
        def _convert(obj):
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        data = {
            label: {k: _convert(v) for k, v in props.items()}
            for label, props in self._data.items()
        }
        if job_meta is not None:
            data["job_meta"] = _convert(job_meta)
        with open(filepath, "w") as fh:
            json.dump(data, fh, indent=2)
        return filepath

    def to_csv(self, filepath):
        """Write properties to a flat CSV file."""
        fieldnames = ["label", "area", "cx", "cy", "Ixx", "Iyy", "Ixy",
                      "r_gyr_x", "r_gyr_y"]
        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            for props in self._data.values():
                row = {k: (float(props[k]) if k != "label" else props[k])
                       for k in fieldnames}
                writer.writerow(row)
        return filepath

    def to_dataframe(self):
        """Return a pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). "
                              "Install with: pip install pandas")
        rows = []
        for label, props in self._data.items():
            row = {"label": label}
            row.update({k: float(v) for k, v in props.items()
                        if k != "label"})
            rows.append(row)
        return pd.DataFrame(rows).set_index("label")


# ---------------------------------------------------------------------------
# Midline export
# ---------------------------------------------------------------------------

def export_midlines_csv(midline_dict, filepath):
    """Write medial axis polylines to a CSV file.

    Parameters
    ----------
    midline_dict : dict
        {label: list of (N, 2) ndarray} as returned by
        MedialAxisExtractor.extract_for_section().
    filepath : str
        Output CSV path.

    CSV columns: label, branch_id, x, y
    """
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "branch_id", "x", "y"])
        for label, polylines in midline_dict.items():
            for bid, poly in enumerate(polylines):
                for pt in poly:
                    writer.writerow([label, bid, float(pt[0]), float(pt[1])])
    return filepath


# ---------------------------------------------------------------------------
# Full section JSON export
# ---------------------------------------------------------------------------

def _multicell_host(section_geometry):
    """Resolve MultiCellSection (or wrapped BladeSectionGeometry._mcs)."""
    return getattr(section_geometry, "_mcs", None) or section_geometry


def _zero_contour_to_nested(grid, sdf_callable):
    """Extract φ=0 as list of polylines (each Nx2), JSON-ready."""
    phi = grid.eval(sdf_callable)
    segs = grid.zero_contour(phi)
    return [np.asarray(s, dtype=float).tolist() for s in segs]


def export_section_json(
    section_geometry,
    grid,
    midline_dict,
    filepath,
    *,
    include_component_zero_contours=False,
    include_geometry_detail=True,
):
    """Export a complete section description to JSON.

    Includes:
      - Reference airfoil contour vertices (chord-frame ``AirfoilSDF`` polyline)
      - Per-component section properties
      - Medial axis polylines (flat ``medial_axes`` dict, for compatibility)
      - When ``include_geometry_detail`` is True, a nested ``geometry`` block:
        * ``skin``: outer mold line, inner mold line, and ``outer_skin`` medial polylines
        * ``components``: per label, boundary loops (φ=0 of that solid) and medial polylines
      - Optional legacy flat ``component_zero_contours`` (same loops as
        ``geometry["components"][label]["boundary"]`` when both are enabled).

    Parameters
    ----------
    section_geometry : BladeSectionGeometry
    grid : SDFGrid
    midline_dict : dict  {label: list of ndarray}
    filepath : str
    include_component_zero_contours : bool
        If True, also write flat ``component_zero_contours`` (matplotlib contours).
    include_geometry_detail : bool
        If True, populate ``geometry`` with skin mold lines and per-component
        boundaries plus medials. If False, medial polylines are still written
        to the flat ``medial_axes`` key when ``midline_dict`` is non-empty.
    """
    # Airfoil (reference section shape; not the twisted boundary SDFs)
    af_verts = section_geometry.airfoil.vertices.tolist()

    # Section properties
    report = SectionPropertiesReport(section_geometry, grid)

    def _arr(a):
        return np.asarray(a).tolist()

    props_out = {}
    for label, props in report.to_dict().items():
        props_out[label] = {k: float(v) if k != "label" else v
                            for k, v in props.items()}

    # Medial axes (flat, backward compatible)
    midlines_out = {}
    for label, polylines in midline_dict.items():
        midlines_out[label] = [_arr(p) for p in polylines]

    payload = {
        "schema_version": 2,
        "airfoil_vertices": af_verts,
        "reference_airfoil_vertices": af_verts,
        "section_properties": props_out,
        "medial_axes": midlines_out,
    }

    if include_geometry_detail:
        geometry: dict = {"components": {}}
        host = _multicell_host(section_geometry)
        o_sdf = getattr(host, "_skin_outer_boundary_sdf", None)
        i_sdf = getattr(host, "_skin_inner_boundary_sdf", None)

        skin_block = {}
        if o_sdf is not None and i_sdf is not None:
            skin_block["outer_boundary"] = _zero_contour_to_nested(grid, o_sdf)
            skin_block["inner_boundary"] = _zero_contour_to_nested(grid, i_sdf)
        else:
            skin_block["outer_boundary"] = None
            skin_block["inner_boundary"] = None
        skin_block["medial_axes"] = midlines_out.get("outer_skin", [])
        geometry["skin"] = skin_block

        for label in section_geometry.labels:
            geometry["components"][label] = {
                "boundary": _zero_contour_to_nested(
                    grid, section_geometry[label]
                ),
                "medial_axes": midlines_out.get(label, []),
            }
        payload["geometry"] = geometry

    if include_component_zero_contours:
        contours_out = {}
        for label in section_geometry.labels:
            phi = grid.eval(section_geometry[label])
            segs = grid.zero_contour(phi)
            contours_out[label] = [s.tolist() for s in segs]
        payload["component_zero_contours"] = contours_out

    with open(filepath, "w") as fh:
        json.dump(payload, fh, indent=2)
    return filepath
