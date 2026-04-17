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

def export_section_json(section_geometry, grid, midline_dict, filepath):
    """Export a complete section description to JSON.

    Includes:
      - Airfoil vertices
      - Per-component section properties
      - Medial axis polylines

    Parameters
    ----------
    section_geometry : BladeSectionGeometry
    grid : SDFGrid
    midline_dict : dict  {label: list of ndarray}
    filepath : str
    """
    # Airfoil
    af_verts = section_geometry.airfoil.vertices.tolist()

    # Section properties
    report = SectionPropertiesReport(section_geometry, grid)

    def _arr(a):
        return np.asarray(a).tolist()

    props_out = {}
    for label, props in report.to_dict().items():
        props_out[label] = {k: float(v) if k != "label" else v
                            for k, v in props.items()}

    # Medial axes
    midlines_out = {}
    for label, polylines in midline_dict.items():
        midlines_out[label] = [_arr(p) for p in polylines]

    payload = {
        "airfoil_vertices": af_verts,
        "section_properties": props_out,
        "medial_axes": midlines_out,
    }

    with open(filepath, "w") as fh:
        json.dump(payload, fh, indent=2)
    return filepath
