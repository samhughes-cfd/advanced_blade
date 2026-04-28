"""Visualisation helpers for section_properties results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def plot_section_properties_station(section_def: Any, res: Any, out_png: Path) -> None:
    """Plot one section station with elastic/shear/mass centers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("section_properties station plots require matplotlib.") from e

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for sub in section_def.subcomponents:
        pts = np.asarray(sub.midsurface_coords, dtype=np.float64)
        ax.plot(pts[:, 0], pts[:, 1], ".-", lw=1.5, ms=5, label=sub.name)

    def mark(pt: NDArray[np.float64], label: str, color: str) -> None:
        p = np.asarray(pt, dtype=np.float64).ravel()
        ax.plot([p[0]], [p[1]], marker="x", ms=9, mew=2, color=color)
        ax.annotate(label, (p[0], p[1]), textcoords="offset points", xytext=(6, 6), color=color)

    mark(res.elastic_center, "elastic", "C3")
    mark(res.shear_center, "shear", "C4")
    mark(res.mass_center, "mass", "C5")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"section_properties @ z={float(section_def.station_z):.3g} m")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
