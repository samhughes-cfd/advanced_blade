from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while _p.name != "blade_precompute" and _p.parent != _p:
    _p = _p.parent
if _p.name == "blade_precompute":
    sys.path.insert(0, str(_p.parent))

from blade_precompute.section_properties import SectionAnalysis
from blade_precompute.section_properties.__main__ import _smoke_section


def _plot_section(section, res, *, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("This example requires matplotlib. Install via requirements.txt") from e

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for sub in section.subcomponents:
        pts = np.asarray(sub.midsurface_coords, dtype=np.float64)
        ax.plot(pts[:, 0], pts[:, 1], ".-", lw=1.5, ms=5, label=sub.name)

    def mark(pt: np.ndarray, label: str, color: str) -> None:
        p = np.asarray(pt, dtype=np.float64).ravel()
        ax.plot([p[0]], [p[1]], marker="x", ms=9, mew=2, color=color)
        ax.annotate(label, (p[0], p[1]), textcoords="offset points", xytext=(6, 6), color=color)

    mark(res.elastic_center, "elastic", "C3")
    mark(res.shear_center, "shear", "C4")
    mark(res.mass_center, "mass", "C5")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("section_properties smoke section (midsurface)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot the section_properties CLI smoke section.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "section_properties_smoke.png",
        help="Output image path (default: examples/output/section_properties_smoke.png).",
    )
    args = p.parse_args()

    section = _smoke_section()
    res = SectionAnalysis().solve(section)
    _plot_section(section, res, out_path=args.out.resolve())
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()

