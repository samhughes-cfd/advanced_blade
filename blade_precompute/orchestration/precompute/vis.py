"""Read-only visualisation for precompute stage result containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    SectionGeometryOutputs,
    SectionOptimisationOutputs,
    SectionPropertiesOutputs,
    SectionShellModelOutputs,
)


def _display_png_paths(paths: list[Path], *, title: str) -> None:
    """Best-effort: show PNGs from disk (requires matplotlib)."""
    if not paths:
        return
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(paths)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        ax_list = [axes]
    else:
        ax_list = np.atleast_1d(axes).ravel()
    for i, p in enumerate(paths):
        ax = ax_list[i]
        try:
            im = mpimg.imread(str(p))
            ax.imshow(im)
        except Exception:
            ax.text(0.5, 0.5, str(p), ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        ax.set_title(p.name, fontsize=7)
    for j in range(n, len(ax_list)):
        ax_list[j].set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_section_properties_station(section_def: Any, res: Any, out_png: Path) -> None:
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


# Uniform spanwise sample count for all global_beam_model PNGs (Gauss, nodal, and section recovery
# series are linearly interpolated to this grid so x-axes are aligned across figures).
BEAM_SPAN_PLOT_SAMPLES: int = 400


def write_beam_model_pngs(
    out_stage: Path,
    model: Any,
    res: Any,
    loads: Any,
) -> list[Path]:
    """Generate standard global_beam_model PNGs; returns written paths."""
    png_paths: list[Path] = []
    try:
        from blade_precompute.global_beam_model.interface import plot as bmplot

        import matplotlib.pyplot as plt
    except ImportError:
        return png_paths

    z_u = bmplot.span_abscissa_union(res, model, loads, BEAM_SPAN_PLOT_SAMPLES)

    figs: list[tuple[str, Any]] = []
    fig, _ = bmplot.plot_centerline_ref_def(model, res)
    figs.append(("beam_centerline.png", fig))
    fig, _ = bmplot.plot_spanwise_resultants(res, z_abscissa=z_u)
    figs.append(("beam_resultants.png", fig))
    fig, _ = bmplot.plot_spanwise_strains(res, z_abscissa=z_u)
    figs.append(("beam_strains.png", fig))
    fig, _ = bmplot.plot_spanwise_resultants_nodal(res, z_abscissa=z_u)
    figs.append(("beam_resultants_nodal.png", fig))
    fig, _ = bmplot.plot_spanwise_strains_nodal(res, z_abscissa=z_u)
    figs.append(("beam_strains_nodal.png", fig))
    for name, make_fig in (
        ("beam_section_stress.png", lambda: bmplot.plot_spanwise_section_stress(res, z_abscissa=z_u)),
        ("beam_section_strain_laminate.png", lambda: bmplot.plot_spanwise_section_strain_laminate(res, z_abscissa=z_u)),
        ("beam_section_tsai_wu.png", lambda: bmplot.plot_spanwise_section_tsai_wu(res, z_abscissa=z_u)),
        ("beam_section_von_mises_fi.png", lambda: bmplot.plot_spanwise_section_von_mises_fi(res, z_abscissa=z_u)),
        ("beam_section_delamination_fi.png", lambda: bmplot.plot_spanwise_section_delamination_fi(res, z_abscissa=z_u)),
        ("beam_section_stress_secframe.png", lambda: bmplot.plot_spanwise_section_stress_secframe(res, z_abscissa=z_u)),
        ("beam_section_d_tsai_wu_dz.png", lambda: bmplot.plot_spanwise_section_d_tsai_wu_dz(res, z_abscissa=z_u)),
        (
            "beam_section_tsai_wu_fi_heatmap_gp.png",
            lambda: bmplot.plot_spanwise_section_tsai_wu_fi_heatmap(res, source="gp", z_abscissa=z_u),
        ),
        (
            "beam_section_tsai_wu_fi_heatmap_nodal.png",
            lambda: bmplot.plot_spanwise_section_tsai_wu_fi_heatmap(res, source="nodal", z_abscissa=z_u),
        ),
    ):
        try:
            fig, _ = make_fig()
            figs.append((name, fig))
        except ValueError:
            pass
    fig, _ = bmplot.plot_nodal_warping(model, res, z_abscissa=z_u)
    figs.append(("beam_warping.png", fig))
    fig, _ = bmplot.plot_iteration_history(res)
    figs.append(("beam_iteration_history.png", fig))
    fig, _ = bmplot.plot_reactions(res)
    figs.append(("beam_reactions.png", fig))
    fig, _ = bmplot.plot_distributed_loads(model, loads, z_abscissa=z_u)
    figs.append(("beam_distributed_loads.png", fig))

    for name, fig in figs:
        p = (out_stage / name).resolve()
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(p)
    return png_paths


def write_section_optimisation_pngs(
    out_stage: Path,
    z: NDArray[np.float64],
    dv0: Any,
    opt_res: Any | None,
) -> list[Path]:
    from blade_precompute.section_optimisation.interface import plot as dplot

    import matplotlib.pyplot as plt

    png_paths: list[Path] = []
    fig, _ = dplot.plot_design_vector_vs_span(z, dv0, title="Initial design vector (precompute)")
    p = (out_stage / "design_vector.png").resolve()
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    png_paths.append(p)
    if opt_res is not None:
        fig, _ = dplot.plot_design_vector_vs_span(
            z,
            opt_res.dv_opt,
            dv_compare=dv0,
            title="Optimised vs initial design vector (precompute)",
        )
        p2 = (out_stage / "design_vector_optimised.png").resolve()
        fig.savefig(p2, dpi=170, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(p2)
        fig, _ = dplot.plot_optimisation_history(opt_res)
        p3 = (out_stage / "section_optimisation_history.png").resolve()
        fig.savefig(p3, dpi=170, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(p3)
    return png_paths


class SectionGeometryOutputsVis:
    def __init__(self, results: SectionGeometryOutputs) -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        _display_png_paths(list(self._results.png_paths), title="section_geometry")


class SectionPropertiesOutputsVis:
    def __init__(self, results: SectionPropertiesOutputs) -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        _display_png_paths(list(self._results.png_paths), title="section_properties")


class BeamModelOutputsVis:
    def __init__(self, results: BeamModelOutputs) -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        _display_png_paths(list(self._results.png_paths), title="global_beam_model")


class SectionOptimisationOutputsVis:
    def __init__(self, results: SectionOptimisationOutputs) -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        _display_png_paths(list(self._results.png_paths), title="section_optimisation")


class SectionShellModelOutputsVis:
    def __init__(self, results: SectionShellModelOutputs) -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        title = "section_shell_model (skipped)" if self._results.skipped else "section_shell_model"
        _display_png_paths(list(self._results.png_paths), title=title)
