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
    from blade_precompute.section_properties.vis import (
        plot_section_properties_station as plot_section_properties_station_impl,
    )

    plot_section_properties_station_impl(section_def, res, out_png)


# Default uniform spanwise sample count for global_beam_model PNGs when callers omit ``span_plot_samples``.
_DEFAULT_BEAM_PNG_SPAN_SAMPLES: int = 400


def write_beam_model_pngs(
    out_stage: Path,
    model: Any,
    res: Any,
    loads: Any,
    *,
    span_plot_samples: int | None = None,
) -> list[Path]:
    """Generate standard global_beam_model PNGs; returns written paths."""
    png_paths: list[Path] = []
    try:
        from blade_precompute.global_beam_model.interface import plot as bmplot

        import matplotlib.pyplot as plt
    except ImportError:
        return png_paths

    n_samp = int(span_plot_samples) if span_plot_samples is not None else _DEFAULT_BEAM_PNG_SPAN_SAMPLES
    z_u = bmplot.span_abscissa_union(res, model, loads, n_samp)

    figs: list[tuple[str, Any]] = []
    fig, _ = bmplot.plot_centerline_ref_def(model, res)
    figs.append(("beam_centerline.png", fig))
    fig, _ = bmplot.plot_spanwise_resultants(res)
    figs.append(("beam_resultants.png", fig))
    fig, _ = bmplot.plot_spanwise_strains(res)
    figs.append(("beam_strains.png", fig))
    for name, make_fig in (
        ("beam_section_stress.png", lambda: bmplot.plot_spanwise_section_stress(res)),
        ("beam_section_stress_nodal.png", lambda: bmplot.plot_spanwise_section_stress_nodal(res)),
        ("beam_section_strain_laminate.png", lambda: bmplot.plot_spanwise_section_strain_laminate(res)),
        ("beam_section_hashin_fi.png", lambda: bmplot.plot_spanwise_section_hashin_fi(res)),
        ("beam_section_von_mises_fi.png", lambda: bmplot.plot_spanwise_section_von_mises_fi(res)),
        ("beam_section_stress_secframe.png", lambda: bmplot.plot_spanwise_section_stress_secframe(res)),
        ("beam_section_d_hashin_fi_dz.png", lambda: bmplot.plot_spanwise_section_d_hashin_fi_dz(res)),
        (
            "beam_section_hashin_fi_heatmap_gp.png",
            lambda: bmplot.plot_spanwise_section_hashin_fi_heatmap(res, source="gp"),
        ),
        (
            "beam_section_hashin_fi_heatmap_nodal.png",
            lambda: bmplot.plot_spanwise_section_hashin_fi_heatmap(res, source="nodal"),
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

    # Same figures as beam_resultants.png / beam_strains.png (GP markers + nodal projection);
    # duplicate filenames match historical precompute output naming.
    _png_aliases: dict[str, tuple[str, ...]] = {
        "beam_resultants.png": ("beam_resultants_nodal.png",),
        "beam_strains.png": ("beam_strains_nodal.png",),
    }
    for name, fig in figs:
        p = (out_stage / name).resolve()
        fig.savefig(p, dpi=170, bbox_inches="tight")
        png_paths.append(p)
        for alias in _png_aliases.get(name, ()):
            pa = (out_stage / alias).resolve()
            fig.savefig(pa, dpi=170, bbox_inches="tight")
            png_paths.append(pa)
        plt.close(fig)
    return png_paths


def write_section_optimisation_pngs(
    out_stage: Path,
    z: NDArray[np.float64],
    dv0: Any,
    opt_res: Any | None,
    *,
    ev0: Any | None = None,
    ev_opt: Any | None = None,
    problem: Any | None = None,
    composite_subcomp_names: Any | None = None,
) -> list[Path]:
    from blade_precompute.section_optimisation.interface import plot as dplot

    import matplotlib.pyplot as plt

    def _save(name: str, fig: Any) -> None:
        p = (out_stage / name).resolve()
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(p)

    png_paths: list[Path] = []
    fig, _ = dplot.plot_design_vector_vs_span(z, dv0, title="Initial design vector (precompute)")
    _save("design_vector.png", fig)
    ev_final = ev_opt
    if opt_res is not None and ev_final is None and getattr(opt_res, "evaluations", None):
        evs = opt_res.evaluations
        if evs:
            ev_final = evs[-1]
    if opt_res is not None:
        fig, _ = dplot.plot_design_vector_vs_span(
            z,
            opt_res.dv_opt,
            dv_compare=dv0,
            title="Optimised vs initial design vector (precompute)",
        )
        _save("design_vector_optimised.png", fig)
        fig, _ = dplot.plot_optimisation_history(opt_res)
        _save("section_optimisation_history.png", fig)
        fig, _ = dplot.plot_optimisation_history(opt_res, title="Section optimisation convergence history")
        _save("section_optimisation_convergence_history.png", fig)

    if ev0 is not None:
        try:
            fig, _ = dplot.plot_max_fi_vs_span(
                z, ev0, ev_final, problem=problem, title="Max failure index vs span"
            )
            _save("max_fi_vs_span.png", fig)
        except Exception:
            plt.close("all")
        try:
            fig, _ = dplot.plot_fi_reserve_vs_span(z, ev0, ev_final, problem=problem)
            _save("fi_reserve_vs_span.png", fig)
        except Exception:
            plt.close("all")
        try:
            if composite_subcomp_names is not None:
                fig, _ = dplot.plot_governing_subcomp_hashin_vs_span(
                    z, ev0, composite_subcomp_names, problem=problem
                )
                _save("governing_subcomp_hashin.png", fig)
        except Exception:
            plt.close("all")
        try:
            if ev_final is not None:
                fig, _ = dplot.plot_mitc4_vs_hashin_span(z, ev_final, problem=problem)
                _save("mitc4_vs_hashin_span.png", fig)
            else:
                fig, _ = dplot.plot_mitc4_vs_hashin_span(z, ev0, problem=problem)
                _save("mitc4_vs_hashin_span.png", fig)
        except Exception:
            plt.close("all")
        try:
            fig, _ = dplot.plot_resultants_with_max_fi(
                z, ev0, ev_final, problem=problem, title="Beam resultants and max Hashin FI vs span"
            )
            _save("resultants_with_max_fi.png", fig)
        except Exception:
            plt.close("all")
        try:
            fig, _ = dplot.plot_fi_span_heatmap(z, ev0, ev_final, problem=problem)
            _save("fi_span_heatmap.png", fig)
        except Exception:
            plt.close("all")
        try:
            fig, _ = dplot.plot_panel_buckling_fi_vs_span(z, ev0, ev_final, problem=problem)
            _save("panel_buckling_fi_vs_span.png", fig)
        except Exception:
            plt.close("all")
        try:
            fig, _ = dplot.plot_k7_condition_summary(ev0, ev_final, problem=problem)
            _save("k7_condition_summary.png", fig)
        except Exception:
            plt.close("all")
        try:
            dv_share = opt_res.dv_opt if opt_res is not None else dv0
            fig, _ = dplot.plot_thickness_share_vs_span(z, dv_share, problem=problem)
            _save("thickness_normalised_vs_span.png", fig)
        except Exception:
            plt.close("all")
        if opt_res is not None and problem is not None:
            try:
                fig, _ = dplot.plot_optimisation_slack_stiffness_history(opt_res, problem)
                _save("section_optimisation_slack_stiffness_history.png", fig)
            except Exception:
                plt.close("all")
            try:
                fig, _ = dplot.plot_optimisation_objective_dual_axis(opt_res, problem)
                _save("optimisation_objective_dual_axis.png", fig)
            except Exception:
                plt.close("all")
            try:
                if len(getattr(opt_res, "evaluations", []) or []) > 1:
                    fig, _ = dplot.plot_fi_vs_span_per_iteration(z, opt_res, problem=problem)
                    _save("fi_vs_span_per_iteration.png", fig)
            except Exception:
                plt.close("all")
        if opt_res is not None:
            try:
                fig, _ = dplot.plot_thickness_delta_vs_span(z, dv0, opt_res.dv_opt, problem=problem)
                _save("thickness_delta_vs_span.png", fig)
            except Exception:
                plt.close("all")
        try:
            ev_nr = ev_final if ev_final is not None else ev0
            fig, _ = dplot.plot_beam_nr_residual_tail(ev_nr, problem=problem)
            _save("beam_nr_residual_tail.png", fig)
        except Exception:
            plt.close("all")

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
