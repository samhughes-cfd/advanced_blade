from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_optimisation.__main__ import _objective_from_cli
from blade_precompute.section_optimisation.core.types import OptimizationObjective

from blade_precompute.orchestration import (
    MIDLINE_CONTRACT_VERSION,
    PrecomputeOrchestrationContext,
    assert_grid_phi_finite,
    build_section_view,
    load_component_materials_json,
    midline_series_contract_doc,
    resolve_system_type,
    section_boundary_stub_from_labels,
    validate_component_indices,
)

_REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Top-level control settings (edit here, not via CLI)
# ---------------------------------------------------------------------------
#
# Geometry grid (section_geometry stage): if `*_N` is None, keep source table count.
GRID_GEOMETRY_Z_MIN: float | None = None
GRID_GEOMETRY_Z_MAX: float | None = None
GRID_GEOMETRY_N: int | None = None
#
# Structural/design station grid (section_properties/beam/design/buckling stages):
# if `*_N` is None, keep YAML station count.
GRID_STRUCTURAL_Z_MIN: float | None = None
GRID_STRUCTURAL_Z_MAX: float | None = None
GRID_STRUCTURAL_N: int | None = None

# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecomputeInputs:
    spanwise_path: Path
    extreme_loads_path: Path
    span_r_z_m: NDArray[np.float64]
    chord_m: NDArray[np.float64]
    twist_deg: NDArray[np.float64]
    naca_m: NDArray[np.float64]
    naca_p: NDArray[np.float64]
    naca_xx: NDArray[np.float64]
    loads_r_z_m: NDArray[np.float64]
    q_y_Npm: NDArray[np.float64]
    q_z_Npm: NDArray[np.float64]
    m_x_Nmpm: NDArray[np.float64]


@dataclass(frozen=True)
class SectionGeometryOutputs:
    station_indices: list[int]
    station_r_z_m: list[float]
    png_paths: list[Path]
    json_paths: list[Path]


@dataclass(frozen=True)
class SectionPropertiesOutputs:
    station_z: NDArray[np.float64]
    K6: NDArray[np.float64]
    K7: NDArray[np.float64]
    results_summary_json: Path
    png_paths: list[Path]
    section_results: tuple[object, ...]
    section_definitions: tuple[object, ...]
    #: Local orthotropic skin/stringer panel screening (``section_properties``), not GBT ``section_beam_model``.
    panel_local_buckling_json: Path | None = None


@dataclass(frozen=True)
class BeamModelOutputs:
    result_json: Path
    png_paths: list[Path]


@dataclass(frozen=True)
class SectionOptimisationOutputs:
    result_json: Path
    png_paths: list[Path]


@dataclass(frozen=True)
class SectionBucklingOutputs:
    station_json_paths: list[Path]
    part_json_paths: list[Path]
    png_paths: list[Path]
    summary_json: Path


@dataclass(frozen=True)
class LinspaceSpec:
    z_min: float
    z_max: float
    n: int


@dataclass(frozen=True)
class GridConfig:
    geometry: LinspaceSpec
    structural: LinspaceSpec
    plot_station_spec: str
    n_beam_nodes: int


@dataclass(frozen=True)
class SectionGeometryParams:
    inp: PrecomputeInputs
    out_dir: Path
    plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SectionPropertiesParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BeamModelParams:
    inp: PrecomputeInputs
    sec: SectionPropertiesOutputs
    out_dir: Path
    blade_yaml: Path
    n_beam_nodes: int
    orchestration: PrecomputeOrchestrationContext
    save_section_recovery_cache_npz: bool = False
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SectionOptimisationParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    orchestration: PrecomputeOrchestrationContext
    run_blade_optimizer: bool = False
    optimization_objective: OptimizationObjective = "min_mass"
    optimizer_max_iter: int = 120
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SectionBucklingParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    buckling_length_mode: str = "chord"
    buckling_member_length_m: float | None = None
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _station_indices(n: int, spec: str) -> list[int]:
    s = (spec or "").strip().lower()
    if not s:
        s = "root,mid,tip"
    keys = [k.strip() for k in s.split(",") if k.strip()]
    if any(k == "all" for k in keys):
        return list(range(max(0, n)))
    out: list[int] = []
    for k in keys:
        if k == "root":
            out.append(0)
        elif k == "mid":
            out.append(max(0, (n - 1) // 2))
        elif k == "tip":
            out.append(max(0, n - 1))
        elif k.startswith("every-"):
            step = int(k.split("-", 1)[1])
            if step <= 0:
                raise ValueError("every-k requires k>0.")
            out.extend(list(range(0, max(0, n), step)))
        else:
            try:
                out.append(int(k))
            except ValueError as e:
                raise ValueError(
                    f"Unknown station selector {k!r}. Use root,mid,tip,all,every-k or integer indices."
                ) from e
    # unique, stable order
    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        ii = int(np.clip(i, 0, max(0, n - 1)))
        if ii not in seen:
            uniq.append(ii)
            seen.add(ii)
    return uniq


def _linspace_from_spec(spec: LinspaceSpec) -> NDArray[np.float64]:
    n = int(spec.n)
    if n < 1:
        raise ValueError("LinspaceSpec.n must be >= 1.")
    return np.linspace(float(spec.z_min), float(spec.z_max), n, dtype=np.float64)


def _interp_series(z_src: NDArray[np.float64], y_src: NDArray[np.float64], z_dst: NDArray[np.float64]) -> NDArray[np.float64]:
    zs = np.asarray(z_src, dtype=np.float64).ravel()
    ys = np.asarray(y_src, dtype=np.float64).ravel()
    zd = np.asarray(z_dst, dtype=np.float64).ravel()
    if zs.shape[0] != ys.shape[0]:
        raise ValueError("Interpolation source length mismatch.")
    if zs.shape[0] < 2:
        return np.full(zd.shape[0], float(ys[0]) if ys.size else 0.0, dtype=np.float64)
    return np.interp(zd, zs, ys)


def _resample_precompute_inputs(inp: PrecomputeInputs, z_geom: NDArray[np.float64]) -> PrecomputeInputs:
    z0 = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    z1 = np.asarray(z_geom, dtype=np.float64).ravel()
    return dataclasses.replace(
        inp,
        span_r_z_m=z1,
        chord_m=_interp_series(z0, inp.chord_m, z1),
        twist_deg=_interp_series(z0, inp.twist_deg, z1),
        naca_m=_interp_series(z0, inp.naca_m, z1),
        naca_p=_interp_series(z0, inp.naca_p, z1),
        naca_xx=_interp_series(z0, inp.naca_xx, z1),
    )


def _resample_blade_geometry_to_z(bg: Any, z_struct: NDArray[np.float64]) -> Any:
    z_src = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_dst = np.asarray(z_struct, dtype=np.float64).ravel()
    r_ref = np.asarray(bg.r_ref, dtype=np.float64)
    kap = np.asarray(bg.kappa0, dtype=np.float64)
    airfoils = list(bg.airfoil_profiles)
    r_new = np.column_stack([_interp_series(z_src, r_ref[:, j], z_dst) for j in range(r_ref.shape[1])])
    if r_new.shape[1] >= 3:
        r_new[:, 2] = z_dst
    k_new = np.column_stack([_interp_series(z_src, kap[:, j], z_dst) for j in range(kap.shape[1])])
    af_new: list[Any] = []
    if len(airfoils) == z_src.shape[0]:
        for z in z_dst:
            i = int(np.argmin(np.abs(z_src - float(z))))
            af_new.append(airfoils[i])
    else:
        af_new = airfoils
    return dataclasses.replace(
        bg,
        z_stations=z_dst,
        r_ref=r_new,
        kappa0=k_new,
        tau0=_interp_series(z_src, np.asarray(bg.tau0, dtype=np.float64), z_dst),
        chord=_interp_series(z_src, np.asarray(bg.chord, dtype=np.float64), z_dst),
        twist=_interp_series(z_src, np.asarray(bg.twist, dtype=np.float64), z_dst),
        airfoil_profiles=af_new,
    )


def _require_columns(cols: Mapping[str, NDArray[np.float64]], required: Iterable[str], *, path: Path) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}. Present: {sorted(cols.keys())}")


def _naca4(m: float, p: float, xx: float) -> str:
    mi = int(np.clip(int(round(float(m))), 0, 9))
    pi = int(np.clip(int(round(float(p))), 0, 9))
    xxi = int(np.clip(int(round(float(xx))), 0, 99))
    return f"{mi:d}{pi:d}{xxi:02d}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_inputs(data_dir: Path) -> PrecomputeInputs:
    from data_library.plot_inputs import read_columnar_dat

    data_dir = data_dir.resolve()
    span_path = (data_dir / "blade_spanwise_distribution.dat").resolve()
    loads_path = (data_dir / "extreme_load_distribution.dat").resolve()
    if not span_path.is_file():
        raise FileNotFoundError(span_path)
    if not loads_path.is_file():
        raise FileNotFoundError(loads_path)

    span_names, span_data = read_columnar_dat(span_path)
    span_cols = {n: span_data[:, i] for i, n in enumerate(span_names)}
    _require_columns(
        span_cols,
        ("r_z_m", "chord_m", "twist_deg", "naca_m", "naca_p", "naca_xx"),
        path=span_path,
    )

    load_names, load_data = read_columnar_dat(loads_path)
    load_cols = {n: load_data[:, i] for i, n in enumerate(load_names)}
    _require_columns(load_cols, ("r_z_m", "q_y_Npm", "q_z_Npm", "m_x_Nmpm"), path=loads_path)

    return PrecomputeInputs(
        spanwise_path=span_path,
        extreme_loads_path=loads_path,
        span_r_z_m=np.asarray(span_cols["r_z_m"], dtype=np.float64),
        chord_m=np.asarray(span_cols["chord_m"], dtype=np.float64),
        twist_deg=np.asarray(span_cols["twist_deg"], dtype=np.float64),
        naca_m=np.asarray(span_cols["naca_m"], dtype=np.float64),
        naca_p=np.asarray(span_cols["naca_p"], dtype=np.float64),
        naca_xx=np.asarray(span_cols["naca_xx"], dtype=np.float64),
        loads_r_z_m=np.asarray(load_cols["r_z_m"], dtype=np.float64),
        q_y_Npm=np.asarray(load_cols["q_y_Npm"], dtype=np.float64),
        q_z_Npm=np.asarray(load_cols["q_z_Npm"], dtype=np.float64),
        m_x_Nmpm=np.asarray(load_cols["m_x_Nmpm"], dtype=np.float64),
    )


def _resolve_component_materials_path(data_dir: Path, explicit: Path | None) -> Path:
    data_dir = data_dir.resolve()
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    cand = (data_dir / "component_materials.json").resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        f"No --component-materials provided and default file missing: {cand}. "
        "Pass --component-materials path/to.json (see data_library/component_materials.json)."
    )


def _build_orchestration(
    *,
    data_dir: Path,
    blade_yaml: Path,
    system_type_key: str,
    component_materials_path: Path | None,
) -> PrecomputeOrchestrationContext:
    mat_path = _resolve_component_materials_path(data_dir, component_materials_path)
    cmap = load_component_materials_json(mat_path)
    validate_component_indices(blade_yaml.resolve(), cmap)
    layout = resolve_system_type(system_type_key)
    return PrecomputeOrchestrationContext(
        system_type_key=str(system_type_key).strip(),
        layout=layout,
        component_materials=cmap,
    )


# ---------------------------------------------------------------------------
# Stage 1: section_geometry
# ---------------------------------------------------------------------------


def _section_geometry_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionGeometryOutputs:
    out_stage = (out_dir / "section_geometry").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    # section_geometry plotting imports matplotlib at module import time; keep it isolated.
    #
    # NOTE: section_geometry currently uses some absolute imports like `from geometry.csg import ...`
    # (rather than package-relative imports). Ensure that `blade_precompute/section_geometry` is
    # on sys.path so those resolve when invoked from repo root.
    sg_sys_path = (Path(__file__).resolve().parent / "blade_precompute" / "section_geometry").resolve()
    added_sg_path = False
    if str(sg_sys_path) not in sys.path:
        sys.path.insert(0, str(sg_sys_path))
        added_sg_path = True
    try:
        from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
        from blade_precompute.section_geometry.geometry.grid import SDFGrid
        from blade_precompute.section_geometry.interface.export import SectionPropertiesReport

        try:
            from blade_precompute.section_geometry.interface.plot import plot_section
        except Exception:  # pragma: no cover
            plot_section = None
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import section_geometry. Ensure dependencies are installed "
            "(matplotlib is required for plots)."
        ) from e
    finally:
        if added_sg_path:
            try:
                sys.path.remove(str(sg_sys_path))
            except ValueError:
                pass

    idx = _station_indices(int(inp.span_r_z_m.shape[0]), plot_station_spec)
    png_paths: list[Path] = []
    json_paths: list[Path] = []
    rz_used: list[float] = []

    for i in idx:
        rz = float(inp.span_r_z_m[i])
        chord = float(inp.chord_m[i])
        code = _naca4(inp.naca_m[i], inp.naca_p[i], inp.naca_xx[i])
        airfoil = AirfoilSDF.from_naca(code, chord=chord)
        twist_rad = float(np.deg2rad(inp.twist_deg[i]))
        section = build_section_view(airfoil, orchestration.layout, twist_angle_rad=twist_rad)
        grid = SDFGrid.from_airfoil(airfoil, nx=512, ny=220)

        tag = f"i{i:03d}_rz{rz:.3f}"
        props_json = (out_stage / f"section_props_{tag}.json").resolve()
        labels = list(getattr(section, "labels", list(section)))
        job_meta = {
            **orchestration.job_meta(),
            "midline_contract_version": MIDLINE_CONTRACT_VERSION,
            "midline_contract_summary": (midline_series_contract_doc() or "").strip().split("\n\n", 1)[0],
            "section_boundary_export": section_boundary_stub_from_labels(
                labels,
                n_cells_hint=orchestration.layout.n_cells,
            ).to_jsonable(),
        }
        first_label = next(iter(section))
        phi0 = grid.eval(section[first_label])
        assert_grid_phi_finite(phi0)
        SectionPropertiesReport(section, grid).to_json(props_json, job_meta=job_meta)
        json_paths.append(props_json)

        if plot_section is not None:
            fig, _ = plot_section(section, grid, title=f"section_geometry: NACA{code}, chord={chord:.3g} @ r_z={rz:.3g} m")
            png = (out_stage / f"section_{tag}.png").resolve()
            fig.savefig(png, dpi=170, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
            png_paths.append(png)

        rz_used.append(rz)

    _write_json(
        out_stage / "summary.json",
        {
            "stations": [{"i": int(i), "r_z_m": float(inp.span_r_z_m[i])} for i in idx],
            "png_paths": png_paths,
            "json_paths": json_paths,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )

    return SectionGeometryOutputs(
        station_indices=idx,
        station_r_z_m=rz_used,
        png_paths=png_paths,
        json_paths=json_paths,
    )


# ---------------------------------------------------------------------------
# Stage 2: section_properties
# ---------------------------------------------------------------------------


def _default_dv0(n_station: int):
    from blade_precompute.section_optimisation.core.types import DesignVector

    n = int(n_station)
    return DesignVector(
        t_skin=np.full(n, 0.012, dtype=np.float64),
        t_cap=np.full(n, 0.050, dtype=np.float64),
        t_web=np.full(n, 0.015, dtype=np.float64),
    )


def _plot_section_properties_station(section_def, res, out_png: Path) -> None:
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


def _section_properties_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionPropertiesOutputs:
    out_stage = (out_dir / "section_properties").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder
    from blade_precompute.section_properties.api import AnalysisConfig, SectionAnalysis

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    dv0 = _default_dv0(int(bg.z_stations.shape[0]))
    section_defs = SectionBuilder.build(dv0, bg)

    extreme_raw = BladeDesignProblem.load_extreme_loads_dat(inp.extreme_loads_path, z_geometry=None)
    z_raw = np.asarray(extreme_raw.z_stations, dtype=np.float64).ravel()
    if z_raw.size < 2:
        raise ValueError(
            "section_properties local panel buckling requires at least two z stations in extreme loads."
        )

    z = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)

    def _interp_extreme(col: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(z, z_raw, np.asarray(col, dtype=np.float64).ravel())

    N = _interp_extreme(extreme_raw.N)
    Vy = _interp_extreme(extreme_raw.Vy)
    Vz = _interp_extreme(extreme_raw.Vz)
    My = _interp_extreme(extreme_raw.My)
    Mz = _interp_extreme(extreme_raw.Mz)
    T = _interp_extreme(extreme_raw.T)

    chord = np.asarray(inp.chord_m, dtype=np.float64).ravel()
    analysis = SectionAnalysis(config=AnalysisConfig(run_panel_buckling=True, merge_tolerance=1e-6))
    panel_a_m: list[float] = []
    results: list[object] = []
    for i, sd in enumerate(section_defs):
        chord_m_i = float(chord[min(i, chord.shape[0] - 1)]) if chord.size > 0 else 1.0
        a_i, _ = _buckling_member_length_m(
            "dz_station",
            station_index=i,
            chord_m=chord_m_i,
            z_stations=z,
            override_m=None,
        )
        panel_a_m.append(float(a_i))
        f6 = np.array([N[i], My[i], Mz[i], T[i], Vy[i], Vz[i]], dtype=np.float64)
        results.append(
            analysis.solve(
                sd,
                panel_frame_spacing_m=float(a_i),
                panel_reference_forces_6=f6,
            )
        )

    n = len(results)
    K6 = np.stack([np.asarray(r.K6, dtype=np.float64) for r in results], axis=0).reshape(n, 6, 6)
    K7 = np.stack([np.asarray(r.K7, dtype=np.float64) for r in results], axis=0).reshape(n, 7, 7)

    summary_rows = []
    for i, (sd, r) in enumerate(zip(section_defs, results)):
        row: dict[str, Any] = {
            "station_z": float(sd.station_z),
            "area": float(r.area),
            "mass_per_length": float(r.mass_per_length),
            "elastic_center": np.asarray(r.elastic_center, dtype=np.float64),
            "mass_center": np.asarray(r.mass_center, dtype=np.float64),
            "shear_center": np.asarray(r.shear_center, dtype=np.float64),
            "panel_local_a_m": float(panel_a_m[i]),
        }
        pb = getattr(r, "panel_buckling", None)
        if pb is not None:
            row["panel_local_BI_max"] = float(pb.BI_max)
            row["panel_local_n_buckled"] = int(pb.n_buckled)
            row["panel_local_critical_edge"] = int(pb.critical_edge)
        summary_rows.append(row)

    panel_stations: list[dict[str, Any]] = []
    for i, sd in enumerate(section_defs):
        r = results[i]
        pb = getattr(r, "panel_buckling", None)
        f6 = np.array([N[i], My[i], Mz[i], T[i], Vy[i], Vz[i]], dtype=np.float64)
        edge_payload: list[dict[str, Any]] | None = None
        if pb is not None:
            edge_payload = [dataclasses.asdict(er) for er in pb.edge_results]
        panel_stations.append(
            {
                "station_z": float(sd.station_z),
                "panel_a_m": float(panel_a_m[i]),
                "reference_forces_6": f6.tolist(),
                "panel_local": None
                if pb is None
                else {
                    "BI_max": float(pb.BI_max),
                    "critical_edge": int(pb.critical_edge),
                    "n_buckled": int(pb.n_buckled),
                    "edges": edge_payload,
                },
            }
        )

    panel_json_path = (out_stage / "panel_local_buckling.json").resolve()
    _write_json(
        panel_json_path,
        {
            "kind": "orthogonal_skin_panel_local",
            "distinction": "Not GBT section_beam_model member buckling.",
            "blade_yaml": str(blade_yaml.resolve()),
            "stations": panel_stations,
        },
    )

    summary_json = _write_json(
        out_stage / "section_solve_summary.json",
        {
            "blade_yaml": blade_yaml.resolve(),
            "n_station": int(n),
            "stations": summary_rows,
            "K6": K6,
            "K7": K7,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )

    png_paths: list[Path] = []
    for i in _station_indices(int(z.shape[0]), plot_station_spec):
        out_png = (out_stage / f"section_station_i{i:03d}_z{float(z[i]):.3f}.png").resolve()
        _plot_section_properties_station(section_defs[i], results[i], out_png)
        png_paths.append(out_png)

    _write_json(out_stage / "summary.json", {"results_summary_json": summary_json, "png_paths": png_paths})

    return SectionPropertiesOutputs(
        station_z=z,
        K6=K6,
        K7=K7,
        results_summary_json=summary_json,
        png_paths=png_paths,
        section_results=tuple(results),
        section_definitions=tuple(section_defs),
        panel_local_buckling_json=panel_json_path,
    )


# ---------------------------------------------------------------------------
# Stage 3: beam_model
# ---------------------------------------------------------------------------


def _beam_model_impl(
    inp: PrecomputeInputs,
    sec: SectionPropertiesOutputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    n_beam_nodes: int,
    orchestration: PrecomputeOrchestrationContext,
    save_section_recovery_cache_npz: bool = False,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
) -> BeamModelOutputs:
    out_stage = (out_dir / "beam_model").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    from blade_precompute.global_beam_model.api import BeamAnalysis
    from blade_precompute.global_beam_model.core.types import BeamLoads, BoundaryCondition, SolverOptions
    from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
    from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
    from blade_precompute.section_optimisation.api import BladeDesignProblem

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        tau0=np.asarray(bg.tau0, dtype=np.float64),
        chord=np.asarray(bg.chord, dtype=np.float64),
        twist=np.asarray(bg.twist, dtype=np.float64),
        airfoil_profiles=list(bg.airfoil_profiles),
        web_positions=np.asarray(bg.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg.subcomponent_materials),
        chi0=None,
    )

    stations = stations_from_arrays(np.asarray(sec.station_z, dtype=np.float64), sec.K6, sec.K7)
    analysis = BeamAnalysis.from_blade_geometry(geom, int(n_beam_nodes), stations, span_axis=2)

    model = analysis.model
    n_nodes = int(model.n_nodes)
    n_elem = int(len(model.elements))
    z_mid = np.asarray([el.z_mid for el in model.elements], dtype=np.float64)

    # distributed loads from data_library are tabulated on r_z; interpolate onto element midpoints
    qy = np.interp(z_mid, inp.loads_r_z_m, inp.q_y_Npm)
    qz = np.interp(z_mid, inp.loads_r_z_m, inp.q_z_Npm)
    mx = np.interp(z_mid, inp.loads_r_z_m, inp.m_x_Nmpm)

    distributed_q = np.zeros((n_elem, 3), dtype=np.float64)
    distributed_q[:, 1] = qy
    distributed_q[:, 2] = qz

    loads = BeamLoads(
        nodal_F=np.zeros((n_nodes, 3), dtype=np.float64),
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        distributed_q=distributed_q,
        distributed_mz=np.asarray(mx, dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )
    opts = SolverOptions(
        max_iter=70,
        tol_res=5e-2,
        tol_res_rel=5e-3,
        tol_du=1e-7,
        n_gauss=2,
        n_load_steps=18,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res = analysis.solve_static(loads, options=opts)

    try:
        from blade_precompute.global_beam_model.engine.section_recovery import enrich_beam_result_with_section_stress

        res = enrich_beam_result_with_section_stress(
            res,
            station_z=np.asarray(sec.station_z, dtype=np.float64),
            section_results=sec.section_results,
            section_definitions=sec.section_definitions,
        )
    except Exception as exc:
        warnings.warn(f"Section stress/strain recovery skipped: {exc}", stacklevel=1)

    if save_section_recovery_cache_npz:
        try:
            from blade_precompute.global_beam_model.engine.section_recovery import (
                save_section_recovery_cache_to_npz,
            )

            save_section_recovery_cache_to_npz(
                res,
                station_z=np.asarray(sec.station_z, dtype=np.float64),
                section_results=sec.section_results,
                section_definitions=sec.section_definitions,
                path=out_stage / "section_recovery_cache.npz",
            )
        except Exception as exc:
            warnings.warn(f"Section recovery cache NPZ not written: {exc}", stacklevel=1)

    tip_disp = np.asarray(res.nodal_positions[-1] - model.X_ref[-1], dtype=np.float64)
    result_json = _write_json(
        out_stage / "beam_result.json",
        {
            "converged": bool(res.converged),
            "n_iterations": int(res.n_iterations),
            "residual_norm": float(res.residual_norm),
            "tip_displacement_m": tip_disp,
            "z_stations_out": np.asarray(res.z_stations_out, dtype=np.float64) if res.z_stations_out is not None else None,
            "resultants": np.asarray(res.resultants, dtype=np.float64),
            "strains": np.asarray(res.strains, dtype=np.float64),
            "nodal_averages": {
                "z_nodal_out": np.asarray(res.z_nodal_out, dtype=np.float64)
                if res.z_nodal_out is not None
                else None,
                "resultants_nodal": np.asarray(res.resultants_nodal, dtype=np.float64)
                if res.resultants_nodal is not None
                else None,
                "strains_nodal": np.asarray(res.strains_nodal, dtype=np.float64)
                if res.strains_nodal is not None
                else None,
            },
            "section_recovery": {
                "z_section_m": np.asarray(res.z_section_recovery, dtype=np.float64)
                if res.z_section_recovery is not None
                else None,
                "ply_stress_voigt_max_gp_Pa": np.asarray(res.section_stress_voigt_gp, dtype=np.float64)
                if res.section_stress_voigt_gp is not None
                else None,
                "ply_stress_voigt_max_nodal_Pa": np.asarray(res.section_stress_voigt_nodal, dtype=np.float64)
                if res.section_stress_voigt_nodal is not None
                else None,
                "laminate_strain_maxabs_gp": np.asarray(res.section_strain_maxabs_gp, dtype=np.float64)
                if res.section_strain_maxabs_gp is not None
                else None,
                "laminate_strain_maxabs_nodal": np.asarray(res.section_strain_maxabs_nodal, dtype=np.float64)
                if res.section_strain_maxabs_nodal is not None
                else None,
                "tsai_wu_fi_max_gp": np.asarray(res.section_tsai_wu_fi_max_gp, dtype=np.float64)
                if res.section_tsai_wu_fi_max_gp is not None
                else None,
                "tsai_wu_fi_max_nodal": np.asarray(res.section_tsai_wu_fi_max_nodal, dtype=np.float64)
                if res.section_tsai_wu_fi_max_nodal is not None
                else None,
                "von_mises_fi_max_gp": np.asarray(res.section_von_mises_fi_max_gp, dtype=np.float64)
                if res.section_von_mises_fi_max_gp is not None
                else None,
                "von_mises_fi_max_nodal": np.asarray(res.section_von_mises_fi_max_nodal, dtype=np.float64)
                if res.section_von_mises_fi_max_nodal is not None
                else None,
                "delamination_fi_max_gp": np.asarray(res.section_delamination_fi_max_gp, dtype=np.float64)
                if res.section_delamination_fi_max_gp is not None
                else None,
                "delamination_fi_max_nodal": np.asarray(res.section_delamination_fi_max_nodal, dtype=np.float64)
                if res.section_delamination_fi_max_nodal is not None
                else None,
                "ply_stress_secframe_voigt_max_gp_Pa": np.asarray(
                    res.section_stress_voigt_secframe_gp, dtype=np.float64
                )
                if res.section_stress_voigt_secframe_gp is not None
                else None,
                "ply_stress_secframe_voigt_max_nodal_Pa": np.asarray(
                    res.section_stress_voigt_secframe_nodal, dtype=np.float64
                )
                if res.section_stress_voigt_secframe_nodal is not None
                else None,
                "d_tsai_wu_fi_dz_gp": np.asarray(res.section_d_tsai_wu_fi_dz_gp, dtype=np.float64)
                if res.section_d_tsai_wu_fi_dz_gp is not None
                else None,
                "d_tsai_wu_fi_dz_nodal": np.asarray(res.section_d_tsai_wu_fi_dz_nodal, dtype=np.float64)
                if res.section_d_tsai_wu_fi_dz_nodal is not None
                else None,
                "tsai_wu_fi_ply_envelope_gp": np.asarray(res.section_tsai_wu_fi_ply_envelope_gp, dtype=np.float64)
                if res.section_tsai_wu_fi_ply_envelope_gp is not None
                else None,
                "tsai_wu_fi_ply_envelope_nodal": np.asarray(
                    res.section_tsai_wu_fi_ply_envelope_nodal, dtype=np.float64
                )
                if res.section_tsai_wu_fi_ply_envelope_nodal is not None
                else None,
            },
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )

    png_paths: list[Path] = []
    try:
        from blade_precompute.global_beam_model.interface import plot as bmplot

        import matplotlib.pyplot as plt

        figs = []
        fig, _ = bmplot.plot_centerline_ref_def(model, res)
        figs.append(("beam_centerline.png", fig))
        fig, _ = bmplot.plot_spanwise_resultants(res)
        figs.append(("beam_resultants.png", fig))
        fig, _ = bmplot.plot_spanwise_strains(res)
        figs.append(("beam_strains.png", fig))
        fig, _ = bmplot.plot_spanwise_resultants_nodal(res)
        figs.append(("beam_resultants_nodal.png", fig))
        fig, _ = bmplot.plot_spanwise_strains_nodal(res)
        figs.append(("beam_strains_nodal.png", fig))
        for name, plot_fn in (
            ("beam_section_stress.png", bmplot.plot_spanwise_section_stress),
            ("beam_section_strain_laminate.png", bmplot.plot_spanwise_section_strain_laminate),
            ("beam_section_tsai_wu.png", bmplot.plot_spanwise_section_tsai_wu),
            ("beam_section_von_mises_fi.png", bmplot.plot_spanwise_section_von_mises_fi),
            ("beam_section_delamination_fi.png", bmplot.plot_spanwise_section_delamination_fi),
            ("beam_section_stress_secframe.png", bmplot.plot_spanwise_section_stress_secframe),
            ("beam_section_d_tsai_wu_dz.png", bmplot.plot_spanwise_section_d_tsai_wu_dz),
            (
                "beam_section_tsai_wu_fi_heatmap_gp.png",
                lambda r: bmplot.plot_spanwise_section_tsai_wu_fi_heatmap(r, source="gp"),
            ),
            (
                "beam_section_tsai_wu_fi_heatmap_nodal.png",
                lambda r: bmplot.plot_spanwise_section_tsai_wu_fi_heatmap(r, source="nodal"),
            ),
        ):
            try:
                fig, _ = plot_fn(res)
                figs.append((name, fig))
            except ValueError:
                pass
        fig, _ = bmplot.plot_nodal_warping(model, res)
        figs.append(("beam_warping.png", fig))
        fig, _ = bmplot.plot_iteration_history(res)
        figs.append(("beam_iteration_history.png", fig))
        fig, _ = bmplot.plot_reactions(res)
        figs.append(("beam_reactions.png", fig))
        fig, _ = bmplot.plot_distributed_loads(model, loads)
        figs.append(("beam_distributed_loads.png", fig))

        for name, fig in figs:
            p = (out_stage / name).resolve()
            fig.savefig(p, dpi=170, bbox_inches="tight")
            plt.close(fig)
            png_paths.append(p)
    except ImportError:
        pass

    _write_json(out_stage / "summary.json", {"result_json": result_json, "png_paths": png_paths})
    return BeamModelOutputs(result_json=result_json, png_paths=png_paths)


# ---------------------------------------------------------------------------
# Stage 4: section_optimisation
# ---------------------------------------------------------------------------


def _design_eval_payload(ev: Any, dv: Any) -> dict[str, Any]:  # DesignEvaluation, DesignVector
    """Single evaluation block for JSON (ev: DesignEvaluation, dv: DesignVector)."""
    return {
        "mass_kg": float(ev.mass),
        "stiffness_metric_int_trace_k7": float(ev.stiffness_metric),
        "stiffness_metric_over_mass": float(ev.stiffness_metric / max(ev.mass, 1e-300)),
        "max_fi_tw": float(ev.max_fi_tw),
        "max_fi_vm": float(ev.max_fi_vm),
        "max_fi_delam": float(ev.max_fi_delam) if ev.max_fi_delam is not None else None,
        "dv": dv,
    }


def _section_optimisation_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    orchestration: PrecomputeOrchestrationContext,
    run_blade_optimizer: bool = False,
    optimization_objective: OptimizationObjective = "min_mass",
    optimizer_max_iter: int = 120,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionOptimisationOutputs:
    out_stage = (out_dir / "section_optimisation").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_optimisation import BladeOptimizer
    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.core.types import DesignProblem, ExtremeLoads

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    # data_library extreme load tables are often written on a denser span grid than the YAML
    # geometry. Load them without strict z validation, then resample integrated resultants
    # onto the geometry stations.
    extreme_raw = BladeDesignProblem.load_extreme_loads_dat(inp.extreme_loads_path, z_geometry=None)

    z_geom = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_raw = np.asarray(extreme_raw.z_stations, dtype=np.float64).ravel()
    if z_raw.size < 2:
        raise ValueError("ExtremeLoads must contain at least two z stations.")

    def _interp(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return np.interp(z_geom, z_raw, a)

    extreme = ExtremeLoads(
        z_stations=z_geom,
        N=_interp(extreme_raw.N),
        Vy=_interp(extreme_raw.Vy),
        Vz=_interp(extreme_raw.Vz),
        My=_interp(extreme_raw.My),
        Mz=_interp(extreme_raw.Mz),
        T=_interp(extreme_raw.T),
        B=_interp(extreme_raw.bimoment()),
    )
    problem = DesignProblem(
        blade_geometry=bg,
        extreme_loads=extreme,
        solver=None,
        objective=optimization_objective,
        ks_rho=35.0,
        enable_tier3_delam=False,
        n_workers=1,
    )
    sizing = BladeDesignProblem(problem)

    dv0 = _default_dv0(int(bg.z_stations.shape[0]))
    ev0 = sizing.evaluate(dv0)

    eval_payload: dict[str, Any] = {
        "optimization_objective": optimization_objective,
        "blade_optimizer_ran": bool(run_blade_optimizer),
        "grid": dict(grid_meta) if grid_meta is not None else None,
        **_design_eval_payload(ev0, dv0),
    }

    opt_res = None
    if run_blade_optimizer:
        opt = BladeOptimizer(
            sizing.problem,
            options={"maxiter": int(optimizer_max_iter), "ftol": 1e-5, "disp": False},
        )
        opt_res = opt.run(dv0)
        ev_opt = opt_res.evaluations[-1] if opt_res.evaluations else sizing.evaluate(opt_res.dv_opt)
        eval_payload["optimised"] = _design_eval_payload(ev_opt, opt_res.dv_opt)
        eval_payload["blade_optimizer"] = {
            "success": bool(opt_res.success),
            "message": str(opt_res.message),
            "n_iter": int(opt_res.n_iter),
            "max_iter": int(optimizer_max_iter),
        }

    result_json = _write_json(out_stage / "design_eval.json", eval_payload)

    png_paths: list[Path] = []
    try:
        from blade_precompute.section_optimisation.interface import plot as dplot

        import matplotlib.pyplot as plt

        z = np.asarray(bg.z_stations, dtype=np.float64)
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
    except ImportError:
        pass

    _write_json(out_stage / "summary.json", {"result_json": result_json, "png_paths": png_paths})
    return SectionOptimisationOutputs(result_json=result_json, png_paths=png_paths)


# ---------------------------------------------------------------------------
# Stage 5: section_buckling (GBT cross-section + member buckling)
# ---------------------------------------------------------------------------


def _buckling_member_length_m(
    mode: str,
    *,
    station_index: int,
    chord_m: float,
    z_stations: NDArray[np.float64],
    override_m: float | None,
) -> tuple[float, str]:
    """Scalar member length for GBT ``MemberBucklingAnalysis``."""
    if override_m is not None and float(override_m) > 0.0:
        return float(override_m), "cli_override"
    m = (mode or "chord").strip().lower()
    if m == "chord":
        return float(max(chord_m, 1e-4)), "chord"
    if m == "dz_station":
        zs = np.asarray(z_stations, dtype=np.float64).ravel()
        n = int(zs.size)
        if n < 2:
            return float(max(chord_m, 1e-4)), "dz_station_fallback_chord"
        d: float | None = None
        if station_index + 1 < n:
            d = abs(float(zs[station_index + 1] - zs[station_index]))
        elif station_index > 0:
            d = abs(float(zs[station_index] - zs[station_index - 1]))
        else:
            d = abs(float(zs[-1] - zs[0])) / max(n - 1, 1)
        return float(max(d, 1e-4)), "dz_station"
    raise ValueError(f"Unknown buckling length mode {mode!r}. Use chord or dz_station.")


def _section_buckling_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    buckling_length_mode: str = "chord",
    buckling_member_length_m: float | None = None,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionBucklingOutputs:
    out_stage = (out_dir / "section_buckling").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder
    from blade_precompute.section_beam_model.gbt import SectionLoads
    from blade_precompute.section_buckling.interface.plots import plot_buckling_member_overview_grid
    from blade_precompute.section_buckling.interface.precompute import (
        analyze_station_buckling,
        safe_subcomponent_filename_label,
    )

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    dv0 = _default_dv0(int(bg.z_stations.shape[0]))
    section_defs = SectionBuilder.build(dv0, bg)

    extreme_raw = BladeDesignProblem.load_extreme_loads_dat(inp.extreme_loads_path, z_geometry=None)
    z_geom = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_raw = np.asarray(extreme_raw.z_stations, dtype=np.float64).ravel()
    if z_raw.size < 2:
        raise ValueError("ExtremeLoads must contain at least two z stations for section_buckling.")

    def _interp(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return np.interp(z_geom, z_raw, a)

    N = _interp(extreme_raw.N)
    Vy = _interp(extreme_raw.Vy)
    Vz = _interp(extreme_raw.Vz)
    My = _interp(extreme_raw.My)
    Mz = _interp(extreme_raw.Mz)
    T = _interp(extreme_raw.T)

    z = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)
    idx = _station_indices(int(z.shape[0]), plot_station_spec)

    station_json_paths: list[Path] = []
    part_json_paths: list[Path] = []
    png_paths: list[Path] = []
    station_dir_paths: list[Path] = []

    z_stations = np.asarray(bg.z_stations, dtype=np.float64).ravel()

    for i in idx:
        sd = section_defs[i]
        rz = float(z[i])
        chord_m = float(max(np.asarray(bg.chord, dtype=np.float64).ravel()[i], 1e-4))
        L_m, L_src = _buckling_member_length_m(
            buckling_length_mode,
            station_index=i,
            chord_m=chord_m,
            z_stations=z_stations,
            override_m=buckling_member_length_m,
        )
        loads = SectionLoads(
            N=float(N[i]),
            My=float(My[i]),
            Mz=float(Mz[i]),
            Vy=float(Vy[i]),
            Vz=float(Vz[i]),
            T=float(T[i]),
        )
        tag = f"i{i:03d}_z{rz:.3f}"
        st_dir = (out_stage / f"station_{tag}").resolve()
        st_dir.mkdir(parents=True, exist_ok=True)
        coupled_dir = (st_dir / "coupled").resolve()
        coupled_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = (st_dir / "parts").resolve()
        parts_dir.mkdir(parents=True, exist_ok=True)
        station_dir_paths.append(st_dir)

        payload: dict[str, Any]
        try:
            payload = analyze_station_buckling(
                sd,
                section_loads=loads,
                member_length_m=L_m,
                n_cross_section_modes=8,
                n_member_modes=6,
                n_elem=16,
                signature_n_pts=18,
                include_per_subcomponent=True,
                section_modes_wireframe_png=(coupled_dir / "section_modes_wireframe.png").resolve(),
                member_coupled_section_wireframe_png=(
                    coupled_dir / "member_coupled_approx.png"
                ).resolve(),
                part_modes_wireframe_out_dir=parts_dir,
                part_modes_wireframe_tag=tag,
            )
            payload["member_length_policy"] = L_src
            payload["buckling_length_mode_requested"] = buckling_length_mode
        except Exception as e:  # pragma: no cover
            payload = {
                "station_z_m": float(sd.station_z),
                "error": f"{type(e).__name__}: {e}",
            }

        wf_list = payload.pop("_wireframe_png_paths", None)
        if isinstance(wf_list, list):
            for p in wf_list:
                png_paths.append(Path(p))

        parts_index: list[dict[str, Any]] = []
        try:
            for part in payload.get("per_subcomponent") or []:
                safe = safe_subcomponent_filename_label(str(part.get("name", "part")))
                part_dir = (parts_dir / safe).resolve()
                part_dir.mkdir(parents=True, exist_ok=True)
                pj = _write_json(part_dir / "part.json", part)
                part_json_paths.append(pj)
                overview_p = (part_dir / "member_buckling_overview.png").resolve()
                parts_index.append(
                    {
                        "subcomponent_index": part.get("subcomponent_index"),
                        "name": part.get("name"),
                        "role": part.get("role"),
                        "directory": str(part_dir),
                        "json": str(pj.resolve()),
                        "member_overview_png": str(overview_p),
                        "section_modes_wireframe_png": str((part_dir / "section_modes_wireframe.png").resolve()),
                    }
                )
                an = part.get("analysis")
                if isinstance(an, dict) and an.get("error") is None and an.get("member_buckling"):
                    pscope = {"station_z_m": float(payload.get("station_z_m", sd.station_z)), **an}
                    plot_buckling_member_overview_grid(
                        pscope,
                        overview_p,
                        suptitle=f"{part.get('name', 'part')}: member buckling @ z={float(payload.get('station_z_m', sd.station_z)):.3g} m",
                    )
                    png_paths.append(overview_p)
            if parts_index:
                payload["parts_artifact_index"] = parts_index
            if payload.get("error") is None and payload.get("member_buckling"):
                coupled_overview = (coupled_dir / "member_buckling_overview.png").resolve()
                plot_buckling_member_overview_grid(
                    payload,
                    coupled_overview,
                    suptitle=f"Coupled section: member buckling @ z={float(payload.get('station_z_m', sd.station_z)):.3g} m",
                )
                png_paths.append(coupled_overview)
        except ImportError:
            pass

        payload["output_station_directory"] = str(st_dir)
        payload["coupled_directory"] = str(coupled_dir)
        payload["parts_directory"] = str(parts_dir)
        jpath = _write_json(st_dir / "station.json", payload)
        station_json_paths.append(jpath)

    summary_json = _write_json(
        out_stage / "summary.json",
        {
            "station_json_paths": station_json_paths,
            "station_directories": [str(p.resolve()) for p in station_dir_paths],
            "part_json_paths": part_json_paths,
            "png_paths": png_paths,
            "stations": [{"i": int(i), "station_z_m": float(z[i])} for i in idx],
            "buckling_length_mode": buckling_length_mode,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
            "layout": "Per station: section_buckling/station_{tag}/station.json, coupled/*.png, parts/{sub}/part.json",
        },
    )

    return SectionBucklingOutputs(
        station_json_paths=station_json_paths,
        part_json_paths=part_json_paths,
        png_paths=png_paths,
        summary_json=summary_json,
    )


class _StageBase:
    def __init__(self) -> None:
        self._results: Any | None = None

    def get_results(self) -> Any:
        if self._results is None:
            raise RuntimeError(f"{self.__class__.__name__}.execute() must be called before get_results().")
        return self._results


class SectionGeometryStage(_StageBase):
    def __init__(self, params: SectionGeometryParams) -> None:
        super().__init__()
        self.params = params

    def execute(self) -> None:
        self._results = _section_geometry_impl(
            self.params.inp,
            self.params.out_dir,
            plot_station_spec=self.params.plot_station_spec,
            orchestration=self.params.orchestration,
            grid_meta=self.params.grid_meta,
        )

    def get_results(self) -> SectionGeometryOutputs:
        return super().get_results()


class SectionPropertiesStage(_StageBase):
    def __init__(self, params: SectionPropertiesParams) -> None:
        super().__init__()
        self.params = params

    def execute(self) -> None:
        self._results = _section_properties_impl(
            self.params.inp,
            self.params.out_dir,
            blade_yaml=self.params.blade_yaml,
            plot_station_spec=self.params.plot_station_spec,
            orchestration=self.params.orchestration,
            bg_override=self.params.bg_override,
            grid_meta=self.params.grid_meta,
        )

    def get_results(self) -> SectionPropertiesOutputs:
        return super().get_results()


class BeamModelStage(_StageBase):
    def __init__(self, params: BeamModelParams) -> None:
        super().__init__()
        self.params = params

    def execute(self) -> None:
        self._results = _beam_model_impl(
            self.params.inp,
            self.params.sec,
            self.params.out_dir,
            blade_yaml=self.params.blade_yaml,
            n_beam_nodes=self.params.n_beam_nodes,
            orchestration=self.params.orchestration,
            save_section_recovery_cache_npz=self.params.save_section_recovery_cache_npz,
            bg_override=self.params.bg_override,
            grid_meta=self.params.grid_meta,
        )

    def get_results(self) -> BeamModelOutputs:
        return super().get_results()


class SectionOptimisationStage(_StageBase):
    def __init__(self, params: SectionOptimisationParams) -> None:
        super().__init__()
        self.params = params

    def execute(self) -> None:
        self._results = _section_optimisation_impl(
            self.params.inp,
            self.params.out_dir,
            blade_yaml=self.params.blade_yaml,
            orchestration=self.params.orchestration,
            run_blade_optimizer=self.params.run_blade_optimizer,
            optimization_objective=self.params.optimization_objective,
            optimizer_max_iter=self.params.optimizer_max_iter,
            bg_override=self.params.bg_override,
            grid_meta=self.params.grid_meta,
        )

    def get_results(self) -> SectionOptimisationOutputs:
        return super().get_results()


class SectionBucklingStage(_StageBase):
    def __init__(self, params: SectionBucklingParams) -> None:
        super().__init__()
        self.params = params

    def execute(self) -> None:
        self._results = _section_buckling_impl(
            self.params.inp,
            self.params.out_dir,
            blade_yaml=self.params.blade_yaml,
            plot_station_spec=self.params.plot_station_spec,
            orchestration=self.params.orchestration,
            buckling_length_mode=self.params.buckling_length_mode,
            buckling_member_length_m=self.params.buckling_member_length_m,
            bg_override=self.params.bg_override,
            grid_meta=self.params.grid_meta,
        )

    def get_results(self) -> SectionBucklingOutputs:
        return super().get_results()


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------


def _job_dir(base_out: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (base_out.resolve() / ts).resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run all blade_precompute stages and emit PNG/JSON artifacts.")
    p.add_argument("--data-dir", type=Path, default=Path("data_library"), help="Folder with input .dat files.")
    p.add_argument(
        "--yaml",
        type=Path,
        default=Path("example_blade_10.yaml"),
        help="Blade geometry YAML (default: 10-station mesh; regenerate from example_blade_hires.yaml via resample_blade_yaml).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "outputs",
        help="Base output folder (default: ./outputs next to main_precompute.py, not the shell cwd).",
    )
    p.add_argument(
        "--n-beam-nodes",
        type=int,
        default=50,
        help="1D beam FE mesh: number of nodes along the span (independent of blade YAML z_stations count).",
    )
    p.add_argument(
        "--save-section-recovery-cache-npz",
        action="store_true",
        help="After beam solve, write blade_utilities fused RecoveryCache to beam_model/section_recovery_cache.npz.",
    )
    p.add_argument(
        "--plot-stations",
        type=str,
        default="root,mid,tip",
        help="Stations to plot for station-based stages (root,mid,tip,all,every-k or indices).",
    )
    p.add_argument(
        "--system-type",
        type=str,
        default="legacy",
        help="Section layout key (e.g. legacy, 0A, 0B, 1B, 2B, 3B, 2B-F). See blade_precompute.orchestration.system_layout.",
    )
    p.add_argument(
        "--component-materials",
        type=Path,
        default=None,
        help="JSON object {skin, spar_cap, shear_web} → 0-based ply_library index. "
        "Default: <data-dir>/component_materials.json if present.",
    )
    p.add_argument(
        "--buckling-length-mode",
        type=str,
        default="chord",
        choices=("chord", "dz_station"),
        help="GBT member buckling reference length: chord at station, or local spanwise Δz between blade stations.",
    )
    p.add_argument(
        "--buckling-member-length-m",
        type=float,
        default=None,
        help="If set (>0), overrides --buckling-length-mode with this member length [m].",
    )
    p.add_argument(
        "--design-optimise",
        action="store_true",
        help="Run BladeOptimizer (SLSQP) on thickness with KS failure constraints after the initial design evaluation.",
    )
    p.add_argument(
        "--design-objective",
        type=str,
        default="min-mass",
        help="When --design-optimise is set: min-mass (default) or max-specific-stiffness.",
    )
    p.add_argument(
        "--design-max-iter",
        type=int,
        default=120,
        help="SLSQP max iterations for --design-optimise.",
    )
    args = p.parse_args(argv)

    design_objective = _objective_from_cli(args.design_objective)

    inp = load_inputs(args.data_dir)
    job = _job_dir(args.out_dir)
    job.mkdir(parents=True, exist_ok=True)

    orch = _build_orchestration(
        data_dir=args.data_dir,
        blade_yaml=args.yaml,
        system_type_key=str(args.system_type),
        component_materials_path=args.component_materials,
    )

    # Independent grid config (each grid has its own linspace controls).
    z_geom_src = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    gspec = LinspaceSpec(
        z_min=float(GRID_GEOMETRY_Z_MIN) if GRID_GEOMETRY_Z_MIN is not None else float(z_geom_src[0]),
        z_max=float(GRID_GEOMETRY_Z_MAX) if GRID_GEOMETRY_Z_MAX is not None else float(z_geom_src[-1]),
        n=int(GRID_GEOMETRY_N) if GRID_GEOMETRY_N is not None else int(z_geom_src.shape[0]),
    )
    inp_geom = _resample_precompute_inputs(inp, _linspace_from_spec(gspec))

    from blade_precompute.section_optimisation.api import BladeDesignProblem

    bg_raw = BladeDesignProblem.load_geometry(args.yaml.resolve())
    z_struct_src = np.asarray(bg_raw.z_stations, dtype=np.float64).ravel()
    sspec = LinspaceSpec(
        z_min=float(GRID_STRUCTURAL_Z_MIN) if GRID_STRUCTURAL_Z_MIN is not None else float(z_struct_src[0]),
        z_max=float(GRID_STRUCTURAL_Z_MAX) if GRID_STRUCTURAL_Z_MAX is not None else float(z_struct_src[-1]),
        n=int(GRID_STRUCTURAL_N) if GRID_STRUCTURAL_N is not None else int(z_struct_src.shape[0]),
    )
    bg_struct = _resample_blade_geometry_to_z(bg_raw, _linspace_from_spec(sspec))
    grid_cfg = GridConfig(
        geometry=gspec,
        structural=sspec,
        plot_station_spec=str(args.plot_stations),
        n_beam_nodes=int(args.n_beam_nodes),
    )

    _write_json(
        job / "inputs.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "python": sys.version,
            "spanwise_path": inp.spanwise_path,
            "extreme_loads_path": inp.extreme_loads_path,
            "blade_yaml": args.yaml.resolve(),
            "system_type": orch.system_type_key,
            "component_materials": orch.component_materials.to_dict(),
            "component_materials_path": _resolve_component_materials_path(
                args.data_dir, args.component_materials
            ),
            "buckling_length_mode": str(args.buckling_length_mode),
            "buckling_member_length_m": args.buckling_member_length_m,
            "design_optimise": bool(args.design_optimise),
            "design_objective": design_objective,
            "design_max_iter": int(args.design_max_iter),
            "grid_config": grid_cfg,
        },
    )

    sg_stage = SectionGeometryStage(
        SectionGeometryParams(
            inp=inp_geom,
            out_dir=job,
            plot_station_spec=args.plot_stations,
            orchestration=orch,
            grid_meta={"type": "geometry", "linspace": gspec},
        )
    )
    sg_stage.execute()
    sg = sg_stage.get_results()

    sp_stage = SectionPropertiesStage(
        SectionPropertiesParams(
            inp=inp_geom,
            out_dir=job,
            blade_yaml=args.yaml.resolve(),
            plot_station_spec=args.plot_stations,
            orchestration=orch,
            bg_override=bg_struct,
            grid_meta={"type": "structural", "linspace": sspec},
        )
    )
    sp_stage.execute()
    sp = sp_stage.get_results()

    sb_stage = SectionBucklingStage(
        SectionBucklingParams(
            inp=inp_geom,
            out_dir=job,
            blade_yaml=args.yaml.resolve(),
            plot_station_spec=args.plot_stations,
            orchestration=orch,
            buckling_length_mode=str(args.buckling_length_mode),
            buckling_member_length_m=args.buckling_member_length_m,
            bg_override=bg_struct,
            grid_meta={"type": "structural", "linspace": sspec},
        )
    )
    sb_stage.execute()
    sb = sb_stage.get_results()

    bm_stage = BeamModelStage(
        BeamModelParams(
            inp=inp_geom,
            sec=sp,
            out_dir=job,
            blade_yaml=args.yaml.resolve(),
            n_beam_nodes=int(args.n_beam_nodes),
            orchestration=orch,
            save_section_recovery_cache_npz=bool(args.save_section_recovery_cache_npz),
            bg_override=bg_struct,
            grid_meta={
                "type": "beam",
                "structural_linspace": sspec,
                "n_beam_nodes": int(args.n_beam_nodes),
            },
        )
    )
    bm_stage.execute()
    bm = bm_stage.get_results()

    do_stage = SectionOptimisationStage(
        SectionOptimisationParams(
            inp=inp_geom,
            out_dir=job,
            blade_yaml=args.yaml.resolve(),
            orchestration=orch,
            run_blade_optimizer=bool(args.design_optimise),
            optimization_objective=design_objective,
            optimizer_max_iter=int(args.design_max_iter),
            bg_override=bg_struct,
            grid_meta={"type": "design", "structural_linspace": sspec},
        )
    )
    do_stage.execute()
    do = do_stage.get_results()

    _write_json(
        job / "summary.json",
        {
            "job_dir": job,
            "system_type": orch.system_type_key,
            "component_materials": orch.component_materials.to_dict(),
            "section_geometry": sg,
            "section_properties": sp,
            "section_buckling": sb,
            "beam_model": bm,
            "section_optimisation": do,
        },
    )

    print(str(job))
    return 0


if __name__ == "__main__":
    # Ensure repo root is on sys.path when invoked from elsewhere.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())

