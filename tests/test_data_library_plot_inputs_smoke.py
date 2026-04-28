"""Smoke: data_library.plot_inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_library.plot_inputs import (
    _canonicalise_unit,
    plot_blade_spanwise_dat,
    plot_extreme_load_distribution_dat,
    plot_operational_load_heatmap,
    plot_operational_timeseries_dat,
    read_columnar_dat_with_units,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_data_library_plots_run():
    root = _repo_root()
    fig, _ = plot_blade_spanwise_dat(root / "data_library" / "blade_spanwise_distribution.dat")
    plt.close(fig)
    fig, _ = plot_extreme_load_distribution_dat(root / "data_library" / "extreme_load_distribution.dat")
    plt.close(fig)
    fig, _ = plot_operational_timeseries_dat(root / "data_library" / "operational_load_timeseries.dat")
    plt.close(fig)
    fig, _ = plot_operational_load_heatmap(root / "data_library" / "operational_load_timeseries.dat")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unit-row validation for all four DAT_STYLE-conforming files
# ---------------------------------------------------------------------------

_EXPECTED_UNITS: dict[str, list[str]] = {
    "blade_spanwise_distribution.dat": [
        "m", "m", "-", "-", "m", "deg", "-", "-", "-", "-", "1/m", "1/m", "1/m"
    ],
    "extreme_load_distribution.dat": ["m", "m", "N/m", "N/m", "N*m/m"],
    "operational_load_timeseries.dat": ["s", "m", "m", "N/m", "N/m", "N*m/m"],
}


@pytest.mark.parametrize("filename,expected_units", list(_EXPECTED_UNITS.items()))
def test_dat_unit_rows_match_spec(filename: str, expected_units: list[str]) -> None:
    path = _repo_root() / "data_library" / filename
    names, units, data = read_columnar_dat_with_units(path)
    assert len(units) == len(names), (
        f"{filename}: # units row has {len(units)} entries but header has {len(names)} columns."
    )
    assert len(units) == len(expected_units), (
        f"{filename}: expected {len(expected_units)} units but file declares {len(units)}."
    )
    for col, got, want in zip(names, units, expected_units):
        assert _canonicalise_unit(got) == _canonicalise_unit(want), (
            f"{filename} column {col!r}: declared {got!r}, expected {want!r}."
        )
    assert data.shape[1] == len(names)


def test_dat_units_absent_raises(tmp_path: Path) -> None:
    """read_columnar_dat_with_units must raise ValueError when no # units: line is present."""
    f = tmp_path / "no_units.dat"
    f.write_text("# just a comment\ncol_a  col_b\n1.0  2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="units"):
        read_columnar_dat_with_units(f)


def test_canonicalise_unit_equivalences() -> None:
    """Verify the canonicaliser treats different representations as equal."""
    assert _canonicalise_unit("N*m/m") == _canonicalise_unit("N\u00b7m/m")
    assert _canonicalise_unit("kg/m^3") == _canonicalise_unit("kg/m^3")
    assert _canonicalise_unit("Pa") == _canonicalise_unit("pa")
    assert _canonicalise_unit("1/m^1") == _canonicalise_unit("1/m")
