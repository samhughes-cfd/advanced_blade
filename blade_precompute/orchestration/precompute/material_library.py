"""Load ``material_library.dat`` and apply per-subcomponent material IDs to ``OptimBladeGeometry``.

The ``.dat`` file is the human-readable, machine-validated tabulated format described in
``data_library/DAT_STYLE.md``. It contains two sections (``[orthotropic_laminate_ply]`` and
``[isotropic]``); each section declares its column units in a ``# units:`` row that is
validated by :func:`load_material_library_dat` at load time.
"""

from __future__ import annotations

import dataclasses as _dc
from pathlib import Path
from typing import Any, Literal, Mapping

from blade_precompute.section_optimisation.core.types import OptimBladeGeometry, ThicknessRole
from blade_precompute.section_properties.engine.geometry import MaterialAssignment
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly
from blade_precompute.section_properties.io.materials_loader import orthotropic_ply_from_dict

from blade_precompute.section_optimisation.engine.ply_angle_constraints import validate_stack_angles_for_role

MaterialKind = Literal["orthotropic_laminate_ply", "isotropic"]

_LOGICAL_SUBCOMPONENT_ALIASES: dict[str, str] = {
    "spar": "spar_cap",
    "web": "shear_web",
}

_REQUIRED_LOGICAL_KEYS: frozenset[str] = frozenset({"skin", "spar_cap", "shear_web"})
_OPTIONAL_LOGICAL_KEYS: frozenset[str] = frozenset({"core"})
_ALLOWED_LOGICAL_KEYS: frozenset[str] = _REQUIRED_LOGICAL_KEYS | _OPTIONAL_LOGICAL_KEYS


def normalize_logical_subcomponent_material_map(raw: Mapping[str, int]) -> dict[str, int]:
    """
    Map user keys (``spar`` / ``web`` aliases) to canonical subcomponent keys.

    Required keys: ``skin`` / ``spar_cap`` / ``shear_web``.
    Optional key: ``core``.
    """
    out: dict[str, int] = {}
    for k, v in raw.items():
        key = str(k).strip().lower()
        canon = _LOGICAL_SUBCOMPONENT_ALIASES.get(key, key)
        if canon not in _ALLOWED_LOGICAL_KEYS:
            raise KeyError(
                f"Unknown subcomponent material key {k!r}. "
                "Use skin, spar_cap (or spar), shear_web (or web), optional core; "
                f"allowed: {sorted(_ALLOWED_LOGICAL_KEYS)}."
            )
        out[canon] = int(v)
    missing = _REQUIRED_LOGICAL_KEYS - frozenset(out.keys())
    if missing:
        raise KeyError(f"subcomponent_materials missing canonical keys after alias resolution: {sorted(missing)}")
    extra = frozenset(out.keys()) - _ALLOWED_LOGICAL_KEYS
    if extra:
        raise KeyError(f"Unexpected keys: {sorted(extra)}")
    return out


def _infer_role(name: str, explicit: dict[str, ThicknessRole]) -> ThicknessRole:
    if name in explicit:
        return explicit[name]
    n = name.lower()
    if "skin" in n:
        return "skin"
    if "cap" in n:
        return "cap"
    if "web" in n:
        return "web"
    return "fixed"


@_dc.dataclass(frozen=True)
class MaterialRow:
    material_id: int
    name: str
    kind: MaterialKind

    # Orthotropic (optional if isotropic)
    E1: float | None = None
    E2: float | None = None
    G12: float | None = None
    nu12: float | None = None
    rho: float | None = None
    t_ply: float | None = None
    Xt: float | None = None
    Xc: float | None = None
    Yt: float | None = None
    Yc: float | None = None
    S12: float | None = None
    Zt: float | None = None
    S13: float | None = None
    S23: float | None = None

    # Isotropic (optional if orthotropic)
    E: float | None = None
    nu: float | None = None
    sigma_allow: float | None = None

    def as_orthotropic_dict(self) -> dict[str, Any]:
        if self.kind != "orthotropic_laminate_ply":
            raise TypeError(f"Row {self.material_id} is not orthotropic.")
        keys = (
            "E1",
            "E2",
            "G12",
            "nu12",
            "rho",
            "t_ply",
            "Xt",
            "Xc",
            "Yt",
            "Yc",
            "S12",
            "Zt",
            "S13",
            "S23",
        )
        d: dict[str, Any] = {}
        for k in keys:
            v = getattr(self, k)
            if v is None:
                raise ValueError(f"material_id {self.material_id} ({self.name}): missing {k}")
            d[k] = float(v)
        return d


_UNITS_PREFIX = "# units:"

_EXPECTED_UNITS_ORTHOTROPIC: dict[str, str] = {
    "material_id": "-",
    "name": "-",
    "E1": "Pa",
    "E2": "Pa",
    "G12": "Pa",
    "nu12": "-",
    "rho": "kg/m^3",
    "t_ply": "m",
    "Xt": "Pa",
    "Xc": "Pa",
    "Yt": "Pa",
    "Yc": "Pa",
    "S12": "Pa",
    "Zt": "Pa",
    "S13": "Pa",
    "S23": "Pa",
}

_EXPECTED_UNITS_ISOTROPIC: dict[str, str] = {
    "material_id": "-",
    "name": "-",
    "rho": "kg/m^3",
    "E": "Pa",
    "nu": "-",
    "sigma_allow": "Pa",
}


def _assert_units(
    section: str,
    columns: list[str],
    units: list[str],
    expected: Mapping[str, str],
    *,
    path: Path,
) -> None:
    """Raise ``ValueError`` listing every offending column when the file's units do not match ``expected``."""
    from data_library.plot_inputs import _canonicalise_unit  # local import to avoid cycles

    if len(columns) != len(units):
        raise ValueError(
            f"{path} [{section}]: header has {len(columns)} columns but # units row has {len(units)}."
        )
    mismatches: list[str] = []
    for col, got in zip(columns, units):
        want = expected.get(col)
        if want is None:
            mismatches.append(f"{col!r} (unexpected column)")
            continue
        if _canonicalise_unit(got) != _canonicalise_unit(want):
            mismatches.append(f"{col!r}: got {got!r}, expected {want!r}")
    missing = [c for c in expected if c not in columns]
    if missing:
        mismatches.append(f"missing required columns: {missing}")
    if mismatches:
        raise ValueError(f"{path} [{section}]: unit / column mismatch -> " + "; ".join(mismatches))


def _parse_dat_sections(
    path: Path,
) -> dict[str, tuple[list[str], list[str], list[list[str]]]]:
    """
    Return ``{section_name: (columns, units, rows)}`` for a multi-section ``.dat`` file.

    A section starts at a line ``[section_name]`` and ends at the next section marker
    or end of file. Within a section the same parsing rules as the single-section reader
    apply: comment lines starting with ``#`` are ignored except for one ``# units:`` line
    that must precede the column header. Data rows are preserved as raw whitespace-split
    string tokens so the caller can dispatch numeric / textual coercion per column.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    sections: dict[str, tuple[list[str], list[str], list[list[str]]]] = {}

    current_name: str | None = None
    current_units: list[str] | None = None
    current_columns: list[str] | None = None
    current_rows: list[list[str]] = []

    def _close_current() -> None:
        nonlocal current_name, current_units, current_columns, current_rows
        if current_name is None:
            return
        if current_columns is None:
            raise ValueError(f"{path} [{current_name}]: no header row before next section / EOF.")
        if current_units is None:
            raise ValueError(
                f"{path} [{current_name}]: no '# units:' row before header. See DAT_STYLE.md."
            )
        if current_name in sections:
            raise ValueError(f"{path}: duplicate section [{current_name}].")
        sections[current_name] = (current_columns, current_units, current_rows)
        current_name = None
        current_units = None
        current_columns = None
        current_rows = []

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]") and len(s) > 2:
            _close_current()
            current_name = s[1:-1].strip()
            current_units = None
            current_columns = None
            current_rows = []
            continue
        if s.startswith("#"):
            if current_name is None:
                continue  # global preamble; not part of any section
            if current_columns is None and s.lower().startswith(_UNITS_PREFIX):
                payload = s[len(_UNITS_PREFIX) :].strip()
                current_units = [tok.strip() for tok in payload.split(",") if tok.strip()]
            continue
        if current_name is None:
            raise ValueError(
                f"{path}: data line found outside any [section] -> {s!r}. See DAT_STYLE.md."
            )
        parts = s.split()
        if current_columns is None:
            current_columns = parts
            continue
        if len(parts) != len(current_columns):
            raise ValueError(
                f"{path} [{current_name}]: row has {len(parts)} fields, expected {len(current_columns)}: {s!r}"
            )
        current_rows.append(parts)

    _close_current()

    if not sections:
        raise ValueError(f"{path}: no [section] blocks parsed.")
    return sections


def load_material_library_dat(path: str | Path) -> dict[int, MaterialRow]:
    """
    Parse ``material_library.dat`` (see ``data_library/DAT_STYLE.md``).

    The file must contain two sections: ``[orthotropic_laminate_ply]`` and ``[isotropic]``.
    Each section's ``# units:`` row is validated against the per-column expected units
    (Pa for stiffness/strength, kg/m^3 for density, m for thickness, ``-`` for ids/ratios/names).
    Material IDs must be unique across both sections.
    """
    path = Path(path)
    sections = _parse_dat_sections(path)

    required = {"orthotropic_laminate_ply", "isotropic"}
    missing = required - set(sections)
    if missing:
        raise ValueError(f"{path}: missing required section(s) {sorted(missing)}.")
    extra = set(sections) - required
    if extra:
        raise ValueError(f"{path}: unexpected section(s) {sorted(extra)}.")

    out: dict[int, MaterialRow] = {}

    ortho_cols, ortho_units, ortho_rows = sections["orthotropic_laminate_ply"]
    _assert_units(
        "orthotropic_laminate_ply", ortho_cols, ortho_units, _EXPECTED_UNITS_ORTHOTROPIC, path=path
    )
    for fields in ortho_rows:
        d = dict(zip(ortho_cols, fields))
        mid = int(d["material_id"])
        name = str(d["name"]).strip() or f"id_{mid}"
        try:
            r = MaterialRow(
                material_id=mid,
                name=name,
                kind="orthotropic_laminate_ply",
                E1=float(d["E1"]),
                E2=float(d["E2"]),
                G12=float(d["G12"]),
                nu12=float(d["nu12"]),
                rho=float(d["rho"]),
                t_ply=float(d["t_ply"]),
                Xt=float(d["Xt"]),
                Xc=float(d["Xc"]),
                Yt=float(d["Yt"]),
                Yc=float(d["Yc"]),
                S12=float(d["S12"]),
                Zt=float(d["Zt"]),
                S13=float(d["S13"]),
                S23=float(d["S23"]),
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path} [orthotropic_laminate_ply] row id={mid}: {e}") from e
        _ = r.as_orthotropic_dict()  # validate completeness
        if mid in out:
            raise ValueError(f"{path}: duplicate material_id {mid}.")
        out[mid] = r

    iso_cols, iso_units, iso_rows = sections["isotropic"]
    _assert_units("isotropic", iso_cols, iso_units, _EXPECTED_UNITS_ISOTROPIC, path=path)
    for fields in iso_rows:
        d = dict(zip(iso_cols, fields))
        mid = int(d["material_id"])
        name = str(d["name"]).strip() or f"id_{mid}"
        try:
            r = MaterialRow(
                material_id=mid,
                name=name,
                kind="isotropic",
                rho=float(d["rho"]),
                E=float(d["E"]),
                nu=float(d["nu"]),
                sigma_allow=float(d["sigma_allow"]),
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path} [isotropic] row id={mid}: {e}") from e
        if mid in out:
            raise ValueError(f"{path}: duplicate material_id {mid}.")
        out[mid] = r

    if not out:
        raise ValueError(f"{path}: no material rows parsed.")
    return out


def validate_material_library_bindings(
    table: Mapping[int, MaterialRow],
    logical: Mapping[str, int],
    *,
    blade_subcomponent_names: frozenset[str] | None = None,
) -> None:
    """Ensure every referenced ``material_id`` exists; optional blade-subcomponent hint for diagnostics."""
    for role, mid in logical.items():
        if mid not in table:
            hint = f" (blade subcomponents: {sorted(blade_subcomponent_names)})" if blade_subcomponent_names else ""
            raise ValueError(f"subcomponent_materials[{role}]={mid} not found in material_library{hint}.")


def resolve_material_id_for_subcomponent(
    blade_subcomponent_name: str,
    bg: OptimBladeGeometry,
    logical: Mapping[str, int],
) -> int | None:
    """Return material id for this subcomponent, or ``None`` to keep existing assignment."""
    role = _infer_role(blade_subcomponent_name, dict(bg.thickness_role))
    if role == "skin":
        return int(logical["skin"])
    if role == "cap":
        return int(logical["spar_cap"])
    if role == "web":
        return int(logical["shear_web"])
    return None


def _material_assignment_for_row_angles(
    row: MaterialRow, angles_deg: list[float], role: ThicknessRole, subcomponent: str
) -> MaterialAssignment:
    """Laminate for orthotropic rows, with angles validated; isotropic rows ignore ``angles_deg``."""
    if row.kind == "isotropic":
        return _isotropic_from_row(row)
    validate_stack_angles_for_role(
        role, (float(x) for x in angles_deg), subcomponent=subcomponent
    )
    if row.kind != "orthotropic_laminate_ply":
        raise TypeError(f"Unsupported material kind {row.kind!r} for {subcomponent!r}.")
    return _laminate_from_row_and_angles(row, [float(x) for x in angles_deg])


def subcomponent_box_materials_from_csv(
    table: Mapping[int, MaterialRow],
    logical: Mapping[str, int],
    *,
    layup_skin: list[float],
    layup_cap: list[float],
    layup_web: list[float],
) -> tuple[dict[str, MaterialAssignment], dict[str, ThicknessRole]]:
    """
    Laminates for ``skin``, ``cap_ps``, ``web`` (same subcomponent names as the example blade spec
    and :func:`load_blade_geometry`).

    Subcomponent name ``cap_ps`` is assigned thickness role **cap**; it uses ``logical['spar_cap']``.
    """
    subs: dict[str, MaterialAssignment] = {}
    skin_row = table[int(logical["skin"])]
    cap_row = table[int(logical["spar_cap"])]
    web_row = table[int(logical["shear_web"])]
    subs["skin"] = _material_assignment_for_row_angles(
        skin_row, list(layup_skin), "skin", "skin"
    )
    subs["cap_ps"] = _material_assignment_for_row_angles(
        cap_row, list(layup_cap), "cap", "cap_ps"
    )
    subs["web"] = _material_assignment_for_row_angles(
        web_row, list(layup_web), "web", "web"
    )
    roles = {"skin": "skin", "cap_ps": "cap", "web": "web"}
    return subs, roles


def _laminate_from_row_and_angles(row: MaterialRow, angles_deg: list[float]) -> LaminateDefinition:
    d = row.as_orthotropic_dict()
    pname = row.name
    ply = orthotropic_ply_from_dict(pname, d)
    plies = [(ply, float(a)) for a in angles_deg]
    return LaminateDefinition(plies=plies, shear_lag_correction=True)


def _isotropic_from_row(row: MaterialRow) -> IsotropicMaterial:
    if row.kind != "isotropic" or row.E is None or row.nu is None or row.rho is None or row.sigma_allow is None:
        raise TypeError("Not a complete isotropic row.")
    return IsotropicMaterial(
        name=row.name,
        E=float(row.E),
        nu=float(row.nu),
        rho=float(row.rho),
        sigma_allow=float(row.sigma_allow),
    )


def apply_material_library_to_blade_geometry(
    bg: OptimBladeGeometry,
    table: Mapping[int, MaterialRow],
    logical: Mapping[str, int],
) -> OptimBladeGeometry:
    """
    Replace ``subcomponent_materials`` entries whose role maps into ``logical`` with CSV-driven
    assignments; preserve layup angles from the existing laminate template when swapping orthotropic
    bulk, and validate angles against role allowlists.
    """
    new_subs: dict[str, MaterialAssignment] = {}
    roles = dict(bg.thickness_role)
    for name, mat in bg.subcomponent_materials.items():
        mid = resolve_material_id_for_subcomponent(name, bg, logical)
        if mid is None:
            new_subs[name] = mat
            continue
        row = table[mid]
        role = _infer_role(name, roles)
        if row.kind == "isotropic":
            new_subs[name] = _isotropic_from_row(row)
            continue
        if isinstance(mat, LaminateDefinition):
            angles = [float(ang) for _, ang in mat.plies]
            validate_stack_angles_for_role(role, angles, subcomponent=name)
            new_subs[name] = _laminate_from_row_and_angles(row, angles)
        else:
            raise TypeError(f"Cannot replace non-laminate material for {name!r} with orthotropic row.")
    return _dc.replace(bg, subcomponent_materials=new_subs)


def material_resolution_manifest(
    *,
    material_library_path: Path | None,
    logical: Mapping[str, int] | None,
    table: Mapping[int, MaterialRow] | None,
) -> dict[str, Any]:
    """Snippet for ``inputs.json`` (resolved material names per logical role)."""
    if not material_library_path or logical is None or table is None:
        return {"enabled": False}
    resolved = {
        role: {"material_id": int(mid), "name": table[int(mid)].name, "kind": table[int(mid)].kind}
        for role, mid in logical.items()
    }
    return {
        "enabled": True,
        "material_library_path": str(Path(material_library_path).resolve()),
        "subcomponent_material_ids": {k: int(v) for k, v in logical.items()},
        "resolved": resolved,
    }
