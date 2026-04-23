"""Unit tests for the cross-section modal analysis pipeline."""

import numpy as np
import pytest

from section_beam_model.gbt import (
    CrossSection,
    CrossSectionModalAnalysis,
    DEFAULT_BEAM_EXPORT_MODE_LABELS,
    IsotropicMaterial,
    Lamina,
    LaminateMaterial,
    SectionLoads,
    WallDefinition,
    classical_export_indices,
    export_label_to_coarse_bucket,
    select_modes,
    validate_export_classification,
)
from section_beam_model.gbt.prebuckling import PreBucklingAnalysis
from section_beam_model.gbt.section_stiffness_export import (
    gbt_to_beam_stiffness,
    gbt_to_k7,
    section_stiffness_to_k6,
)

MAT = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)

def make_c_section():
    return CrossSection([
        WallDefinition([0,0],    [0,-0.1],    MAT, n_strips=4, name='web'),
        WallDefinition([0,0],    [0.05, 0],   MAT, n_strips=2, name='top'),
        WallDefinition([0,-0.1], [0.05,-0.1], MAT, n_strips=2, name='bot'),
    ])


def make_box_section(b: float = 0.06, h: float = 0.20) -> CrossSection:
    hb, hh = 0.5 * b, 0.5 * h
    return CrossSection(
        [
            WallDefinition([-hb, -hh], [hb, -hh], MAT, n_strips=8, name="bot"),
            WallDefinition([hb, -hh], [hb, hh], MAT, n_strips=4, name="right"),
            WallDefinition([hb, hh], [-hb, hh], MAT, n_strips=8, name="top"),
            WallDefinition([-hb, hh], [-hb, -hh], MAT, n_strips=4, name="left"),
        ]
    )


def make_laminate_box_section(b: float = 0.06, h: float = 0.20) -> CrossSection:
    plies = [
        Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, angle=0, t=0.25e-3),
        Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, angle=90, t=0.25e-3),
        Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, angle=90, t=0.25e-3),
        Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, angle=0, t=0.25e-3),
    ]
    lm = LaminateMaterial(plies)
    hb, hh = 0.5 * b, 0.5 * h
    return CrossSection(
        [
            WallDefinition([-hb, -hh], [hb, -hh], lm, n_strips=8, name="bot"),
            WallDefinition([hb, -hh], [hb, hh], lm, n_strips=4, name="right"),
            WallDefinition([hb, hh], [-hb, hh], lm, n_strips=8, name="top"),
            WallDefinition([-hb, hh], [-hb, -hh], lm, n_strips=4, name="left"),
        ]
    )

def test_modal_runs():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    assert len(result.eigenvalues) == 4

def test_eigenvalues_positive():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=6)
    assert np.all(result.eigenvalues > 0)

def test_modal_rigidity_positive():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    for k in range(len(result.eigenvalues)):
        assert result.modal_rigidity(k) > 0, f"D_{k} not positive"

def test_mode_shapes_shape():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    assert result.modes.shape[1] == 4

def test_classify_mode_returns_string():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    for k in range(len(result.eigenvalues)):
        cls = result.classify_mode(k)
        assert isinstance(cls, str)
        assert cls in ("rigid_body", "local", "distortional", "global")


def test_select_modes_by_label():
    sec = make_box_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=24)
    sel = select_modes(full, mode_labels=["bending_x", "bending_y"])
    assert len(sel.eigenvalues) == 2
    labs = {sel.classify_export_mode(k) for k in range(2)}
    assert labs == {"bending_x", "bending_y"}


def test_select_modes_by_count():
    sec = make_c_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=10)
    sel = select_modes(full, n_modes=4)
    assert len(sel.eigenvalues) == 4
    assert np.allclose(np.sort(sel.eigenvalues), np.sort(full.eigenvalues)[:4])


def test_gbt_to_beam_stiffness_isotropic():
    sec = make_box_section()
    loads = SectionLoads(N=-1.0)
    ref = PreBucklingAnalysis(sec, loads).section_properties()
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    assert st.EA == pytest.approx(ref["EA"], rel=1e-9, abs=1.0)
    assert st.EI_x > 0.0 and st.EI_y > 0.0 and st.GJ > 0.0
    assert st.GA_x > 0.0 and st.GA_y > 0.0


def test_eiyz_symmetric_box_small():
    sec = make_box_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    ei_ref = max(st.EI_x, st.EI_y)
    assert abs(st.EIyz) < 1e-6 * ei_ref


def test_gbt_to_k7_symmetric_box_psd():
    sec = make_box_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    K6 = section_stiffness_to_k6(st)
    K7 = gbt_to_k7(full, K6)
    assert K7.shape == (7, 7)
    assert np.allclose(K7, K7.T)
    assert K7[6, 6] > 0.0
    w = np.linalg.eigvalsh(0.5 * (K7 + K7.T))
    assert float(w.min()) >= -1e-6 * max(float(np.max(np.diag(K7))), 1.0)


def test_gbt_to_k7_c_section_torsion_warping_coupling():
    sec = make_c_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=40)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    K6 = section_stiffness_to_k6(st)
    K7 = gbt_to_k7(full, K6)
    assert abs(float(K7[3, 6])) > 0.0


def test_gbt_to_k7_full_vlasov_c_section():
    sec = make_c_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=40)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    K6 = section_stiffness_to_k6(st)
    K7 = gbt_to_k7(full, K6, full_vlasov=True)
    assert abs(float(K7[3, 6])) > 0.0
    assert K7.shape == (7, 7)
    assert np.allclose(K7, K7.T)
    w = np.linalg.eigvalsh(0.5 * (K7 + K7.T))
    assert float(w.min()) >= -1e-6 * max(float(np.max(np.diag(K7))), 1.0)


def test_gbt_to_k7_full_vlasov_symmetric_box():
    sec = make_box_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    K6 = section_stiffness_to_k6(st)
    K7 = gbt_to_k7(full, K6, full_vlasov=True)
    k00 = max(float(K7[0, 0]), 1.0)
    k11 = max(float(K7[1, 1]), 1.0)
    assert abs(float(K7[0, 6])) < 1e-6 * k00
    assert abs(float(K7[1, 6])) < 1e-6 * k11


def test_classify_mode_delegates_to_export():
    sec = make_c_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=28)
    for k in range(len(full.eigenvalues)):
        lab = full.classify_export_mode(k)
        cls = full.classify_mode(k)
        assert cls == export_label_to_coarse_bucket(lab)
        assert cls in ("rigid_body", "local", "distortional", "global")


def test_validate_export_classification_box():
    sec = make_box_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=28)
    d = validate_export_classification(full, sec)
    assert d["distinct_indices"] and d["axial_membrane"] and d["torsion_metric"]


def test_validate_export_classification_c_section():
    sec = make_c_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=40)
    d = validate_export_classification(full, sec)
    assert sum(1 for v in d.values() if not v) <= 1


def test_classical_export_indices_distinct():
    for maker in (make_box_section, make_c_section, make_laminate_box_section):
        sec = maker()
        full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=28)
        picks = classical_export_indices(full, sec)
        vals = list(picks.values())
        assert len(vals) == len(set(vals))


def test_bending_mode_disambiguation_warns():
    """With only three modes, bending_y may fall back to bending_x; expect a user warning."""
    sec = make_box_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=3)
    with pytest.warns(UserWarning, match="same mode index"):
        classical_export_indices(full, sec)
