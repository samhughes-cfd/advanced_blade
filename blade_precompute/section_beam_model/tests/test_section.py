"""Unit tests for section.py shared-node assembly."""

import numpy as np
import pytest

from blade_precompute.section_beam_model.gbt import CrossSection, IsotropicMaterial, WallDefinition

MAT = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)

def c_section():
    web   = WallDefinition([0,0], [0,-0.1], MAT, n_strips=4, name='web')
    f_top = WallDefinition([0,0], [0.05,0], MAT, n_strips=2, name='top')
    f_bot = WallDefinition([0,-0.1],[0.05,-0.1], MAT, n_strips=2, name='bot')
    return CrossSection([web, f_top, f_bot])

def test_node_count():
    sec = c_section()
    # web: 5 nodes, top: 3 nodes, bot: 3 nodes, minus 2 shared at web ends = 9
    assert sec.n_nodes == 9

def test_strip_count():
    sec = c_section()
    assert sec.n_strips == 8  # 4 + 2 + 2

def test_shared_nodes_at_junctions():
    """Web start and top flange start must share a node."""
    sec = c_section()
    # Node at (0,0) should belong to both wall 0 and wall 1
    junction_nodes = [n for n in sec._nodes if len(n.wall_ids) > 1]
    assert len(junction_nodes) == 2, f"Expected 2 junctions, got {len(junction_nodes)}"

def test_dof_map_shape():
    sec = c_section()
    dm = sec.dof_map(4)
    assert dm.shape == (sec.n_nodes, 4)

def test_strip_global_dofs_length():
    sec = c_section()
    gdofs = sec.strip_global_dofs(0, 4)
    assert len(gdofs) == 8  # 2 nodes * 4 dofs

def test_centroid_inside_section():
    sec = c_section()
    yc, zc = sec.centroid()
    # y centroid should be between 0 and 0.05, z between -0.1 and 0
    assert 0 <= yc <= 0.05
    assert -0.1 <= zc <= 0

def test_abd_matrix_retrieved():
    sec = c_section()
    abd = sec.strip_abd(0)
    assert abd.shape == (6, 6)


def test_enclosed_area_raises_for_open_section():
    """Shoelace area is undefined for open strips; require a closed loop."""
    web = WallDefinition([0, 0], [0.1, 0], MAT, n_strips=4, name="open")
    sec = CrossSection([web])
    with pytest.raises(ValueError, match="enclosed_area\\(\\) is only valid for closed sections"):
        sec.enclosed_area()
