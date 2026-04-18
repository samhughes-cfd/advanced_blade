"""section.py - Cross-section with shared-node wall connectivity."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
import numpy as np
from numpy.typing import NDArray

@dataclass
class SectionNode:
    node_id: int; y: float; z: float
    wall_ids: list = field(default_factory=list)
    @property
    def coords(self): return np.array([self.y, self.z])

@dataclass
class WallDefinition:
    node_start: object; node_end: object; material: object
    n_strips: int = 4; name: str = ""
    def __post_init__(self):
        self.node_start = np.asarray(self.node_start, float)
        self.node_end   = np.asarray(self.node_end,   float)
    @property
    def length(self): return float(np.linalg.norm(self.node_end - self.node_start))
    @property
    def tangent(self): return (self.node_end - self.node_start)/self.length
    @property
    def normal(self):
        t = self.tangent; return np.array([-t[1], t[0]])

@dataclass
class WallStrip:
    strip_id: int; wall_id: int; node_i: int; node_j: int
    length: float; material: object
    def midpoint(self, nodes):
        return 0.5*(nodes[self.node_i].coords + nodes[self.node_j].coords)

class CrossSection:
    def __init__(self, walls, node_tol=1e-9):
        self.walls = walls; self.node_tol = node_tol
        self._nodes = []; self._strips = []; self._build()

    def _add_or_get_node(self, y, z, wall_id):
        for n in self._nodes:
            if abs(n.y-y)<self.node_tol and abs(n.z-z)<self.node_tol:
                if wall_id not in n.wall_ids: n.wall_ids.append(wall_id)
                return n.node_id
        nid = len(self._nodes)
        self._nodes.append(SectionNode(nid, y, z, [wall_id]))
        return nid

    def _build(self):
        self._nodes.clear(); self._strips.clear(); sid = 0
        for wi, wall in enumerate(self.walls):
            n = wall.n_strips
            ts = np.linspace(0,1,n+1)
            coords = np.outer(1-ts, wall.node_start)+np.outer(ts, wall.node_end)
            nids = [self._add_or_get_node(coords[k,0], coords[k,1], wi) for k in range(n+1)]
            ds = wall.length/n
            for k in range(n):
                self._strips.append(WallStrip(sid, wi, nids[k], nids[k+1], ds, wall.material))
                sid += 1

    @property
    def n_nodes(self): return len(self._nodes)
    @property
    def n_strips(self): return len(self._strips)
    @property
    def node_coordinates(self): return np.array([[n.y,n.z] for n in self._nodes])
    def get_strip(self, i): return self._strips[i]
    def get_node(self, i):  return self._nodes[i]

    def dof_map(self, n_dof_per_node):
        return np.arange(self.n_nodes*n_dof_per_node).reshape(self.n_nodes, n_dof_per_node)

    def strip_global_dofs(self, strip_id, n_dof_per_node):
        s = self._strips[strip_id]; dm = self.dof_map(n_dof_per_node)
        return np.concatenate([dm[s.node_i], dm[s.node_j]])

    def strip_abd(self, i):   return self._strips[i].material.abd_matrix()
    def strip_shear_stiffness(self, i):
        m = self._strips[i].material
        return m.shear_stiffness() if hasattr(m,"shear_stiffness") else np.zeros((2,2))
    def strip_thickness(self, i):
        m = self._strips[i].material
        for a in ("t","total_thickness"):
            if hasattr(m,a): return float(getattr(m,a))
        return 1e-3
    def all_abd_matrices(self): return [self.strip_abd(i) for i in range(self.n_strips)]

    def centroid(self):
        total=ysum=zsum=0.0
        for s in self._strips:
            mid=s.midpoint(self._nodes); ysum+=mid[0]*s.length; zsum+=mid[1]*s.length; total+=s.length
        return np.array([ysum/total, zsum/total])

    def second_moments(self):
        yc,zc=self.centroid(); Iyy=Izz=Iyz=0.0
        for s in self._strips:
            mid=s.midpoint(self._nodes); dy=mid[0]-yc; dz=mid[1]-zc; ds=s.length
            Iyy+=dz**2*ds; Izz+=dy**2*ds; Iyz+=dy*dz*ds
        return {"Iyy":Iyy,"Izz":Izz,"Iyz":Iyz}

    def enclosed_area(self):
        pts = np.array([self._nodes[s.node_i].coords for s in self._strips])
        area=0.0
        for i in range(len(pts)):
            j=(i+1)%len(pts); area+=pts[i,0]*pts[j,1]-pts[j,0]*pts[i,1]
        return abs(area)/2.0

    def extensional_stiffness(self) -> float:
        """Stripwise axial stiffness ``∑ A₁₁ ds`` [N] from each strip's CLPT ``A`` matrix."""
        ea = 0.0
        for i in range(self.n_strips):
            a11 = float(self.strip_abd(i)[0, 0])
            ea += a11 * float(self.get_strip(i).length)
        return float(ea)

    def validate(self):
        issues=[]
        for i,w in enumerate(self.walls):
            if w.length<1e-12: issues.append(f"Wall {i} zero length")
        ends={}
        for s in self._strips:
            for nid in (s.node_i, s.node_j): ends[nid]=ends.get(nid,0)+1
        dangle=[nid for nid,c in ends.items() if c==1]
        if dangle: issues.append(f"Dangling nodes: {dangle} (open section?)")
        return issues

    def summary(self):
        lines=[f"CrossSection: {len(self.walls)} walls, {self.n_strips} strips, {self.n_nodes} nodes"]
        for i,w in enumerate(self.walls):
            lines.append(f"  Wall {i:2d} '{w.name}': length={w.length:.4f}m  n_strips={w.n_strips}  mat={type(w.material).__name__}")
        for iss in self.validate(): lines.append(f"  ⚠ {iss}")
        return "\n".join(lines)