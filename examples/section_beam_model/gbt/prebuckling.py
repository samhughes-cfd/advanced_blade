"""prebuckling.py - Pre-buckling stress recovery from beam-level loads."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .section import CrossSection

# Numerical floors
_DET_FLOOR: float = 1e-30
"""Singularity guard for the bending rigidity determinant EIyy·EIzz − EIyz²."""
_MIN_ENCLOSED_AREA_M2: float = 1e-12
"""Minimum enclosed cross-section area [m²] (1 mm²) for Bredt closed-section torsion."""


@dataclass
class SectionLoads:
    N:float=0.0; My:float=0.0; Mz:float=0.0
    Vy:float=0.0; Vz:float=0.0; T:float=0.0

def _section_stiffness(section):
    EA=EAy=EAz=0.0
    for i in range(section.n_strips):
        A11=section.strip_abd(i)[0,0]; ds=section.get_strip(i).length
        mid=section.get_strip(i).midpoint(section._nodes)
        EA+=A11*ds; EAy+=A11*mid[0]*ds; EAz+=A11*mid[1]*ds
    yc=EAy/EA; zc=EAz/EA
    EIyy=EIzz=EIyz=0.0
    for i in range(section.n_strips):
        A11=section.strip_abd(i)[0,0]; ds=section.get_strip(i).length
        mid=section.get_strip(i).midpoint(section._nodes)
        dy=mid[0]-yc; dz=mid[1]-zc
        EIyy+=A11*dz**2*ds; EIzz+=A11*dy**2*ds; EIyz+=A11*dy*dz*ds
    return dict(EA=EA,EIyy=EIyy,EIzz=EIzz,EIyz=EIyz,yc=yc,zc=zc)

class PreBucklingAnalysis:
    def __init__(self, section, loads):
        self.section=section; self.loads=loads
        self._props=_section_stiffness(section)
    def section_properties(self): return dict(self._props)
    def axial_stress_resultants(self):
        p=self._props; EA=p["EA"]; EIyy=p["EIyy"]; EIzz=p["EIzz"]
        EIyz=p["EIyz"]; yc=p["yc"]; zc=p["zc"]
        N,My,Mz=self.loads.N,self.loads.My,self.loads.Mz
        det=EIyy*EIzz-EIyz**2
        if abs(det)<_DET_FLOOR: det=max(EIyy*EIzz,_DET_FLOOR)
        ky=( EIzz*My-EIyz*Mz)/det; kz=(-EIyz*My+EIyy*Mz)/det
        Nx=np.zeros(self.section.n_strips)
        for i in range(self.section.n_strips):
            A11=self.section.strip_abd(i)[0,0]
            mid=self.section.get_strip(i).midpoint(self.section._nodes)
            Nx[i]=A11*(N/EA+ky*(mid[1]-zc)+kz*(mid[0]-yc))
        return Nx
    def shear_flow(self):
        """Shear flow from transverse shear (strip integration).

        Torsional (Bredt-Batho) shear flow for closed cells is not included.
        This function is valid for the GBT pre-buckling state under pure
        bending/axial load. For torque-dominated load cases, results will
        be approximate.
        """
        p=self._props; EIyy=p["EIyy"]; EIzz=p["EIzz"]; EIyz=p["EIyz"]
        yc=p["yc"]; zc=p["zc"]; Vy=self.loads.Vy; Vz=self.loads.Vz
        det=EIyy*EIzz-EIyz**2
        q=np.zeros(self.section.n_strips)
        if abs(det)>1e-30:
            cy=(Vz*EIzz-Vy*EIyz)/det; cz=(Vy*EIyy-Vz*EIyz)/det
            Sy=Sz=0.0
            for i in range(self.section.n_strips):
                A11=self.section.strip_abd(i)[0,0]; ds=self.section.get_strip(i).length
                mid=self.section.get_strip(i).midpoint(self.section._nodes)
                Sz+=A11*(mid[0]-yc)*ds; Sy+=A11*(mid[1]-zc)*ds
                q[i]=-(cy*Sy+cz*Sz)
        if abs(self.loads.T)>0:
            Aenc=self.section.enclosed_area()
            if Aenc>_MIN_ENCLOSED_AREA_M2: q+=self.loads.T/(2.0*Aenc)
        return q
    def strip_stress_resultants(self):
        Nx=self.axial_stress_resultants(); q=self.shear_flow()
        return np.column_stack([Nx, np.zeros_like(Nx), q])
    def run(self): return self.strip_stress_resultants()