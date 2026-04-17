"""kinematics.py - Wall strip kinematic operators for GBT."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class KinematicModel(ABC):
    @abstractmethod
    def membrane_bkin(self, ds: float) -> NDArray: pass
    @abstractmethod
    def bending_bkin(self, ds: float) -> NDArray: pass
    def shear_bkin(self, ds: float) -> NDArray:
        return np.zeros((2, self.n_dof_per_strip))
    @property
    @abstractmethod
    def n_dof_per_strip(self) -> int: pass

class KirchhoffKinematics(KinematicModel):
    @property
    def n_dof_per_strip(self): return 8
    def membrane_bkin(self, ds):
        B = np.zeros((3, 8))
        B[1, 1] = -1.0/ds; B[1, 5] = 1.0/ds
        B[2, 0] = -1.0/ds; B[2, 4] = 1.0/ds
        return B
    def bending_bkin(self, ds):
        B = np.zeros((3, 8)); L = ds; xi = 0.5
        d2N1 = (-6.0 + 12.0*xi)/L**2
        d2N2 = (-4.0 +  6.0*xi)/L
        d2N3 = ( 6.0 - 12.0*xi)/L**2
        d2N4 = (-2.0 +  6.0*xi)/L
        B[1,2]=d2N1; B[1,3]=d2N2; B[1,6]=d2N3; B[1,7]=d2N4
        return B

class MindlinKinematics(KinematicModel):
    @property
    def n_dof_per_strip(self): return 10
    def membrane_bkin(self, ds):
        B = np.zeros((3, 10))
        B[1,1]=-1.0/ds; B[1,6]=1.0/ds
        B[2,0]=-1.0/ds; B[2,5]=1.0/ds
        return B
    def bending_bkin(self, ds):
        B = np.zeros((3, 10))
        B[1,3]=-1.0/ds; B[1,8]=1.0/ds
        return B
    def shear_bkin(self, ds):
        B = np.zeros((2, 10)); L = ds
        B[0,2]=-1.0/L; B[0,7]=1.0/L
        B[0,3]=-0.5;   B[0,8]=-0.5
        B[1,4]=-0.5;   B[1,9]=-0.5
        return B