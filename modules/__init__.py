"""
AeroViz modules package.
"""

from .cbt.airfoil import Airfoil
from .cbt.analysis import FlutterAnalysis
from .cbt.parametric import ParametricStudy

__all__ = ['Airfoil', 'FlutterAnalysis', 'ParametricStudy']
