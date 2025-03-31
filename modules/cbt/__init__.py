"""
Coupled Bending Torsion Flutter Analysis Module.

This module provides classes and functions for analyzing coupled bending-torsion
flutter in aerodynamic systems.
"""

from .airfoil import Airfoil
from .analysis import FlutterAnalysis
from .parametric import ParametricStudy

__all__ = [
    'Airfoil', 
    'FlutterAnalysis', 
    'ParametricStudy'
]
