from typing import Dict, Any, Tuple, Union, List
from dataclasses import dataclass

Number = Union[int, float]

materials = {
            'Aluminum': {'density': 2700, 'E': 70e9, 'G': 26e9, 'v': 0.33},
            'Mild Steel': {'density': 7850, 'E': 200e9, 'G': 77e9, 'v': 0.3},
            'Hardened Steel': {'density': 7850, 'E': 210e9, 'G': 81e9, 'v': 0.3},
            'Titanium': {'density': 4500, 'E': 116e9, 'G': 45e9, 'v': 0.3},
            'ANSYS Structural Steel': {'density': 7850, 'E': 2e11, 'G': 7.6923e10, 'v': 0.3},
            }
