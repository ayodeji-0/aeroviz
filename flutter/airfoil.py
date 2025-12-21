from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from numpy.typing import NDArray
import streamlit as st

class Airfoil:
    """
    Class to represent a NACA 4-digit airfoil with variable parameters.
    """  
    def __init__(self, 
                 max_camber: float, 
                 camber_position: float, 
                 thickness: float, 
                 num_points: int = 100, 
                 length: int = 1, 
                 centrepos: float = 0.5):
        """
        Initialize an airfoil with given parameters.

        Parameters:
            max_camber (float): Maximum camber as percentage of chord (0-9.9)
            camber_position (float): Position of max camber as fraction of chord (0-0.9)
            thickness (float): Maximum thickness as percentage of chord (0-40)
            num_points (int): Number of points for discretization (default: 100)
            length (float): Chord length (default: 1.0)
            centrepos (float): Center position along chord (default: 0.5)

        Raises:
            ValueError: If any parameter is outside its valid range
        """
        # Input validation
        # if not 0 <= max_camber <= 9.9:
        #     raise ValueError("max_camber must be between 0 and 9.9")
        if not 0 <= camber_position <= 9.0:
            raise ValueError("camber_position must be between 0 and  9.0")
        if not 0 <= camber_position <= 9:
            raise ValueError("camber_position must be between 0 and 9")
        if not 0 <= thickness <= 40:
            raise ValueError("thickness must be between 0 and 40")
        if num_points < 10:
            raise ValueError("num_points must be at least 10")
        if length <= 0:
            raise ValueError("length must be positive")
        if not 0 <= centrepos <= 1:
            raise ValueError("centrepos must be between 0 and 1")

        self.max_camber = max_camber
        self.camber_position = camber_position
        self.thickness = thickness
        self.num_points = num_points
        self.length = length
        self.centrepos = centrepos
        self.coords: Optional[NDArray[np.float64]] = None
        self.code: Optional[str] = None

    def generate_naca_airfoil4(self) -> None:
        """
        Generate coordinates for a 4-digit NACA airfoil.

        Updates the coords and code attributes of the airfoil instance.
        """
        # Convert parameters to decimals
        m = self.max_camber / 100
        p = self.camber_position / 10
        t = self.thickness / 100
        
        beta = np.linspace(0, np.pi, self.num_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Initialize arrays
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        # Compute camber line and its slope
        for i in range(self.num_points):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
        
        theta = np.arctan(dyc_dx)

        # Calculate surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine coordinates
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
                                  
        # Scale and center
        x_coords *= self.length
        y_coords *= self.length
        x_coords -= self.centrepos * self.length

        self.coords = np.column_stack((x_coords, y_coords))
        self.code = f"{int(self.max_camber)}{int(self.camber_position)}{int(self.thickness):02d}"

    def update(self, 
              max_camber: float, 
              camber_pos: float, 
              thickness: float, 
              num_points: int = 100, 
              chord_length: float = 1.0, 
              centrepos: float = 0.5) -> None:
        """
        Update airfoil parameters and regenerate coordinates.

        Parameters:
            max_camber (float): Maximum camber as percentage of chord (0-9.9)
            camber_pos (float): Position of max camber as fraction of chord (0-0.9)
            thickness (float): Maximum thickness as percentage of chord (0-40)
            num_points (int): Number of points for discretization (default: 100)
            chord_length (float): Chord length (default: 1.0)
            centrepos (float): Center position along chord (default: 0.5)

        Raises:
            ValueError: If any parameter is outside its valid range
        """
        # Create a new instance to validate parameters
        temp = Airfoil(max_camber, camber_pos, thickness, num_points, chord_length, centrepos)
        
        # If validation passed, update attributes
        self.max_camber = max_camber
        self.camber_position = camber_pos
        self.thickness = thickness
        self.num_points = num_points
        self.length = chord_length
        self.centrepos = centrepos
        
        # Regenerate coordinates
        self.generate_naca_airfoil4()
    
    def plot(self, show_chord: bool = False, color: str = 'blue', alpha: float = 0.6) -> Figure:
        """
        Plot the airfoil shape.

        Parameters:
            show_chord (bool): Whether to display chord line
            color (str): Color for airfoil fill
            alpha (float): Transparency level (0-1)

        Returns:
            Figure: Matplotlib figure object

        Raises:
            ValueError: If coordinates haven't been generated or alpha is invalid
        """
        if self.coords is None:
            raise ValueError("Airfoil coordinates are not initialized. Call generate_naca_airfoil4() first.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        fig, ax = plt.subplots()

        # Fill the enclosed airfoil shape
        ax.fill(self.coords[:, 0], self.coords[:, 1], color=color, alpha=alpha, edgecolor='black', linewidth=2)

        # Show chord line if requested
        if show_chord:
            ax.axhline(0, color='r', linestyle='--', lw=1)
            ax.text(self.coords[:, 0].mean(), min(self.coords[:, 1]) - 0.05, 
                   'Chord Line', color='r', ha='center')

        # Formatting
        ax.set_aspect('equal', 'box')
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)
        ax.axis('off')

        return fig
    