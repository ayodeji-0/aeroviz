from typing import Dict, Any, Tuple, Union, List
from dataclasses import dataclass

Number = Union[int, float]
SystemConfigValue = List[Union[str, Tuple[Number, Number, Number, Number], str]]


airfoil_parameters = {
    #each value is a list of a tuple of (min, max, step, default), help text
   
    'Max Camber': [(0.0, 9.5, 0.1, 0.0), "Maximum camber as a percentage of the chord. Controls the curvature of the mean camber line, affecting lift characteristics."],
    'Camber Position': [(0.0, 9.0, 0.1, 0.0), "Position along the chord where maximum camber occurs. Determines where the airfoil bends most."],
    'Thickness': [(1.0, 40.0, 0.5, 12.0), "Maximum thickness of the airfoil as a percentage of the chord. Influences structural strength and aerodynamic performance."],
    'Length': [(0.0, 10.0, 0.1, 1.0), "Scaling factor applied to the chord length. Sets the physical size of the airfoil model."],
    'Discretization': [(10, 100, 25, 100), "Number of points used to define the airfoil surface. Higher values increase resolution for plotting and computation."],
    'Centre Position': [(0.0, 1.0, 0.01, 0.5), "Horizontal position of the airfoil center along its span."],

     
}
Number = Union[int, float]

system_configuration: Dict[str, SystemConfigValue] = {
    # each value is a list of LaTex enabled Name, tuple of (min, max, step, default), help text
   

    'mu': ['Mass Ratio · $μ$', (0.1, 100.0, 0.1, 20.0), "Mass ratio between the structure and the surrounding air. Influences the system's inertia and overall stability."],
    'sigma': ['Frequency Ratio · $σ$', (0.1, 2.0, 0.01, 0.4), "Ratio of bending to torsional natural frequencies. Determines how strongly the two modes interact."],
    'V': ['Reduced Velocity · $V$', (0.1, 100.0, 0.1, 2.0), "Reduced velocity of the freestream relative to structural dynamics. Key parameter for identifying flutter onset."],
    'a': ['Torsional Axis Location · $a$', (0.0, 1.0, 0.01, 0.0), "Location of the torsional axis along the chord. Influences aerodynamic moment arm and coupling strength."],
    'b': ['Semi-Chord Length · $b$', (0.1, 1.0, 0.01, 0.5), "Half the chord length of the airfoil. Used as a reference scale in defining motion and aerodynamic effects."],
    'e': ['Eccentricity · $e$', (-0.5, 1.0, 0.01, 0.0), "Position of the center of mass along the chord. Affects inertial coupling between bending and torsion."],
    'r': ['Radius of Gyration · $r$', (0.01, 1.0, 0.01, 0.5), "Non-dimensional measure of rotational inertia. Governs the blade’s resistance to angular acceleration."],
    'w_theta': ['Torsional Vibration Frequency · $\\omega_\\theta$', (0.0, 1000.0, 1.0, 100.0), "Natural frequency of torsional motion in a vacuum. Sets the time scale for torsional response."],
}              


# Dictionary of all possible modes for analysis
mode_options: Dict[str, str] = {
    'Steady - State Space': 'Steady - State Space',
    'Quasi Steady - State Space': 'Quasi Steady - State Space',
}

# Dictionary of all possible independent variables for the parametric study
ps_indep_dict: Dict[str, str] = {
    'Mass Ratio · $\\mu$': 'mu',
    'Frequency Ratio · $\\sigma$': 'sigma',
    'Reduced Velocity · $V$': 'V',
    'Torsional Axis Location · $a$': 'a',
    'Semi-Chord Length · $b$': 'b',
    'Eccentricity · $e$': 'e',
    'Radius of Gyration · $r$': 'r',
    'Torsional Vibration Frequency · $\\omega_\\theta$': 'w_theta'  
}

# Dictionary of all possible dependent variables for the parametric study
ps_dep_dict: Dict[str, str] = {
    'Aeroelastic Frequency · $\\omega/\\omega_{\\theta}$': 'omega',
    'Damping Ratio · $\\zeta$': 'zeta',
    'Eigenvalues': 'vals',
    'Eigenvectors': 'vecs'
}

help_text = {
    # Aesthetics & Playback
    'transparency': "Transparency of the airfoil fill color",
    "duration": "Duration of the animation in seconds",
    "fps": "Frames per second for the animation",

    # Parametric Study
    'independent_variable': "Parameter to vary",
    'dependent_variable': "Parameters to observe as a function of the independent variable",
}

# Create similar dicts for aesthetics and playback
# Include parametric study variables help text for indep dict



@dataclass
class PlotParams:
    """Class to hold plot styling parameters."""
    text_color: str = "white"
    legend_edge_color: str = "black"
    legend_label_color: str = "black"
    axes_label_color: str = "white"
    xtick_color: str = "white"
    ytick_color: str = "white"

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to matplotlib-compatible dictionary."""
        return {
            "text.color": self.text_color,
            "legend.edgecolor": self.legend_edge_color,
            "legend.labelcolor": self.legend_label_color,
            "axes.labelcolor": self.axes_label_color,
            "xtick.color": self.xtick_color,
            "ytick.color": self.ytick_color
        }

# Create default plot parameters
plot_params = PlotParams().to_dict()
