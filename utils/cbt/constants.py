from typing import Dict, Any
from dataclasses import dataclass

# Dictionary of all possible independent variables for the parametric study
ps_indep_dict: Dict[str, str] = {
    'Mass Ratio · μ': 'mu',
    'Frequency Ratio · σ': 'sigma',
    'Reduced Velocity · V': 'V',
    'Torsional Axis Location · a': 'a',
    'Semi-Chord Length · b': 'b',
    'Eccentricity · e': 'e',
    'Radius of Gyration · r': 'r',
    'Torsional Vibration Frequency · w_θ': 'w_theta'  
}

# Dictionary of all possible dependent variables for the parametric study
ps_dep_dict: Dict[str, str] = {
    'Aeroelastic Frequency · w/w_ϴ': 'omega',
    'Damping Ratio · ζ': 'zeta',
    'Eigenvalues': 'vals',
    'Eigenvectors': 'vecs'
}

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
