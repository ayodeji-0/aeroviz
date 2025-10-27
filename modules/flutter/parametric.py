from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff # for table and streamlines
from plotly.subplots import make_subplots

from .analysis import FlutterAnalysis
from utils.flutter.constants import ps_indep_dict, ps_dep_dict

class ParametricStudy:
    """Class to carry out parametric studies on flutter analysis."""

    def __init__(self, 
                 independent_var: str, 
                 min_val: float, 
                 max_val: float, 
                 step: float, 
                 dependent_vars: List[str]):
        """
        Initialize a parametric study.

        Parameters:
            independent_var (str): Parameter to vary (must be in ps_indep_dict)
            min_val (float): Minimum value for parameter
            max_val (float): Maximum value for parameter
            step (float): Step size for parameter variation
            dependent_vars (List[str]): List of dependent variables to observe (must be in ps_dep_dict)

        Raises:
            ValueError: If parameters are invalid or variables aren't in dictionaries
        """
        if independent_var not in ps_indep_dict:
            raise ValueError(f"independent_var must be one of {list(ps_indep_dict.keys())}")
        
        for var in dependent_vars:
            if var not in ps_dep_dict:
                raise ValueError(f"dependent variables must be from {list(ps_dep_dict.keys())}")

        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        if step <= 0:
            raise ValueError("step must be positive")
        if not dependent_vars:
            raise ValueError("dependent_vars cannot be empty")

        self.independent_var = independent_var
        self.min = min_val
        self.max = max_val
        self.step = step
        self.dependent_vars = dependent_vars
        self.var_count = len(dependent_vars)
        self.param = ps_indep_dict[independent_var]

        # Study Setup
        self.x_list = np.arange(self.min, self.max + self.step, self.step)
        self.results: Optional[NDArray[np.float64]] = None

    def run_study(self, sys_params: Tuple[float, ...]) -> None:
        """
        Run the parametric study.

        Parameters:
            sys_params (Tuple[float, ...]): System parameters for flutter analysis

        Raises:
            ValueError: If system parameters are invalid
        """
        # if len(sys_params) != 8:  # mu, sigma, V, a, b, e, r, w_theta
        #     raise ValueError("sys_params must contain exactly 8 parameters")

        # Initialize results array
        self.results = np.zeros((len(self.x_list), self.var_count + 1))

        # Initialize the flutter analysis object
        fa = FlutterAnalysis(*sys_params)

        # Loop through each value of the independent variable
        for i, x in enumerate(self.x_list):
            # Update parameter
            setattr(fa, self.param, x)
            # Compute the response
            fa.compute_response()

            # Store results
            self.results[i, 0] = x  # Independent variable value
            for j, dep_var in enumerate(self.dependent_vars):
                value = getattr(fa, ps_dep_dict[dep_var])
                self.results[i, j + 1] = value if np.isscalar(value) else np.mean(value)

    def plot(self) -> Optional[Figure]:
        """
        Plot the results of the parametric study.

        Returns:
            Optional[Figure]: Matplotlib figure object if results exist, None otherwise

        Raises:
            ValueError: If study hasn't been run yet
        """
        if self.results is None:
            raise ValueError("No results available. Run the study first!")

        # Create one plot per dependent variable
        fig, axes = plt.subplots(self.var_count, 1, figsize=(8, 4 * self.var_count), squeeze=False)
        axes = axes.flatten()

        for j in range(self.var_count):
            axes[j].plot(self.results[:, 0], self.results[:, j + 1], 'bo-', label=self.dependent_vars[j])
            axes[j].set_xlabel(self.independent_var)
            axes[j].set_ylabel(self.dependent_vars[j])
            axes[j].set_title(f"{self.dependent_vars[j]} vs. {self.independent_var}")
            axes[j].legend(bbox_to_anchor=(1, 1), loc='upper right')
            axes[j].grid(True)

        fig.tight_layout()
        fig.patch.set_facecolor('none')
        return fig

    # def plotly_plot(self):
    #     """
    #     Plot the results of the parametric study using Plotly.

    #     Returns:
    #         plotly.graph_objs.Figure: Plotly figure object if results exist

    #     Raises:
    #         ValueError: If study hasn't been run yet
    #     """


    #     if self.results is None:
    #         raise ValueError("No results available. Run the study first!")

    #     # Create subplot with one row per dependent variable
    #     fig = make_subplots(
    #         rows=self.var_count, 
    #         cols=1,
    #         subplot_titles=[f"${var}$ vs. ${self.independent_var}$" for var in self.dependent_vars],
    #         vertical_spacing=0.1
    #     )

    #     # Add traces for each dependent variable
    #     for j in range(self.var_count):
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.results[:, 0], 
    #                 y=self.results[:, j + 1],
    #                 mode='lines+markers',
    #                 name=self.dependent_vars[j],
    #                 line=dict(color='royalblue'),
    #                 marker=dict(size=8)
    #             ),
    #             row=j+1, 
    #             col=1
    #         )
            
    #         # Update axes labels
    #         fig.update_xaxes(
    #             title_text=f"${self.independent_var}$",
    #             row=j+1, 
    #             col=1,
    #             gridcolor='lightgray'
    #         )
            
    #         fig.update_yaxes(
    #             title_text=f"${self.dependent_vars[j]}$",
    #             row=j+1, 
    #             col=1,
    #             gridcolor='lightgray'
    #         )

    #     # Update layout
    #     fig.update_layout(
    #         height=300 * self.var_count,
    #         width=800,
    #         showlegend=True,
    #         legend=dict(
    #             yanchor="top",
    #             y=0.99,
    #             xanchor="right",
    #             x=0.99
    #         ),
    #         template="plotly_white",
    #         margin=dict(t=50, b=20, l=80, r=20),
    #     )

    #     return fig
    
    def plotly_plot(self):
        """
        Plot the results of the parametric study using Plotly.

        Returns:
            plotly.graph_objs.Figure: Plotly figure object if results exist

        Raises:
            ValueError: If study hasn't been run yet
        """
        if self.results is None:
            raise ValueError("No results available. Run the study first!")

        # Look up display names from the keys in your dictionaries
        # Use internal parameter name (self.param) to lookup the display version
        x_display = next((k for k, v in ps_indep_dict.items() if v == self.param), self.param)
        
        # Create subplot with one row per dependent variable
        fig = make_subplots(
            rows=self.var_count, 
            cols=1,
            subplot_titles=[f"{var} vs. {x_display}" for var in self.dependent_vars],
            vertical_spacing=0.1
        )

        # Add traces for each dependent variable
        colors = px.colors.qualitative.Plotly  # Use Plotly's default color sequence
        for j, var_name in enumerate(self.dependent_vars):
            # Get a color from the palette (cycling if needed)
            color = colors[j % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=self.results[:, 0], 
                    y=self.results[:, j + 1],
                    mode='lines+markers',
                    name=var_name,
                    line=dict(color=color),
                    marker=dict(size=8, color=color)
                ),
                row=j+1, 
                col=1
            )
            
            # Update axes labels
            fig.update_xaxes(
                title_text=x_display,  # Use the display name with LaTeX
                row=j+1, 
                col=1,
                gridcolor='lightgray'
            )
            
            fig.update_yaxes(
                title_text=var_name,  # Use the display name with LaTeX
                row=j+1, 
                col=1,
                gridcolor='lightgray'
            )

        # Update layout
        fig.update_layout(
            height=300 * self.var_count,
            width=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            template="plotly_white",
            margin=dict(t=80, b=20, l=80, r=20),  # Increased top margin for titles
            font=dict(
                family="Arial, sans-serif",
                size=12
            )
        )

        return fig