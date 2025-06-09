import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
import scipy.linalg as la
import streamlit as st
from IPython.display import HTML

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff # for table and streamlines
from plotly.subplots import make_subplots

class FlutterAnalysis:
    """
    Class to represent flutter analysis for a coupled system.
    """

    def __init__(self, mu, sigma, V, a, b, e, r, mode='Steady - State Space', w_theta=100):
        self.mu = mu
        self.sigma = sigma
        self.V = V
        self.a = a
        self.b = b
        self.e = e
        self.r = r
        self.mode = mode
        self.w_theta = w_theta

        self.x_theta = None
        self.vals = None
        self.vecs = None
        self.omega = None
        self.zeta = None

    def compute_response(self):
        """
        Compute the eigenvalues and eigenvectors for analysis of the flutter response.
        """
        # Torsional Axis Offset
        self.x_theta = (self.e - self.a)

        # Mass and Stiffness Matrices
        M = np.array([
            [1, self.x_theta],
            [self.x_theta, self.r**2]
        ])

        if self.mode == 'Steady - State Space':
            K = np.array([
                [self.sigma**2 / self.V**2, 2/self.mu],
                [0, (self.r**2 / self.V**2) - ((2 / self.mu) * (self.a + 0.5))]
            ])

            # Create state space representation
            A = np.block([
                [np.zeros_like(M), -np.eye(2)],
                [la.inv(M) @ (K), np.zeros_like(M)]
            ])

            p, self.vecs = la.eig(A)

            # Dimensionalize eigenvalues
            self.vals = p * self.V * self.w_theta
            
            # Split lambda into components
            self.omega = np.abs(self.vals.imag)  # Frequency component
            self.zeta = -self.vals.real / np.abs(self.vals.imag)  # Damping ratio

        #elif self.mode == 'Quasi Steady - State Space':
        elif self.mode == 'Quasi Steady - State Space':
            raise NotImplementedError("Quasi Steady mode is not implemented yet, use Steady for now.")

        else:
            raise ValueError("Only Steady and Quasi Steady modes are currently implemented, with state space representation.")

    def plot_displacements(self, duration=10, width=600, height=600):
        """
        Plot the displacement time history for flutter analysis.
        """
        if self.vals is None or self.vecs is None:
            raise ValueError("Run compute_response() first")

        t = np.linspace(0, duration, 500)
        
        # Extract eigenvalues and eigenvectors
        lambda_vals = self.vals[:4]
        real_parts = np.real(lambda_vals)
        imag_parts = np.imag(lambda_vals)
        
        h_tidals = np.real(self.vecs[0, :4]) * self.b
        theta_tidals = np.real(self.vecs[1, :4])

        # Compute phase differences
        phase_diffs = np.angle(self.vecs[1, :4] / self.vecs[0, :4])

        # Set up plotting
        width /= 100
        height /= 100
        fig, axes = plt.subplots(4, 1, figsize=(width, height))
        fig.suptitle("Coupled Flutter Modes - Time Response", fontsize=14)
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)

        # Compute and plot displacements
        h_t = np.array([
            h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t) 
            for i in range(4)
        ])

        theta_t = np.array([
            theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i])
            for i in range(4)
        ])

        for i in range(4):
            axes[i].plot(t, h_t[i], 'b-', label=f"Mode {i+1} Plunge")
            axes[i].plot(t, theta_t[i], 'r--', label=f"Mode {i+1} Twist")
            axes[i].set_xlabel("Vibration Period")
            axes[i].set_ylabel("Displacement")
            axes[i].legend()
            axes[i].grid()
            axes[i].set_title(f"Mode {i+1} Displacement")

        return fig

    def plotly_plot_displacements(self, duration=10, width=600, height=600):
            """
            Plot the displacement time history for flutter analysis using Plotly.
            """
            if self.vals is None or self.vecs is None:
                raise ValueError("Eigenvalues and eigenvectors are not computed. Run analysis first.")
            
            # Time discretization
            t = np.linspace(0, duration, 500)
            
            # Extract eigenvalues and eigenvectors
            lambda_vals = self.vals[:4]  # Select first 4 eigenvalues
            real_parts = np.real(lambda_vals)  # Gamma (damping)
            imag_parts = np.imag(lambda_vals)  # Omega (frequency)
            
            # Extract plunge and torsional displacements
            h_tidals = np.real(self.vecs[0, :4]) * self.b  # Extract real plunge displacements
            theta_tidals = np.real(self.vecs[1, :4])  # Extract real torsional displacements
            
            # Compute phase differences
            phase_diffs = np.angle(self.vecs[1, :4] / self.vecs[0, :4])
            
            # Define time range
            t = np.linspace(0, duration, duration * 100)
            
            # Create plotly subplots
            fig = make_subplots(rows=4, cols=1, 
                                subplot_titles=[f"Mode {i+1} Displacement" for i in range(4)],
                                vertical_spacing=0.1)
            
            # Precompute displacement histories
            h_t = np.array([
                h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t) for i in range(4)
            ])
            
            theta_t = np.array([
                theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i])
                for i in range(4)
            ])
            
            # Add traces for each mode
            for i in range(4):
                fig.add_trace(
                    go.Scatter(x=t, y=h_t[i], name=f"Mode {i+1} Plunge", line=dict(color="blue")),
                    row=i+1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=t, y=theta_t[i], name=f"Mode {i+1} Twist", line=dict(color="red", dash="dash")),
                    row=i+1, col=1
                )
            
            # Update layout
            fig.update_layout(
                height=height,
                width=width,
                title_text="Coupled Flutter Modes - Time Response",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,30,30,0.3)",
                font=dict(color="white"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Vibration Period", gridcolor="gray", showgrid=True)
            fig.update_yaxes(title_text="Displacement", gridcolor="gray", showgrid=True)
            
            return fig
    
    
    # def animate_flutter(self, airfoil_coords, duration=10, fps=30, scale=1.0, properties=None):
    #     """
    #     Animate the flutter response showing airfoil motion.

    #     airfoil_coords:  (N,2) array of the nominal airfoil coordinates in a "chord" frame,
    #                     e.g. x ∈ [0, 1], y ∈ [–0.1, +0.1].
    #     scale:           a scalar to resize those coords on‐screen.
    #     properties:      dict with keys:
    #                     - 'airfoil_color':   color string for facecolor
    #                     - 'transparency':    between 0–100 (percent opacity)
    #                     - 'show_chord':      bool, whether to draw the chord line
    #     """

    #     # Default properties if None
    #     if properties is None:
    #         properties = {
    #             'airfoil_color': '#ffffff',
    #             'transparency': 50,
    #             'show_chord': True
    #         }

    #     # Progress bar in Streamlit
    #     anim_bar = st.progress(0, text="Rendering Animation...")
    #     self.progress_bar = anim_bar

    #     # Make sure we have eigen‐stuff computed
    #     if getattr(self, 'vals', None) is None or getattr(self, 'vecs', None) is None:
    #         self.compute_response()

    #     # Extract the first four eigen‐pairs
    #     lambda_vals = self.vals[:4]              # shape (4,)
    #     real_parts = np.real(lambda_vals)        # damping “gamma”
    #     imag_parts = np.imag(lambda_vals)        # frequency “omega”

    #     # Eigenvectors: assume vecs shape is (2, M) or (n,4)
    #     # Here we take the first 4 modes: row 0 = plunge shape, row 1 = torsion shape
    #     h_tidals = np.real(self.vecs[0, :4]) * self.b      # real “plunge” amplitude scaled by semi‐chord
    #     theta_tidals = np.real(self.vecs[1, :4])           # real “twist” amplitude

    #     # Phase difference between plunge & pitch
    #     phase_diffs = np.angle(self.vecs[1, :4] / self.vecs[0, :4])

    #     # Time‐axis
    #     t = np.linspace(0, duration, int(duration * fps))

    #     # 1) Prepare “centered + scaled” airfoil coordinates
    #     #    Here we assume `airfoil_coords` is something like a (N,2) array
    #     #    with x-values from 0 to 1, y-values from –something..+something.
    #     #    We want to shift it so that its quarter‐chord (or elastic axis) is at x=0.
    #     #    Let’s assume the nominal “chord” is max(x)–min(x). We can center around its midpoint.
    #     coords = np.array(airfoil_coords) * scale

    #     # Center about the mean chord location: find mid‐chord
    #     x_coords = coords[:, 0]
    #     mid_chord = 0.5 * (x_coords.max() + x_coords.min())
    #     coords[:, 0] -= mid_chord

    #     # Now coords are roughly centered around x=0. If you want the elastic axis at x=a,
    #     # you can add +self.a to these x‐values and then rotate around that same point.
    #     coords[:, 0] += self.a

    #     # 2) Set up the figure with 4 rows, 2 columns
    #     fig, axes = plt.subplots(
    #         nrows=4, ncols=2,
    #         figsize=(8, 8),
    #         gridspec_kw={'width_ratios': [1.2, 1]}
    #     )
    #     fig.suptitle("Coupled Flutter Modes ‐ Time Response", fontsize=14)

    #     # 3) Precompute plunge & twist histories
    #     h_t = np.array([
    #         h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t)
    #         for i in range(4)
    #     ])
    #     theta_t = np.array([
    #         theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i])
    #         for i in range(4)
    #     ])

    #     # 4) Create one Polygon patch per mode, **in its “zero‐deflection”** position
    #     airfoil_patches = []
    #     for i in range(4):
    #         patch = Polygon(
    #             coords,
    #             closed=True,
    #             edgecolor='k',
    #             facecolor=properties['airfoil_color'],
    #             alpha=properties['transparency'] / 100.0
    #         )
    #         # Add it to the left‐column axes
    #         axes[i, 0].add_patch(patch)

    #         # Draw chord line if requested
    #         if properties.get('show_chord', False):
    #             chord_y0 = coords[:, 1].mean()
    #             axes[i, 0].plot(
    #                 [coords[:, 0].min(), coords[:, 0].max()],
    #                 [chord_y0, chord_y0],
    #                 'k--', lw=0.8
    #             )

    #         # Set a symmetric view window around x=0
    #         span = 1.5 * self.b
    #         axes[i, 0].set_xlim(-span + self.a, span + self.a)
    #         axes[i, 0].set_ylim(-span, span)
    #         axes[i, 0].set_aspect('equal')
    #         axes[i, 0].set_title(f"Mode {i+1} Animation")
    #         airfoil_patches.append(patch)

    #     # 5) On the right column, plot the time histories of h_t and theta_t
    #     for i in range(4):
    #         axes[i, 1].plot(t, h_t[i], 'b-', label=f"Mode {i+1} Plunge")
    #         axes[i, 1].plot(t, theta_t[i], 'r--', label=f"Mode {i+1} Twist")
    #         axes[i, 1].set_xlabel("Time [s]")
    #         axes[i, 1].set_ylabel("Displacement")
    #         axes[i, 1].legend(fontsize=8)
    #         axes[i, 1].grid(True)
    #         axes[i, 1].set_title(f"Mode {i+1} Amplitude")

    #     # 6) Animation update function: rotate about (self.a,0) & translate in y by h_t
    #     def update(frame):
    #         # Update progress bar
    #         pct = int((frame / len(t)) * 100)
    #         elapsed = frame / fps
    #         self.progress_bar.progress(
    #             pct,
    #             text=f"Time Elapsed: {elapsed:0.1f}s  (Rendering…)"
    #         )

    #         for i in range(4):
    #             # Build a composite Affine2D: first rotate around (self.a, 0),
    #             # then translate vertically by h_t[i,frame].
    #             rot = transforms.Affine2D().rotate_around(
    #                 self.a, 0.0,
    #                 theta_t[i, frame]  # already in radians
    #             )
    #             trans = rot.translate(0.0, h_t[i, frame]) + axes[i, 0].transData

    #             airfoil_patches[i].set_transform(trans)

    #         return airfoil_patches

    #     # 7) Create the FuncAnimation
    #     ani = FuncAnimation(
    #         fig,
    #         update,
    #         frames=len(t),
    #         blit=True,
    #         interval=1000 / fps
    #     )

    #     # 8) Once done, clear the progress bar and return the HTML for Streamlit
    #     anim_html = ani.to_jshtml()
    #     self.progress_bar.empty()
    #     plt.close(fig)   # close the figure so it doesn’t display twice

    #     return anim_html
    # def animate_flutter(self, airfoil_coords, duration=10, fps=30, properties=None):
    #     """
    #     Animate the flutter response showing airfoil motion.
    #     """
    #     if properties is None:
    #         properties = {
    #             'airfoil_color': '#ffffff',
    #             'transparency': 50,
    #             'show_chord': True
    #         }

    #     anim_bar = st.progress(0, text="Rendering Animation...")
    #     self.progress_bar = anim_bar

    #     if self.vals is None:
    #         self.compute_response()

    #     # Extract eigenvalues and eigenvectors
    #     lambda_vals = self.vals[:4]
    #     real_parts = np.real(lambda_vals)
    #     imag_parts = np.imag(lambda_vals)
        
    #     h_tidals = np.real(self.vecs[0, :4]) * self.b
    #     theta_tidals = np.real(self.vecs[1, :4])
    #     phase_diffs = np.angle(self.vecs[1, :4] / self.vecs[0, :4])

    #     t = np.linspace(0, duration, duration * fps)

    #     # Setup figure
    #     fig, axes = plt.subplots(4, 2, figsize=(8, 8))
    #     fig.suptitle("Coupled Flutter Modes - Time Response", fontsize=14)

    #     # Create airfoil patches
    #     airfoil_patches = []
    #     for i in range(4):
    #         patch = Polygon(airfoil_coords, closed=True, edgecolor='k', 
    #                       facecolor=properties['airfoil_color'], 
    #                       alpha=properties['transparency']/100)
    #         axes[i, 0].add_patch(patch)
    #         axes[i, 0].set_xlim(-1.5 * self.b, 1.5 * self.b)
    #         axes[i, 0].set_ylim(-1.5 * self.b, 1.5 * self.b)
    #         axes[i, 0].set_aspect('equal')
    #         axes[i, 0].set_title(f"Mode {i+1} Animation")
    #         airfoil_patches.append(patch)

    #     # Compute displacements
    #     h_t = np.array([
    #         h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t)
    #         for i in range(4)
    #     ])

    #     theta_t = np.array([
    #         theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i])
    #         for i in range(4)
    #     ])

    #     # Plot displacement histories
    #     for i in range(4):
    #         axes[i, 1].plot(t, h_t[i], 'b-', label=f"Mode {i+1} Plunge")
    #         axes[i, 1].plot(t, theta_t[i], 'r--', label=f"Mode {i+1} Twist")
    #         axes[i, 1].set_xlabel("Vibration Period")
    #         axes[i, 1].set_ylabel("Displacement")
    #         axes[i, 1].legend()
    #         axes[i, 1].grid()
    #         axes[i, 1].set_title(f"Mode {i+1} Amplitude & Phase")

    #     def update(frame):
    #         progress = int((frame / len(t)) * 100)
    #         self.progress_bar.progress(progress, text=f"Time Elapsed: {int(frame/fps)}s\nRendering Animation...")
            
    #         for i in range(4):
    #             trans = transforms.Affine2D().rotate_deg_around(
    #                 self.a, 0, np.degrees(theta_t[i, frame])
    #             ).translate(0, h_t[i, frame]) + axes[i, 0].transData
    #             airfoil_patches[i].set_transform(trans)
            
    #         return airfoil_patches

    #     ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000/fps)
    #     anim = ani.to_jshtml()
    #     self.progress_bar.empty()
        
    #     return anim

    # def animate_flutter(self, airfoil_coords, duration=10, fps=30, scale=1.0, n_modes=4, properties=None):
    #     """
    #     Animate the flutter response showing airfoil motion.

    #     Parameters:
    #     -----------
    #     airfoil_coords : array
    #         (N,2) array of the nominal airfoil coordinates in a "chord" frame,
    #         e.g. x ∈ [0, 1], y ∈ [–0.1, +0.1].
    #     duration : float, optional
    #         Animation duration in seconds, default 10
    #     fps : int, optional
    #         Frames per second, default 30
    #     scale : float, optional
    #         Scalar to resize coordinates on-screen, default 1.0
    #     n_modes : int, optional
    #         Number of modes to display (1-4), default 4
    #     properties : dict, optional
    #         Dictionary with visualization properties:
    #         - 'airfoil_color': color string for facecolor (default '#ffffff')
    #         - 'transparency': between 0-100 (percent opacity) (default 50)
    #         - 'show_chord': bool, whether to draw chord line (default True)
    #         - 'angled_text': bool, whether to angle mode labels (default False)
    #         - 'annotated_text_color': color for text annotations (default 'black')

    #     Returns:
    #     --------
    #     str: HTML string containing the animation for Streamlit display
    #     """
    #     # Ensure n_modes is between 1 and 4
    #     n_modes = max(1, min(4, n_modes))
        
    #     # Default properties if None
    #     if properties is None:
    #         properties = {
    #             'airfoil_color': '#ffffff',
    #             'transparency': 50,
    #             'show_chord': True,
    #             'angled_text': False,
    #             'annotated_text_color': 'black'
    #         }

    #     # Progress bar in Streamlit
    #     anim_bar = st.progress(0, text="Rendering Animation...")
    #     self.progress_bar = anim_bar

    #     # Make sure we have eigenvalues computed
    #     if getattr(self, 'vals', None) is None or getattr(self, 'vecs', None) is None:
    #         self.compute_response()

    #     # Extract the eigenvalues and eigenvectors for the modes
    #     lambda_vals = self.vals[:n_modes]           # shape (n_modes,)
    #     real_parts = np.real(lambda_vals)           # damping "gamma"
    #     imag_parts = np.imag(lambda_vals)           # frequency "omega"

    #     # Extract the eigenvectors
    #     h_tidals = np.real(self.vecs[0, :n_modes]) * self.b   # plunge amplitude
    #     theta_tidals = np.real(self.vecs[1, :n_modes])        # twist amplitude

    #     # Phase difference between plunge & pitch
    #     phase_diffs = np.angle(self.vecs[1, :n_modes] / self.vecs[0, :n_modes])

    #     # Time axis
    #     t = np.linspace(0, duration, int(duration * fps))

    #     # Prepare airfoil coordinates
    #     coords = np.array(airfoil_coords) * scale

    #     # Center about the mean chord location
    #     x_coords = coords[:, 0]
    #     mid_chord = 0.5 * (x_coords.max() + x_coords.min())
    #     coords[:, 0] -= mid_chord

    #     # Shift so elastic axis is at x=a
    #     coords[:, 0] += self.a

    #     # Set up the figure with n_modes rows, each with a 1:2 grid
    #     # Left column: animation, Right column: two stacked plots
    #     fig = plt.figure(figsize=(12, 3.5 * n_modes))
    #     gs = fig.add_gridspec(n_modes, 3, width_ratios=[2, 1, 1])
        
    #     fig.suptitle("Coupled Flutter Modes - Time Response", fontsize=14)
    #     fig.patch.set_facecolor('none')
    #     fig.patch.set_alpha(0)
        
    #     # Precompute displacement histories
    #     h_t = np.array([
    #         h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t)
    #         for i in range(n_modes)
    #     ])
    #     theta_t = np.array([
    #         theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i])
    #         for i in range(n_modes)
    #     ])

    #     # Lists to store all animation objects
    #     airfoil_patches = []
    #     time_lines = []
    #     phase_points = []
        
    #     # Create subplots and initialize plots
    #     for i in range(n_modes):
    #         # Animation subplot (left)
    #         ax_anim = fig.add_subplot(gs[i, 0])
            
    #         # Create polygon patch for airfoil
    #         patch = Polygon(
    #             coords,
    #             closed=True,
    #             edgecolor='k',
    #             facecolor=properties['airfoil_color'],
    #             alpha=properties['transparency'] / 100.0
    #         )
    #         ax_anim.add_patch(patch)
    #         airfoil_patches.append(patch)
            
    #         # Draw chord line if requested
    #         if properties.get('show_chord', True):
    #             chord_y0 = coords[:, 1].mean()
    #             ax_anim.plot(
    #                 [coords[:, 0].min(), coords[:, 0].max()],
    #                 [chord_y0, chord_y0],
    #                 'k--', lw=0.8
    #             )
            
    #         # Set view limits
    #         span = 1.5 * self.b
    #         ax_anim.set_xlim(-span + self.a, span + self.a)
    #         ax_anim.set_ylim(-span, span)
    #         ax_anim.set_aspect('equal')
    #         ax_anim.set_title(f"Mode {i+1} Animation")
    #         ax_anim.grid(True, alpha=0.3)
            
    #         # Add elastic axis marker
    #         ax_anim.plot([self.a], [0], 'ro', markersize=4)
            
    #         # Mode parameters text
    #         mode_text = (f"ω = {abs(imag_parts[i]):.2f}, "
    #                     f"γ = {real_parts[i]:.2f}")
    #         ax_anim.text(0.02, 0.98, mode_text, transform=ax_anim.transAxes,
    #                     va='top', ha='left', fontsize=10,
    #                     bbox=dict(facecolor='white', alpha=0.7))
            
    #         # Displacement history subplot (right top)
    #         ax_disp = fig.add_subplot(gs[i, 1])
    #         ax_disp.plot(t, h_t[i], 'b-', label="Plunge")
    #         ax_disp.plot(t, theta_t[i], 'r--', label="Twist")
    #         ax_disp.set_title(f"Mode {i+1} Displacement")
    #         ax_disp.set_xlabel("Time [s]")
    #         ax_disp.set_ylabel("Displacement")
    #         ax_disp.legend(fontsize=8)
    #         ax_disp.grid(True)
            
    #         # Add time indicator for displacement plot
    #         time_line = ax_disp.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    #         time_lines.append(time_line)
            
    #         # Phase portrait subplot (right bottom)
    #         ax_phase = fig.add_subplot(gs[i, 2])
    #         ax_phase.plot(h_t[i], theta_t[i], 'g-')
    #         ax_phase.set_title(f"Mode {i+1} Phase Portrait")
    #         ax_phase.set_xlabel("Plunge (h)")
    #         ax_phase.set_ylabel("Twist (θ)")
    #         ax_phase.grid(True)
            
    #         # Add current point for phase plot
    #         phase_point, = ax_phase.plot([], [], 'go', markersize=6)
    #         phase_points.append(phase_point)

    #     # Animation update function
    #     def update(frame):
    #         # Update progress bar
    #         pct = int((frame / len(t)) * 100)
    #         elapsed = frame / fps
    #         self.progress_bar.progress(
    #             pct,
    #             text=f"Time Elapsed: {elapsed:0.1f}s  (Rendering...)"
    #         )
            
    #         # List to store all updated artists
    #         artists = []
            
    #         # Update each mode's visuals
    #         for i in range(n_modes):
    #             # Create transformation for airfoil
    #             rotation = transforms.Affine2D().rotate_around(
    #                 self.a, 0.0, 
    #                 theta_t[i, frame]
    #             )
    #             translation = rotation + transforms.Affine2D().translate(0, h_t[i, frame])
                
    #             # Apply transformation to airfoil patch
    #             airfoil_patches[i].set_transform(translation + fig.gca().transData)
    #             artists.append(airfoil_patches[i])
                
    #             # Update time indicator line
    #             time_lines[i].set_xdata([t[frame], t[frame]])
    #             artists.append(time_lines[i])
                
    #             # Update phase point
    #             phase_points[i].set_data([h_t[i, frame]], [theta_t[i, frame]])
    #             artists.append(phase_points[i])
            
    #         return artists

    #     # Create animation with explicit artist specification
    #     ani = FuncAnimation(
    #         fig,
    #         update,
    #         frames=len(t),
    #         blit=True,
    #         interval=1000 / fps
    #     )

    #     # Apply tight layout before animation
    #     plt.tight_layout()
    #     plt.subplots_adjust(top=0.92)  # Adjust for suptitle

    #     # Generate HTML and clean up
    #     plt.rcParams['animation.embed_limit'] = 2**128  # Set a very high embed limit
    #     anim_html = ani.to_jshtml()
    #     self.progress_bar.empty()
    #     plt.close(fig)  # Prevent double display

    #     return anim_html


    def animate_flutter(self,
                        airfoil_coords,
                        duration=10,
                        fps=30,
                        scale=1.0,
                        n_modes=4,
                        properties=None):
        # … your docstring & default‐props omitted for brevity …

        # 1) Prepare data & eigen‐stuff
        if properties is None:
            properties = {
                'airfoil_color': '#ffffff',
                'transparency': 50,
                'show_chord': True
            }
        if getattr(self, 'vals', None) is None:
            self.compute_response()

        lambda_vals = self.vals[:n_modes]
        real_parts = np.real(lambda_vals)
        imag_parts = np.imag(lambda_vals)

        h_tidals = np.real(self.vecs[0, :n_modes]) * self.b
        theta_tidals = np.real(self.vecs[1, :n_modes])
        phase_diffs = np.angle(self.vecs[1, :n_modes] / self.vecs[0, :n_modes])

        t = np.linspace(0, duration, int(duration * fps))

        # 2) Center & scale airfoil coords so chord‐midpoint sits at x=self.a
        coords = np.array(airfoil_coords) * scale
        mid = 0.5 * (coords[:,0].max() + coords[:,0].min())
        coords[:,0] = coords[:,0] - mid + self.a

        # 3) Precompute time‐histories
        h_t = np.array([ h_tidals[i] * np.exp(real_parts[i]*t) *
                        np.cos(imag_parts[i]*t) for i in range(n_modes) ])
        theta_t = np.array([ theta_tidals[i] * np.exp(real_parts[i]*t) *
                            np.cos(imag_parts[i]*t + phase_diffs[i])
                            for i in range(n_modes) ])

        # 4) Set up subplots
        fig, axes = plt.subplots(
            nrows=n_modes, ncols=3,
            figsize=(12, 3*n_modes),
            gridspec_kw={'width_ratios':[2,1,1]}
        )
        fig.suptitle("Coupled Flutter Modes – Time Response", fontsize=14)

        # Keep references to all “dynamic” artists & their axes
        airfoil_patches = []
        anim_axes = []
        time_lines = []
        phase_points = []

        span = 1.5 * self.b

        for i in range(n_modes):
            # — Animation panel —
            axA = axes[i,0]
            patch = Polygon(coords, closed=True,
                            facecolor=properties['airfoil_color'],
                            edgecolor='k',
                            alpha=properties['transparency']/100.)
            axA.add_patch(patch)
            if properties['show_chord']:
                chord_y = coords[:,1].mean()
                axA.plot([coords[:,0].min(), coords[:,0].max()],
                        [chord_y, chord_y], 'k--', lw=0.7)

            axA.set_xlim(self.a - span, self.a + span)
            axA.set_ylim(-span, span)
            axA.set_aspect('equal')
            axA.set_title(f"Mode {i+1} Animation")
            axA.grid(alpha=0.3)

            airfoil_patches.append(patch)
            anim_axes.append(axA)

            # — Displacement vs time —
            axD = axes[i,1]
            axD.plot(t, h_t[i], 'b-', label='Plunge')
            axD.plot(t, theta_t[i], 'r--', label='Twist')
            axD.set_title(f"Mode {i+1} Displacement")
            axD.set_xlabel("Time [s]")
            axD.legend(fontsize=8)
            axD.grid(alpha=0.3)
            line = axD.axvline(0, color='k', alpha=0.3)
            time_lines.append(line)

            # — Phase portrait —
            axP = axes[i,2]
            axP.plot(h_t[i], theta_t[i], 'g-')
            axP.set_title(f"Mode {i+1} Phase")
            axP.set_xlabel("h")
            axP.set_ylabel("θ")
            axP.grid(alpha=0.3)
            pt, = axP.plot([], [], 'go', markersize=5)
            phase_points.append(pt)

        # 5) Update function
        def update(frame):
            artists = []
            for j in range(n_modes):
                ax = anim_axes[j]
                # Build rotation about (self.a, 0) then translation in y
                trans = (transforms.Affine2D()
                        .rotate_around(self.a, 0, theta_t[j,frame])
                        .translate(0, h_t[j,frame])
                        + ax.transData)
                airfoil_patches[j].set_transform(trans)
                artists.append(airfoil_patches[j])

                # move the time-line
                time_lines[j].set_xdata([t[frame], t[frame]])
                artists.append(time_lines[j])

                # move the phase point
                phase_points[j].set_data([h_t[j,frame]], [theta_t[j,frame]])
                artists.append(phase_points[j])

            return artists

        # 6) Animate
        ani = FuncAnimation(
            fig, update, frames=len(t),
            interval=1000/fps, blit=True
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # 7) Streamlit progress + export
        bar = st.progress(0, text="Rendering Animation...")
        self.progress_bar = bar

        plt.rcParams['animation.embed_limit'] = 2**128
        html = ani.to_jshtml()
        self.progress_bar.empty()
        plt.close(fig)

        return html
