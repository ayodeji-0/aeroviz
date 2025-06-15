import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
import scipy.linalg as la
import streamlit as st
from IPython.display import HTML

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
    
def animate_flutter(self, airfoil_coords, duration=10, fps=30, scale=1.0, properties=None, n_modes=1):
    """
    Animate the flutter response showing airfoil motion, displacement history, and phase lag.

    Parameters:
        airfoil_coords: (N, 2) array of airfoil geometry
        duration: total animation duration in seconds
        fps: frames per second
        scale: scaling factor for airfoil size
        properties: dict for visual appearance
        n_modes: number of modes to animate
    """

    if properties is None:
        properties = {
            'airfoil_color': '#ffffff',
            'transparency': 50,
            'show_chord': True
        }

    if getattr(self, 'vals', None) is None or getattr(self, 'vecs', None) is None:
        self.compute_response()

    # Time vector
    t = np.linspace(0, duration, int(duration * fps))

    # Get eigen-data for requested modes
    lambda_vals = self.vals[:n_modes]
    h_tidals = np.real(self.vecs[0, :n_modes]) * self.b
    theta_tidals = np.real(self.vecs[1, :n_modes])
    phase_diffs = np.angle(self.vecs[1, :n_modes] / self.vecs[0, :n_modes])
    real_parts = np.real(lambda_vals)
    imag_parts = np.imag(lambda_vals)

    # Compute time histories
    h_t = np.array([h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t) for i in range(n_modes)])
    theta_t = np.array([theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i]) for i in range(n_modes)])

    # Prepare airfoil coords (centered and offset)
    coords = np.array(airfoil_coords) * scale
    x_coords = coords[:, 0]
    mid_chord = 0.5 * (x_coords.max() + x_coords.min())
    coords[:, 0] -= mid_chord
    coords[:, 0] += self.a

    # Set up 2-column layout (left = airfoil, right = stacked plots)
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax_airfoil = fig.add_subplot(gs[:, 0])
    ax_disp = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 1])

    fig.suptitle("Flutter Mode Animation", fontsize=14)

    # Create patch
    patch = Polygon(coords, closed=True, edgecolor='k',
                    facecolor=properties['airfoil_color'],
                    alpha=properties['transparency'] / 100.0)
    ax_airfoil.add_patch(patch)

    if properties.get('show_chord', False):
        chord_y0 = coords[:, 1].mean()
        ax_airfoil.plot([coords[:, 0].min(), coords[:, 0].max()], [chord_y0, chord_y0], 'k--', lw=0.8)

    ax_airfoil.set_xlim(-1.5 * self.b + self.a, 1.5 * self.b + self.a)
    ax_airfoil.set_ylim(-1.5 * self.b, 1.5 * self.b)
    ax_airfoil.set_aspect('equal')
    ax_airfoil.set_title("Airfoil Motion")

    # Elastic axis marker
    axis_marker, = ax_airfoil.plot(-self.a, 0, 'ro', markersize=5)

    # Displacement history
    disp_lines = []
    for i in range(n_modes):
        line_h, = ax_disp.plot([], [], label=f"Mode {i+1} Plunge", linestyle='-')
        line_theta, = ax_disp.plot([], [], label=f"Mode {i+1} Twist", linestyle='--')
        disp_lines.append((line_h, line_theta))

    ax_disp.set_xlim(0, duration)
    ymax = 1.1 * max(np.max(np.abs(h_t)), np.max(np.abs(theta_t)))
    ax_disp.set_ylim(-ymax, ymax)
    ax_disp.set_title("Displacement vs Time")
    ax_disp.set_xlabel("Time [s]")
    ax_disp.set_ylabel("Displacement")
    ax_disp.grid(True)
    ax_disp.legend()

    # Phase portrait
    phase_lines = []
    phase_dots = []
    for i in range(n_modes):
        line, = ax_phase.plot([], [], label=f"Mode {i+1}")
        dot, = ax_phase.plot([], [], 'o')
        phase_lines.append(line)
        phase_dots.append(dot)

    ax_phase.set_xlim(np.min(h_t) * 1.1, np.max(h_t) * 1.1)
    ax_phase.set_ylim(np.min(theta_t) * 1.1, np.max(theta_t) * 1.1)
    ax_phase.set_xlabel("Plunge (h)")
    ax_phase.set_ylabel("Twist (θ)")
    ax_phase.set_title("Phase Portrait")
    ax_phase.grid(True)
    ax_phase.legend()

    # Progress bar
    if hasattr(st, 'progress'):
        anim_bar = st.progress(0, text="Rendering Animation...")
    else:
        anim_bar = None

    # Update function
    def update(frame):
        if anim_bar:
            pct = int((frame / len(t)) * 100)
            anim_bar.progress(pct, text=f"Time Elapsed: {frame / fps:.1f}s (Rendering...)")

        trans = transforms.Affine2D().rotate_around(
            self.a, 0.0, theta_t[0, frame]
        ).translate(0, h_t[0, frame]) + ax_airfoil.transData
        patch.set_transform(trans)

        angle_rad = theta_t[0, frame]
        rotated_x = -self.a * np.cos(angle_rad)
        rotated_y = -self.a * np.sin(angle_rad) + h_t[0, frame]
        axis_marker.set_data([rotated_x], [rotated_y])

        for i in range(n_modes):
            disp_lines[i][0].set_data(t[:frame+1], h_t[i, :frame+1])
            disp_lines[i][1].set_data(t[:frame+1], theta_t[i, :frame+1])
            phase_lines[i].set_data(h_t[i, :frame+1], theta_t[i, :frame+1])
            phase_dots[i].set_data(h_t[i, frame], theta_t[i, frame])

        return [patch, axis_marker] + [line for pair in disp_lines for line in pair] + phase_lines + phase_dots

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000 / fps)

    if anim_bar:
        anim_bar.empty()
    plt.tight_layout()
    return ani.to_jshtml()