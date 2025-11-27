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

from typing import Union, Dict

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
    
    def animate_flutter(
        self,
        airfoil_coords,
        duration: float = 6,
        fps: int = 30,
        scale: float = 1.0,
        n_modes: int = 1,
        properties: Dict = None,
        debug: bool = False,
    ):
        """
        One-axes animation of the airfoil (plunge + twist). No subplots.

        Returns
        -------
        str : HTML for Streamlit (ani.to_jshtml()).
        """
        # ---- guard / defaults ----
        n_modes = max(1, int(n_modes))
        if getattr(self, "vals", None) is None or getattr(self, "vecs", None) is None:
            self.compute_response()

        # Extract first n_modes safely
        n_avail = min(n_modes, len(self.vals))
        lam = self.vals[:n_avail]
        g = np.real(lam)             # damping
        w = np.imag(lam)             # frequency

        # State-space eigenvectors: assume state = [h, theta, hdot, thetadot]
        h_amp     = np.real(self.vecs[0, :n_avail]) * self.b
        theta_amp = np.real(self.vecs[1, :n_avail])
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.angle(self.vecs[1, :n_avail] / self.vecs[0, :n_avail])
        phi = np.nan_to_num(phi)

        # Time
        n_frames = max(2, int(duration * fps))
        t = np.linspace(0.0, duration, n_frames)

        # Histories = sum of selected modes
        h_t  = np.sum([h_amp[i]     * np.exp(g[i]*t) * np.cos(w[i]*t)              for i in range(n_avail)], axis=0)
        th_t = np.sum([theta_amp[i] * np.exp(g[i]*t) * np.cos(w[i]*t + phi[i])     for i in range(n_avail)], axis=0)

        # ---- base geometry, centered around elastic axis ----
        base = np.asarray(airfoil_coords, float) * scale           # (N,2)
        # Center by mid-chord then move so pivot is at (self.a, 0)
        xmid = 0.5*(base[:,0].max() + base[:,0].min())
        base[:,0] -= xmid
        base[:,0] += self.a

        # Relative to pivot for rotation
        x_rel0 = base[:,0] - self.a
        y_rel0 = base[:,1].copy()

        # ---- view box computed from geometry + motion ----
        x_min, x_max = base[:,0].min(), base[:,0].max()
        y_min, y_max = base[:,1].min(), base[:,1].max()
        chord = max(1e-12, x_max - x_min)               # chord length in current units
        geom_span = max(chord, (y_max - y_min))
        plunge_pad = float(np.nanmax(np.abs(h_t))) if np.isfinite(h_t).any() else 0.0
        span = 0.7*geom_span + plunge_pad               # zoom factor; tweak 0.7→0.5 for tighter view
        span = max(span, 0.1*geom_span, 1e-6)

        # ---- figure ----
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.set_title("Airfoil motion (plunge + twist)")

        # Center view around pivot
        ax.set_xlim(self.a - 1.2*span, self.a + 1.2*span)
        ax.set_ylim(-1.2*span, 1.2*span)

        # Force a visible style (ignore rcParams and any white-on-white surprises)
        patch = Polygon(
            base,
            closed=True,
            facecolor=properties["airfoil_color"],         # force bright fill
            edgecolor="k",
            linewidth=2.0,
            alpha=properties["transparency"],               # fully opaque to avoid background blending issues
            zorder=5,
        )
        ax.add_patch(patch)

        # Optional chord line + pivot marker
        if True:
            chord_y = base[:,1].mean()
            ax.plot([base[:,0].min(), base[:,0].max()], [chord_y, chord_y], "k--", lw=1.0, zorder=6)
        ax.plot(self.a, 0.0, "ko", ms=5, zorder=7)

        # Debug overlays: raw vertices and bounding box
        if debug:
            ax.scatter(base[:,0], base[:,1], s=8, c="blue", zorder=8)
            ax.add_patch(Polygon(
                np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]),
                fill=False, edgecolor="green", linestyle="--", linewidth=1.0, zorder=4
            ))

        # ---- Streamlit progress (coarse) ----
        bar = st.progress(0, text="Rendering animation…")

        # ---- update: rotate & translate vertices directly ----
        def update(frame):
            if frame % max(1, n_frames // 20) == 0:
                pct = int(100 * frame / (n_frames - 1))
                bar.progress(pct, text=f"Rendering animation… ({pct}%)")

            theta = th_t[frame]
            h = h_t[frame]
            c, s = np.cos(theta), np.sin(theta)

            x = c * x_rel0 - s * y_rel0 + self.a
            y = s * x_rel0 + c * y_rel0 + h

            patch.set_xy(np.column_stack((x, y)))
            return (patch,)

        ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
        plt.tight_layout()
        plt.rcParams["animation.embed_limit"] = 2**128

        html = ani.to_jshtml()
        bar.progress(100, text="Done.")
        bar.empty()
        plt.close(fig)
        return html

    def debug_static_foil(self, airfoil_coords, scale=1.0):
        base = np.asarray(airfoil_coords, float) * scale
        xmid = 0.5*(base[:,0].max()+base[:,0].min())
        base[:,0] -= xmid
        base[:,0] += self.a

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)
        patch = Polygon(base, closed=True, facecolor='red', edgecolor='k', lw=2)
        ax.add_patch(patch)
        ax.plot(self.a, 0, 'ko', ms=5)
        x_min, x_max = base[:,0].min(), base[:,0].max()
        y_min, y_max = base[:,1].min(), base[:,1].max()
        chord = max(1e-12, x_max-x_min)
        span = 0.7*max(chord, y_max-y_min)
        ax.set_xlim(self.a-1.2*span, self.a+1.2*span)
        ax.set_ylim(-1.2*span, 1.2*span)
        plt.show()

    
    # def animate_flutter(
    #     self,
    #     airfoil_coords,
    #     duration: float = 6,
    #     fps: int = 30,
    #     scale: float = 1.0,
    #     n_modes: int = 1,
    #     properties: Dict = None,
    # ):
    #     """
    #     Animate airfoil plunge + twist for the first n_modes.
    #     Layout: n_modes rows × 2 cols (left=airfoil motion, right=displacements with moving dots).
    #     Returns HTML string (ani.to_jshtml()) for Streamlit.
    #     """
    #     # --- visuals defaults ---
    #     properties = properties or {}
    #     airfoil_color = properties.get("airfoil_color", "#e53935")  # visible red
    #     transparency  = properties.get("transparency", 90) / 100.0
    #     show_chord    = properties.get("show_chord", True)

    #     # --- eigen stuff ---
    #     if getattr(self, "vals", None) is None or getattr(self, "vecs", None) is None:
    #         self.compute_response()

    #     n_modes = max(1, int(n_modes))
    #     n_avail = min(n_modes, len(self.vals))
    #     n_modes = n_avail

    #     lam   = self.vals[:n_modes]
    #     gamma = np.real(lam)   # damping
    #     omega = np.imag(lam)   # frequency (rad/s in your scaling)

    #     # assume state ordering = [h, theta, hdot, thetadot]
    #     h_amp     = np.real(self.vecs[0, :n_modes]) * self.b
    #     theta_amp = np.real(self.vecs[1, :n_modes])
    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         phase = np.angle(self.vecs[1, :n_modes] / self.vecs[0, :n_modes])
    #     phase = np.nan_to_num(phase)

    #     # --- time grid & histories (each mode separately) ---
    #     n_frames = max(2, int(duration * fps))
    #     t = np.linspace(0.0, duration, n_frames)

    #     h_t = np.array([
    #         h_amp[i]     * np.exp(gamma[i] * t) * np.cos(omega[i] * t)
    #         for i in range(n_modes)
    #     ])                 # shape (n_modes, n_frames)
    #     th_t = np.array([
    #         theta_amp[i] * np.exp(gamma[i] * t) * np.cos(omega[i] * t + phase[i])
    #         for i in range(n_modes)
    #     ])                 # shape (n_modes, n_frames)

    #     # --- base geometry, centered about elastic axis x=a ---
    #     base = np.asarray(airfoil_coords, float) * scale
    #     xmid = 0.5 * (base[:, 0].max() + base[:, 0].min())
    #     base[:, 0] -= xmid
    #     base[:, 0] += self.a

    #     # store relative coords for rotation
    #     x_rel0 = base[:, 0] - self.a
    #     y_rel0 = base[:, 1].copy()

    #     # --- compute sensible view box from geometry + max plunge amplitude ---
    #     x_min, x_max = base[:, 0].min(), base[:, 0].max()
    #     y_min, y_max = base[:, 1].min(), base[:, 1].max()
    #     chord_len = max(1e-9, x_max - x_min)
    #     geom_span = max(chord_len, (y_max - y_min))
    #     plunge_pad = float(np.nanmax(np.abs(h_t))) if np.isfinite(h_t).any() else 0.0
    #     span = max(0.6 * geom_span + plunge_pad, 1e-6)

    #     # --- figure ---
    #     fig, axes = plt.subplots(
    #         nrows=n_modes, ncols=2,
    #         figsize=(11, 3.6 * n_modes),
    #         gridspec_kw={"width_ratios": [2, 3]},
    #     )
    #     fig.suptitle("Coupled Flutter Modes – Airfoil Motion & Displacements", fontsize=14)

    #     # normalize axes shape for n_modes == 1
    #     if n_modes == 1:
    #         axes = np.array([axes])

    #     airfoil_patches = []
    #     disp_points_h   = []
    #     disp_points_th  = []
    #     time_lines      = []
    #     anim_axes       = []

    #     for i in range(n_modes):
    #         # LEFT: airfoil motion
    #         axA = axes[i, 0]
    #         patch = Polygon(
    #             base,
    #             closed=True,
    #             facecolor=airfoil_color,
    #             edgecolor="k",
    #             linewidth=1.5,
    #             alpha=transparency,
    #             zorder=5,
    #         )
    #         axA.add_patch(patch)
    #         if show_chord:
    #             chord_y = base[:, 1].mean()
    #             axA.plot([x_min, x_max], [chord_y, chord_y], "k--", lw=0.9, zorder=6)
    #         axA.plot(self.a, 0.0, "ko", ms=4, zorder=7)

    #         axA.set_aspect("equal", "box")
    #         axA.grid(alpha=0.25)
    #         axA.set_xlim(self.a - 1.2 * span, self.a + 1.2 * span)
    #         axA.set_ylim(-1.2 * span, 1.2 * span)
    #         axA.set_title(f"Mode {i+1}: Airfoil Motion")

    #         airfoil_patches.append(patch)
    #         anim_axes.append(axA)

    #         # RIGHT: displacement vs time with moving dots
    #         axD = axes[i, 1]
    #         axD.plot(t, h_t[i],  "b-", label="Plunge h")
    #         axD.plot(t, th_t[i], "r--", label="Twist θ")
    #         axD.set_xlabel("Time")
    #         axD.set_ylabel("Displacement")
    #         axD.grid(alpha=0.25)
    #         axD.legend(fontsize=8)
    #         axD.set_title(f"Mode {i+1}: Displacements")

    #         # moving dots + vertical time line
    #         pt_h,  = axD.plot([], [], "bo", ms=6)
    #         pt_th, = axD.plot([], [], "ro", ms=6, mfc="none")
    #         disp_points_h.append(pt_h)
    #         disp_points_th.append(pt_th)

    #         vline = axD.axvline(t[0], color="k", alpha=0.35)
    #         time_lines.append(vline)

    #     # --- Streamlit progress (coarse) ---
    #     bar = st.progress(0, text="Rendering animation…")

    #     # --- update (rotate/translate vertices directly; move dots/line) ---
    #     def update(frame):
    #         if frame % max(1, n_frames // 20) == 0:
    #             pct = int(100 * frame / (n_frames - 1))
    #             bar.progress(pct, text=f"Rendering animation… ({pct}%)")

    #         artists = []
    #         for i in range(n_modes):
    #             theta = th_t[i, frame]
    #             h = h_t[i, frame]
    #             c, s = np.cos(theta), np.sin(theta)

    #             x = c * x_rel0 - s * y_rel0 + self.a
    #             y = s * x_rel0 + c * y_rel0 + h
    #             airfoil_patches[i].set_xy(np.column_stack((x, y)))
    #             artists.append(airfoil_patches[i])

    #             # move time cursor + dots
    #             time_lines[i].set_xdata([t[frame], t[frame]])
    #             disp_points_h[i].set_data([t[frame]], [h_t[i, frame]])
    #             disp_points_th[i].set_data([t[frame]], [th_t[i, frame]])
    #             artists.extend([time_lines[i], disp_points_h[i], disp_points_th[i]])

    #         return artists

    #     ani = plt.matplotlib.animation.FuncAnimation(
    #         fig, update, frames=n_frames, interval=1000 / fps, blit=True
    #     )
    #     plt.tight_layout()
    #     plt.subplots_adjust(top=0.90)
    #     plt.rcParams["animation.embed_limit"] = 2**128

    #     html = ani.to_jshtml()
    #     bar.progress(100, text="Done.")
    #     bar.empty()
    #     plt.close(fig)
    #     return html
