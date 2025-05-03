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

    def animate_flutter(self, airfoil_coords, duration=10, fps=30, properties=None):
        """
        Animate the flutter response showing airfoil motion.
        """
        if properties is None:
            properties = {
                'airfoil_color': '#ffffff',
                'transparency': 50,
                'show_chord': True
            }

        anim_bar = st.progress(0, text="Rendering Animation...")
        self.progress_bar = anim_bar

        if self.vals is None:
            self.compute_response()

        # Extract eigenvalues and eigenvectors
        lambda_vals = self.vals[:4]
        real_parts = np.real(lambda_vals)
        imag_parts = np.imag(lambda_vals)
        
        h_tidals = np.real(self.vecs[0, :4]) * self.b
        theta_tidals = np.real(self.vecs[1, :4])
        phase_diffs = np.angle(self.vecs[1, :4] / self.vecs[0, :4])

        t = np.linspace(0, duration, duration * fps)

        # Setup figure
        fig, axes = plt.subplots(4, 2, figsize=(8, 8))
        fig.suptitle("Coupled Flutter Modes - Time Response", fontsize=14)

        # Create airfoil patches
        airfoil_patches = []
        for i in range(4):
            patch = Polygon(airfoil_coords, closed=True, edgecolor='k', 
                          facecolor=properties['airfoil_color'], 
                          alpha=properties['transparency']/100)
            axes[i, 0].add_patch(patch)
            axes[i, 0].set_xlim(-1.5 * self.b, 1.5 * self.b)
            axes[i, 0].set_ylim(-1.5 * self.b, 1.5 * self.b)
            axes[i, 0].set_aspect('equal')
            axes[i, 0].set_title(f"Mode {i+1} Animation")
            airfoil_patches.append(patch)

        # Compute displacements
        h_t = np.array([
            h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t)
            for i in range(4)
        ])

        theta_t = np.array([
            theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i])
            for i in range(4)
        ])

        # Plot displacement histories
        for i in range(4):
            axes[i, 1].plot(t, h_t[i], 'b-', label=f"Mode {i+1} Plunge")
            axes[i, 1].plot(t, theta_t[i], 'r--', label=f"Mode {i+1} Twist")
            axes[i, 1].set_xlabel("Vibration Period")
            axes[i, 1].set_ylabel("Displacement")
            axes[i, 1].legend()
            axes[i, 1].grid()
            axes[i, 1].set_title(f"Mode {i+1} Amplitude & Phase")

        def update(frame):
            progress = int((frame / len(t)) * 100)
            self.progress_bar.progress(progress, text=f"Time Elapsed: {int(frame/fps)}s\nRendering Animation...")
            
            for i in range(4):
                trans = transforms.Affine2D().rotate_deg_around(
                    self.a, 0, np.degrees(theta_t[i, frame])
                ).translate(0, h_t[i, frame]) + axes[i, 0].transData
                airfoil_patches[i].set_transform(trans)
            
            return airfoil_patches

        ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000/fps)
        anim = ani.to_jshtml()
        self.progress_bar.empty()
        
        return anim
