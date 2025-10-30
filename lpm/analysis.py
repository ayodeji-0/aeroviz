import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation



class BliskAnalysis:
    """
    Class for analyzing the flutter characteristics of a blisk.
    """
    def __init__(self, blisk_obj, time, intervals):
        self.blisk_obj = blisk_obj
        self.mu = blisk_obj.mu
        self.kappa = blisk_obj.kappa
        self.num_blades = blisk_obj.num_blades
        self.time = time
        self.interval = intervals
        self.eigenvalues = None
        self.eigenvectors = None
        self.natural_frequencies = None

    # def compute_deformations(self):
    #     """
    #     Compute harmonic deformations of a bladed disk system by solving the eigenproblem.
        
    #     Returns:
    #         A list of deformations for each time step: [t, disk_def, blade_def]
    #         where disk_def and blade_def are arrays of length N representing deformations at each nodal diameter.
    #     """
    #     # Define parameters
    #     nodal_diameters = np.arange(-self.num_blades // 2, self.num_blades // 2 + 1)  # Nodal diameters
    #     print("Nodal Diameters: ", nodal_diameters)
    #     delta_theta = 2 * np.pi / self.num_blades  # Angular difference between blades
    #     time = np.linspace(0, self.time, self.interval)
        
    #     # Precompute thetas
    #     thetas = np.linspace(0, 2 * np.pi, self.num_blades, endpoint=False)

    #     # Initialize output
    #     self.results = []

    #     # Precompute eigenvalues and eigenvectors for each nodal diameter
    #     eigen_data = []

    #     self.natural_frequencies = []
    #     self.eigenvectors = []
    #     self.eigenvalues = []
        
    #     for nodal_diameter in nodal_diameters:
    #         sin_theta = np.sin(nodal_diameter * delta_theta / 2) ** 2

    #         # Stiffness matrix K
    #         K = np.array([
    #             [1 + 4 * self.kappa * sin_theta, -1],
    #             [-1, 1]
    #         ])

    #         # Mass matrix M
    #         M = np.array([
    #             [self.mu, 0],
    #             [0, 1]
    #         ])

    #         # Solve the eigenvalue problem using numpy's eig
    #         w_tidal_squared, xy_hat = np.linalg.eig(np.linalg.inv(M) @ K)
    #         print("Eigenvalues: ", w_tidal_squared)
    #         print("Eigenvectors: ", xy_hat)
    #         frequencies = np.sqrt(np.real(w_tidal_squared))  # Only take the real part
    #         self.natural_frequencies.append(frequencies)
    #         self.eigenvalues.append(w_tidal_squared)
    #         self.eigenvectors.append(xy_hat)
    #         eigen_data.append((nodal_diameter, frequencies, xy_hat))

    #     # Calculate deformations for each time step
    #     for t in time:
    #         disk_def = np.zeros(self.num_blades)
    #         blade_def = np.zeros(self.num_blades)

    #         for idx, theta in enumerate(thetas):
    #             for nodal_diameter, frequencies, xy_hat in eigen_data:
    #                 omega_t = frequencies * t

    #                 # Extract disk and blade components
    #                 x_hat = xy_hat[0, :]  # Disk component
    #                 y_hat = xy_hat[1, :]  # Blade component

    #                 # Compute deformations using harmonic solutions
    #                 disk_contrib = np.real(x_hat * np.exp(1j * (nodal_diameter * theta + omega_t)))
    #                 blade_contrib = np.real(y_hat * np.exp(1j * (nodal_diameter * theta + omega_t)))

    #                 disk_def[idx] += np.sum(disk_contrib)
    #                 blade_def[idx] += np.sum(blade_contrib)

    #         # Append the result as [t, disk_def, blade_def]
    #         self.results.append([t, disk_def, blade_def])

    #         self.disk_def = [[disp[0],disp[1]] for disp in self.results]
    #         self.blade_def = [[disp[0],disp[2]] for disp in self.results]
    #     return self.results 
    def compute_deformations(self):
        """
        Compute harmonic deformations of a bladed disk system by solving the eigenproblem.
        Also extracts mode vs frequency data for validation against ANSYS.
        
        Returns:
            - self.results: List of deformation results for time-based analysis.
            - self.mode_frequencies_df: pd.DataFrame storing mode vs frequency per harmonic.
        """
    

        # Define parameters
        nodal_diameters = np.arange(0, self.num_blades // 2 + 1)  # Only positive ND (0 to N/2)
        delta_theta = 2 * np.pi / self.num_blades  # Angular difference between blades
        thetas = np.linspace(0, 2 * np.pi, self.num_blades, endpoint=False)

        # Initialize storage
        self.results = []
        self.natural_frequencies = []
        self.eigenvectors = []
        self.eigenvalues = []

        mode_data = []  # For mode vs frequency comparison

        # Solve eigenproblem per nodal diameter
        for mode_number, nodal_diameter in enumerate(nodal_diameters):
            sin_theta = np.sin(nodal_diameter * delta_theta / 2) ** 2

            # Stiffness matrix K
            K = np.array([
                [1 + 4 * self.kappa * sin_theta, -1],
                [-1, 1]
            ])

            # Mass matrix M
            M = np.array([
                [self.mu, 0],
                [0, 1]
            ])

            # Solve eigenproblem
            w_tidal_squared, xy_hat = np.linalg.eig(np.linalg.inv(M) @ K)

            # Extract positive real frequencies
            frequencies = np.sqrt(np.real(w_tidal_squared))
            self.natural_frequencies.append(frequencies)
            self.eigenvalues.append(w_tidal_squared)
            self.eigenvectors.append(xy_hat)

            # Store mode vs physical natural frequency
            # first get blade natural frequencies
            w_blade = np.sqrt(self.blisk_obj.k_blade / self.blisk_obj.M_blade)
            for f in frequencies:
                mode_data.append([mode_number, nodal_diameter, (w_blade* f)/ (2 * np.pi)])  # Mode index starts at 0

        # Create DataFrame (Mode vs Frequency at each Harmonic Index)
        self.mode_frequencies_df = pd.DataFrame(mode_data, columns=["Mode", "Harmonic Index", "Frequency [Hz]"])

        # ------------------------------------
        # Generate Deformations for Animation
        # ------------------------------------
        time = np.linspace(0, self.time, self.interval)
        eigen_data = list(zip(nodal_diameters, self.natural_frequencies, self.eigenvectors))

        self.results = []  # Initialize results list

        for t in time:
            disk_def = np.zeros(self.num_blades)
            blade_def = np.zeros(self.num_blades)

            for idx, theta in enumerate(thetas):
                for nodal_diameter, frequencies, xy_hat in eigen_data:
                    omega_t = frequencies * t

                    x_hat = xy_hat[0, :]
                    y_hat = xy_hat[1, :]

                    disk_contrib = np.real(x_hat * np.exp(1j * (nodal_diameter * theta + omega_t)))
                    blade_contrib = np.real(y_hat * np.exp(1j * (nodal_diameter * theta + omega_t)))

                    disk_def[idx] += np.sum(disk_contrib)
                    blade_def[idx] += np.sum(blade_contrib)
                    
            # Append the result as [t, disk_def, blade_def]
            self.results.append([t, disk_def, blade_def])

            self.disk_def = [[disp[0],disp[1]] for disp in self.results]
            self.blade_def = [[disp[0],disp[2]] for disp in self.results]

        return self.results, self.mode_frequencies_df

    def plot_deformations(self):
        """Plot the deformations of the disk and blades."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([disp[0] for disp in self.disk_def], [disp[1] for disp in self.disk_def], label='Disk')
        ax.plot([disp[0] for disp in self.blade_def], [disp[1] for disp in self.blade_def], label='Blades')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Deformation')
        ax.set_title('Disk and Blade Deformations')
        ax.legend()
        return fig

    # def plot_mode_shapes(self, modes=[0]):
    #     """Plot the mode shapes of the disk and blades."""
    #     # Extract mode shape data
    #     disk_mode = []
    #     blade_mode = []
    #     for mode in modes:
    #         disk_mode += self.eigenvectors[0, :, mode]
    #         blade_mode += self.eigenvectors[1, :, mode]

    #     # Plot mode shapes
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.plot(disk_mode, label='Disk')
    #     ax.plot(blade_mode, label='Blades')
    #     ax.set_xlabel('Nodal Diameter')
    #     ax.set_ylabel('Mode Shape')
    #     ax.set_title(f'Mode {mode + 1} Shapes')
    #     ax.legend()
    #     return fig
    
    def animate_deformations(self):
        """Animate the deformations of the disk and blades."""

        # Set Up Animation
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.coolwarm

        # Discretize the geometry for plotting
        num_radial = 5
        radial_positions = np.linspace(0, self.blisk_obj.disk_radius, num_radial)

        disk_patches = []
        for i in range(num_radial - 1):
            r_inner = radial_positions[i]
            r_outer = radial_positions[i + 1]
            for j in range(self.num_blades):
                theta1 = j * 2.0 * np.pi / self.num_blades
                theta2 = (j + 1) * 2.0 * np.pi / self.num_blades
                x1, y1 = r_inner * np.cos(theta1), r_inner * np.sin(theta1)
                x2, y2 = r_inner * np.cos(theta2), r_inner * np.sin(theta2)
                x3, y3 = r_outer * np.cos(theta2), r_outer * np.sin(theta2)
                x4, y4 = r_outer * np.cos(theta1), r_outer * np.sin(theta1)

                poly = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], facecolor=cmap(0.0), alpha=0.6)
                ax.add_patch(poly)
                disk_patches.append((poly, j))

        num_blade_segments = 5
        blade_patches = []
        for j in range(self.num_blades):
            angle_j = (j + 0.5) * 2.0 * np.pi / self.num_blades
            for seg in range(num_blade_segments):
                frac1 = seg / num_blade_segments
                frac2 = (seg + 1) / num_blade_segments
                r1 = self.blisk_obj.disk_radius + frac1 * self.blisk_obj.blade_length
                r2 = self.blisk_obj.disk_radius + frac2 * self.blisk_obj.blade_length

                x1 = r1 * np.cos(angle_j) - (self.blisk_obj.blade_width / 2) * np.sin(angle_j)
                y1 = r1 * np.sin(angle_j) + (self.blisk_obj.blade_width / 2) * np.cos(angle_j)
                x2 = r1 * np.cos(angle_j) + (self.blisk_obj.blade_width / 2) * np.sin(angle_j)
                y2 = r1 * np.sin(angle_j) - (self.blisk_obj.blade_width / 2) * np.cos(angle_j)
                x3 = r2 * np.cos(angle_j) + (self.blisk_obj.blade_width / 2) * np.sin(angle_j)
                y3 = r2 * np.sin(angle_j) - (self.blisk_obj.blade_width / 2) * np.cos(angle_j)
                x4 = r2 * np.cos(angle_j) - (self.blisk_obj.blade_width / 2) * np.sin(angle_j)
                y4 = r2 * np.sin(angle_j) + (self.blisk_obj.blade_width / 2) * np.cos(angle_j)

                poly = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], facecolor=cmap(0.0), alpha=0.6)
                ax.add_patch(poly)
                blade_patches.append((poly, j))

        ax.set_aspect('equal', 'box')
        margin = self.blisk_obj.disk_radius + self.blisk_obj.blade_length + 0.02
        ax.set_xlim([-margin, margin])
        ax.set_ylim([-margin, margin])
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label='Normalized Deformation')

        total_frames = len(self.results)

        def update(frame_index):
            t, disk_def, blade_def = self.results[frame_index]

            dmin, dmax = disk_def.min(), disk_def.max()
            bmin, bmax = blade_def.min(), blade_def.max()
            eps = 1e-12

            for (patch, j) in disk_patches:
                val = disk_def[j]
                normed = (val - dmin) / (dmax - dmin + eps)
                patch.set_facecolor(cmap(normed))

            for (patch, j) in blade_patches:
                val = blade_def[j]
                normed = (val - bmin) / (bmax - bmin + eps)
                patch.set_facecolor(cmap(normed))

            ax.set_title(f"Time = {t:.3f}s")
            return [p[0] for p in disk_patches] + [p[0] for p in blade_patches]

        ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)
        anim = ani.to_jshtml()

        #anim_bar.empty()
        self.progress = 0
        return ani

        # # Add progress bar
        # for _ in tqdm(anim, total=total_frames):
        #     pass

        # plt.show()
        # # If you want to save the animation, e.g. as an MP4 or GIF:
        # # anim.save("deformations.mp4", fps=10)