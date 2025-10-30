from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from numpy.typing import NDArray
import streamlit as st

from utils.blisks.constants import materials

class Blisk:
    """
    Class to represent a Blisk (Blade Integrated Disk) for aeroelastic analysis.
    """
    def __init__(self, material='Mild Steel', blade_thickness=0.1,disk_thickness=0.1, num_blades=20, blade_length=0.1, blade_width = 0.0125, disk_radius=0.5, blade_segments=10, radial_segments=10):
        self.num_blades = num_blades
        self.blade_length = blade_length
        self.blade_width = blade_width # radial width of the blade - d
        self.disk_radius = disk_radius

        self.blade_segments = blade_segments
        self.radial_segments = radial_segments

        self.blade_thickness = blade_thickness # into page - b
        self.disk_thickness = disk_thickness # into page
        self.density = materials[material]['density']
        self.E = materials[material]['E']
        self.G = materials[material]['G']
        self.v = materials[material]['v']


    def precompute_parameters(self):
        # Masses
        self.M_disk = np.pi * self.disk_radius**2 * self.disk_thickness * self.density
        self.M_blade = self.blade_length * self.blade_width * self.blade_thickness * self.density
        # Second moments of area
        self.I_disk = 0.5 * self.M_disk * self.disk_radius**2
        self.I_blade = (self.blade_thickness * self.blade_width**3) / 12 # rectangular cross-section, bd^3/12
        # Stiffnesses
        self.k_blade = 3 * self.E * self.I_blade / self.blade_length**3 # Built-in edge
        thresh = self.disk_thickness/self.disk_radius
        if thresh <= 0.1:  # Thin disk
            self.k_disk = 2 * self.E * self.disk_thickness**3 / (3 * (1 - self.v**2) * self.disk_radius**3)
        else:  # Thick disk
            self.k_disk = self.E * np.pi * self.disk_radius**2 / self.disk_thickness

        # Coefficients
        self.kappa = self.k_disk/self.k_blade
        self.mu = self.M_disk/self.M_blade

    def discretize(self):
        self.blade_angles = np.linspace(0, 2*np.pi, self.num_blades, endpoint=False)

        # Discretize the disk
        self.disk_patches = []
        for i in range(self.radial_segments-1):
            for j in range(self.num_blades):
                #Define vertices for each patch (quadrilateral)
                theta1 = self.blade_angles[j] + np.pi/4
                theta2 = self.blade_angles[(j + 1) % self.num_blades] + np.pi/4
                r1, r2 = i / self.radial_segments * self.disk_radius, (i + 1) / self.radial_segments * self.disk_radius

                x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
                x2, y2 = r1 * np.cos(theta2), r1 * np.sin(theta2)
                x3, y3 = r2 * np.cos(theta2), r2 * np.sin(theta2)
                x4, y4 = r2 * np.cos(theta1), r2 * np.sin(theta1)

                # Create patch
                patch = plt.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], color='blue', alpha=0.6)
                self.disk_patches.append(patch)
        
        self.blade_patches = []
        for angle in self.blade_angles:
            for segment in range(self.blade_segments):
                # Compute the start and end positions of the segment
                segment_start = segment * (self.blade_length / self.blade_segments)
                segment_end = (segment + 1) * (self.blade_length / self.blade_segments)
                
                # Compute the blade's center position for the segment
                blade_center_x_start = (self.disk_radius + segment_start) * np.cos(angle)
                blade_center_y_start = (self.disk_radius + segment_start) * np.sin(angle)
                blade_center_x_end = (self.disk_radius + segment_end) * np.cos(angle)
                blade_center_y_end = (self.disk_radius + segment_end) * np.sin(angle)
                
                # Define vertices for each patch (quadrilateral)
                x1, y1 = blade_center_x_start - (self.blade_width / 2) * np.sin(angle), blade_center_y_start + (self.blade_width / 2) * np.cos(angle)
                x2, y2 = blade_center_x_start + (self.blade_width / 2) * np.sin(angle), blade_center_y_start - (self.blade_width / 2) * np.cos(angle)
                x3, y3 = blade_center_x_end + (self.blade_width / 2) * np.sin(angle), blade_center_y_end - (self.blade_width / 2) * np.cos(angle)
                x4, y4 = blade_center_x_end - (self.blade_width / 2) * np.sin(angle), blade_center_y_end + (self.blade_width / 2) * np.cos(angle)
                
                # Create patch
                patch = plt.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], color='red', alpha=0.6)
                self.blade_patches.append(patch)


    def plot(self, color = 'black'):
        """Plot the Blisk structure with blades."""
        self.discretize()
        fig, ax = plt.subplots(figsize=(6, 6))
        for patch in self.disk_patches:
            ax.add_patch(patch)
        for patch in self.blade_patches:
            ax.add_patch(patch)
        
        # Aesthetics
        ax.set_xlim(-self.disk_radius - self.blade_length, self.disk_radius + self.blade_length)
        ax.set_ylim(-self.disk_radius - self.blade_length, self.disk_radius + self.blade_length)
        ax.set_aspect('equal', 'box')
        ax.set_facecolor('#0e1117')
        ax.axis('off')

        return fig