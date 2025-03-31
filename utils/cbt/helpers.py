from typing import Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray

def archimedes_spiral(num_points: int, c: float, spiral_length: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Function to create a torsional spring element using the archimedean spiral.
    
    Parameters:
    num_points (int): Number of points to generate for the spring element.
    c (float): constant for the spiral equation.
    spiral_length (float): Length of the spiral element.

    Returns:
    x_coords: x-coordinates of the spring element in cartesian coordinates.
    y_coords: y-coordinates of the spring element in cartesian coordinates.

    Raises:
        ValueError: If input parameters are invalid
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if c <= 0:
        raise ValueError("c must be positive")
    if spiral_length <= 0:
        raise ValueError("spiral_length must be positive")

    theta = np.linspace(0, spiral_length * np.pi, num_points)
    r = c * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def linear_spring(num_points: int, length: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Function to create a linear spring element using a sine wave.
    
    Parameters:
    num_points (int): Number of points to generate for the spring element.
    length (float): Length of the spring element.

    Returns:
    x_coords: x-coordinates of the spring element in cartesian coordinates.
    y_coords: y-coordinates of the spring element in cartesian coordinates.

    Raises:
        ValueError: If input parameters are invalid
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if length <= 0:
        raise ValueError("length must be positive")

    x = np.linspace(0, length, num_points)
    y = np.sin(5 * x)
    return x, y

def linear_spring2(num_points: int, length: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Function to create a linear spring element using a triangular wave.
    
    Parameters:
    num_points (int): Number of points to generate for the spring element.
    length (float): Length of the spring element.

    Returns:
    x_coords: x-coordinates of the spring element in cartesian coordinates.
    y_coords: y-coordinates of the spring element in cartesian coordinates.

    Raises:
        ValueError: If input parameters are invalid
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if length <= 0:
        raise ValueError("length must be positive")

    x = np.linspace(0, length, num_points)
    period = length / 5
    amplitude = 0.1 * length
    fractional_position = (x % period) / period
    y = 2 * amplitude * np.abs(fractional_position - 0.5)
    return x, y

def scale_spring(x: NDArray[np.float64], y: NDArray[np.float64], scale_factor: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Scale spring coordinates by a given factor.
    
    Parameters:
    x (array): x coordinates
    y (array): y coordinates
    scale_factor (float): scaling factor
    
    Returns:
    tuple: scaled x and y coordinates

    Raises:
        ValueError: If input arrays have different lengths or scale_factor is invalid
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    scaling_matrix = np.array([[scale_factor, 0],
                             [0, 1]])
    points = np.vstack((x, y))
    transformed_points = scaling_matrix @ points
    return transformed_points[0], transformed_points[1]

def find_closest_points(reference: NDArray[np.float64], 
                       target: NDArray[np.float64], 
                       max_points: int) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Find the closest points in reference array to target array.
    
    Parameters:
    reference (array): reference points
    target (array): target points
    max_points (int): maximum number of points to return
    
    Returns:
    tuple: closest points and their indices

    Raises:
        ValueError: If input arrays are empty or max_points is invalid
    """
    if len(reference) == 0 or len(target) == 0:
        raise ValueError("Input arrays cannot be empty")
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if max_points > len(reference):
        raise ValueError("max_points cannot be larger than the number of reference points")

    reference = np.asarray(reference)
    target = np.asarray(target)
    differences = np.abs(reference[:, None] - target[None, :])
    closest_indices = np.argmin(differences, axis=0)
    closest_points = np.unique(reference[closest_indices])
    
    if len(closest_points) > max_points:
        distances = np.abs(closest_points[:, None] - target).min(axis=1)
        sorted_indices = np.argsort(distances)
        closest_points = closest_points[sorted_indices][:max_points]
    
    return closest_points, closest_indices
