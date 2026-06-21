"""Orientation estimation using the structure-tensor package."""

from typing import Tuple
import numpy as np
from fiber_tracer.exceptions import BackendNotAvailableError


def compute_structure_tensor_field(
    volume: np.ndarray,
    sigma: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors of the 3D structure tensor."""
    try:
        from structure_tensor import structure_tensor_3d
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Install structure extra: pip install fiber-tracer[structure]"
        ) from exc

    eigenvalues, eigenvectors = structure_tensor_3d(volume, sigma, rho, truncate=4.0)
    return eigenvalues, eigenvectors


def orientation_from_smallest_eigenvector(eigenvectors: np.ndarray) -> np.ndarray:
    """Eigenvector of smallest eigenvalue points along the fiber."""
    # eigenvectors shape: (3, 3, D, H, W) where first dim is eigenvalue index (0=smallest)
    direction = eigenvectors[0]  # (3, D, H, W)
    return direction
