"""Orientation estimation using the structure-tensor package."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from fiber_tracer.config import VoxelSpacing
from fiber_tracer.exceptions import BackendNotAvailableError


def compute_structure_tensor_field(
    volume: np.ndarray,
    sigma: float,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors of the 3D structure tensor."""
    try:
        from structure_tensor import structure_tensor_3d
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Install structure extra: pip install fiber-tracer[structure]"
        ) from exc

    eigenvalues, eigenvectors = structure_tensor_3d(volume, sigma, rho, truncate=4.0)
    return eigenvalues, eigenvectors


def compute_local_orientation_field(
    volume: np.ndarray,
    sigma_um: float,
    rho_um: float,
    voxel_spacing: VoxelSpacing,
    truncate: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors of the 3D gradient structure tensor.

    Eigenvalues are sorted ascending (eigenvalues[0] = smallest).
    Eigenvectors are columns: eigenvectors[i] is the eigenvector for eigenvalues[i].
    Implements a gradient-based structure tensor using scipy.ndimage so it works
    without the optional structure-tensor package.
    """
    spacing = np.array([voxel_spacing.z, voxel_spacing.y, voxel_spacing.x], dtype=float)
    sigma_voxels = sigma_um / spacing
    rho_voxels = rho_um / spacing

    # Gaussian derivatives (gradient) along each axis.
    # Divide by the physical voxel spacing so derivatives are in intensity per
    # physical unit; this gives structure-tensor components consistent units
    # under anisotropic spacing.
    derivatives = []
    for axis in range(3):
        derivative = ndimage.gaussian_filter1d(
            volume,
            sigma=sigma_voxels[axis],
            order=1,
            axis=axis,
            mode="nearest",
            truncate=truncate,
        )
        derivative = derivative / spacing[axis]
        derivatives.append(derivative)

    # Structure-tensor components: outer product of gradients.
    structure_components = {}
    axes = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for i, j in axes:
        key = (min(i, j), max(i, j))
        if key not in structure_components:
            structure_components[key] = derivatives[i] * derivatives[j]

    # Smooth each component with Gaussian of width rho.
    smoothed = {}
    for key, component in structure_components.items():
        smoothed[key] = ndimage.gaussian_filter(
            component,
            sigma=rho_voxels,
            mode="nearest",
            truncate=truncate,
        )

    # Assemble symmetric 3x3 structure tensor at each voxel.
    shape = volume.shape
    tensor = np.zeros(shape + (3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            key = (min(i, j), max(i, j))
            tensor[..., i, j] = smoothed[key]

    # Vectorized eigen-decomposition of symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)

    # np.linalg.eigh returns ascending eigenvalues; eigenvectors are columns.
    # Transpose to (eigen_index, ...) ordering.
    eigenvalues = np.transpose(eigenvalues, axes=(3, 0, 1, 2))
    eigenvectors = np.transpose(eigenvectors, axes=(4, 3, 0, 1, 2))
    return eigenvalues, eigenvectors


def orientation_from_smallest_eigenvector(eigenvectors: np.ndarray) -> np.ndarray:
    """Eigenvector of smallest eigenvalue points along the fiber."""
    # eigenvectors shape: (3, 3, D, H, W) where first dim is eigenvalue index (0=smallest)
    return np.asarray(eigenvectors[0], dtype=np.float64)
