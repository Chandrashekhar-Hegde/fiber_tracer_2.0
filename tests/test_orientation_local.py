"""Tests for the local gradient-based orientation field."""

import numpy as np

from fiber_tracer.config import VoxelSpacing
from fiber_tracer.orientation.structure_tensor import (
    compute_local_orientation_field,
    orientation_from_smallest_eigenvector,
)


def _make_cylinder_fiber(shape, direction, radius_voxels, center=None):
    """Return a binary volume containing a straight cylinder."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    if center is None:
        center = np.array(shape) / 2.0
    z, y, x = np.indices(shape, dtype=float)
    coords = np.stack([z, y, x], axis=-1)
    to_center = coords - np.array(center)
    projection = np.dot(to_center, direction)
    perpendicular = to_center - projection[..., None] * direction
    distance = np.linalg.norm(perpendicular, axis=-1)
    return (distance <= radius_voxels).astype(np.float32)


def test_local_orientation_field_aligns_with_fiber_direction():
    """A phantom with a known fiber direction yields aligned local orientations."""
    shape = (48, 48, 48)
    direction = np.array([0.0, 0.0, 1.0])
    fiber = _make_cylinder_fiber(shape, direction, radius_voxels=5.0)

    eigenvalues, eigenvectors = compute_local_orientation_field(
        fiber,
        sigma_um=1.0,
        rho_um=2.0,
        voxel_spacing=VoxelSpacing(1.0, 1.0, 1.0),
    )

    direction_field = orientation_from_smallest_eigenvector(eigenvectors)

    # Sample near the center of the cylinder, away from edges.
    cx, cy, cz = shape[2] // 2, shape[1] // 2, shape[0] // 2
    sample = direction_field[:, cz - 5 : cz + 5, cy - 5 : cy + 5, cx - 5 : cx + 5]
    sample = sample.reshape(3, -1)
    mean_direction = sample.mean(axis=1)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)

    assert np.abs(np.dot(mean_direction, direction)) > 0.95


def test_local_orientation_field_handles_anisotropic_spacing():
    """Anisotropic spacing preserves the true physical fiber direction."""
    shape = (64, 64, 64)
    spacing = VoxelSpacing(2.0, 1.0, 1.0)
    # Voxel-space direction [0, 1, 1] maps to physical direction [0, 1, 1]
    # after component-wise scaling by (z=2, y=1, x=1).
    direction_voxel = np.array([0.0, 1.0, 1.0])
    true_physical_direction = direction_voxel * np.array([spacing.z, spacing.y, spacing.x])
    true_physical_direction = true_physical_direction / np.linalg.norm(true_physical_direction)
    fiber = _make_cylinder_fiber(shape, direction_voxel, radius_voxels=6.0)

    eigenvalues, eigenvectors = compute_local_orientation_field(
        fiber,
        sigma_um=2.0,
        rho_um=4.0,
        voxel_spacing=spacing,
    )

    assert eigenvalues.shape == (3,) + shape
    assert eigenvectors.shape == (3, 3) + shape

    direction_field = orientation_from_smallest_eigenvector(eigenvectors)

    # Sample near the center of the cylinder, away from edges.
    cx, cy, cz = shape[2] // 2, shape[1] // 2, shape[0] // 2
    sample = direction_field[:, cz - 5 : cz + 5, cy - 5 : cy + 5, cx - 5 : cx + 5]
    sample = sample.reshape(3, -1)
    mean_direction = sample.mean(axis=1)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)

    angle_rad = np.arccos(
        np.clip(np.abs(np.dot(mean_direction, true_physical_direction)), 0.0, 1.0)
    )
    angle_deg = np.degrees(angle_rad)
    assert angle_deg < 15.0


def test_local_orientation_eigenvectors_are_unit_vectors():
    """Every returned eigenvector has unit norm."""
    shape = (16, 16, 16)
    rng = np.random.default_rng(0)
    volume = rng.random(shape).astype(np.float32)

    eigenvalues, eigenvectors = compute_local_orientation_field(
        volume,
        sigma_um=1.0,
        rho_um=1.5,
        voxel_spacing=VoxelSpacing(1.0, 1.0, 1.0),
    )

    # eigenvectors[i] is the field of eigenvectors for eigenvalues[i].
    for i in range(3):
        norms = np.linalg.norm(eigenvectors[i], axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_local_orientation_eigenvalues_sorted_ascending():
    """Eigenvalues are sorted ascending at every voxel."""
    shape = (16, 16, 16)
    rng = np.random.default_rng(1)
    volume = rng.random(shape).astype(np.float32)

    eigenvalues, _ = compute_local_orientation_field(
        volume,
        sigma_um=1.0,
        rho_um=1.5,
        voxel_spacing=VoxelSpacing(1.0, 1.0, 1.0),
    )

    # Check ascending order along the eigenvalue index axis.
    assert np.all(eigenvalues[0] <= eigenvalues[1])
    assert np.all(eigenvalues[1] <= eigenvalues[2])
