"""Synthetic fiber phantoms with ground truth."""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FiberPhantom:
    volume: np.ndarray
    labels: np.ndarray
    orientations: np.ndarray  # Nx3 unit vectors
    diameters_um: np.ndarray
    lengths_um: np.ndarray
    voxel_spacing_um: Tuple[float, float, float]


def generate_straight_fiber(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    direction: np.ndarray,
    radius_voxels: float,
    intensity: float = 1.0,
) -> np.ndarray:
    """Draw a single straight cylinder in a binary volume."""
    direction = direction / np.linalg.norm(direction)
    z, y, x = np.indices(shape, dtype=float)
    coords = np.stack([z, y, x], axis=-1)
    center_vec = np.array(center)
    to_center = coords - center_vec
    projection = np.dot(to_center, direction)
    perpendicular = to_center - projection[:, :, :, None] * direction
    distance = np.linalg.norm(perpendicular, axis=-1)
    volume = np.zeros(shape, dtype=float)
    volume[distance <= radius_voxels] = intensity
    return volume


def generate_fiber_phantom(
    shape: Tuple[int, int, int] = (64, 64, 64),
    n_fibers: int = 10,
    fiber_diameter_um: float = 4.0,
    voxel_spacing_um: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 0.02,
    seed: Optional[int] = None,
) -> FiberPhantom:
    """Generate a phantom with straight, non-touching fibers."""
    rng = np.random.default_rng(seed)
    radius_voxels = 0.5 * fiber_diameter_um / min(voxel_spacing_um)
    volume = np.zeros(shape, dtype=float)
    labels = np.zeros(shape, dtype=np.int32)
    orientations = []
    diameters = []
    lengths = []
    next_label = 1

    for _i in range(n_fibers):
        center = rng.uniform(radius_voxels * 2, np.array(shape) - radius_voxels * 2)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        fiber = generate_straight_fiber(shape, tuple(center), direction, radius_voxels)
        mask = fiber > 0
        if np.any(labels[mask] > 0):
            continue
        labels[mask] = next_label
        next_label += 1
        volume += fiber
        orientations.append(direction)
        diameters.append(fiber_diameter_um)
        lengths.append(min(shape) * min(voxel_spacing_um))

    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
    )


def generate_phantom_with_known_orientation(
    shape: Tuple[int, int, int] = (64, 64, 64),
    direction: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float, float]] = None,
    fiber_diameter_um: float = 4.0,
    voxel_spacing_um: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity: float = 1.0,
) -> FiberPhantom:
    """Generate a deterministic phantom with a single fiber along a known direction.

    This is useful for validating orientation estimation because the ground-truth
    orientation is exactly the requested unit direction.
    """
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("direction must be a non-zero vector")
    direction = direction / norm

    if center is None:
        center = tuple((np.array(shape) - 1.0) / 2.0)

    radius_voxels = 0.5 * fiber_diameter_um / min(voxel_spacing_um)
    volume = generate_straight_fiber(
        shape, center, direction, radius_voxels, intensity=intensity
    )
    labels = (volume > 0).astype(np.int32)
    length = min(shape) * min(voxel_spacing_um)

    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array([direction]),
        diameters_um=np.array([fiber_diameter_um]),
        lengths_um=np.array([length]),
        voxel_spacing_um=voxel_spacing_um,
    )
