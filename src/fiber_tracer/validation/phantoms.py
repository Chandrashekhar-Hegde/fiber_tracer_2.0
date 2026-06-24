"""Synthetic fiber phantoms with ground truth."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FiberPhantom:
    volume: np.ndarray
    labels: np.ndarray
    orientations: np.ndarray  # Nx3 unit vectors
    diameters_um: np.ndarray
    lengths_um: np.ndarray
    voxel_spacing_um: tuple[float, float, float]
    fiber_diameter_um: float


def generate_straight_fiber(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
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


def _sample_direction(rng: np.random.Generator, mode: str, axis: int = 2) -> np.ndarray:
    """Return a unit fiber direction according to *mode*."""
    if mode == "random":
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        return direction / norm
    if mode == "aligned":
        direction = np.zeros(3)
        direction[axis] = 1.0
        return direction
    if mode == "in_plane":
        # Random orientation within the XY plane.
        angle = rng.uniform(0, 2 * np.pi)
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    if mode == "angle":
        # In-plane at a fixed angle (degrees) around Z.
        angle = np.deg2rad(rng.uniform(0, 360))
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    if mode == "orthogonal":
        # Choose one of the three cardinal axes.
        direction = np.zeros(3)
        direction[rng.integers(0, 3)] = 1.0
        return direction
    if mode == "woven":
        # Alternating 0/90 in-plane tows.
        angle = 0.0 if rng.random() > 0.5 else np.pi / 2
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    if mode == "twill":
        # ±45° in-plane tows.
        angle = np.pi / 4 if rng.random() > 0.5 else -np.pi / 4
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    raise ValueError(f"Unknown orientation mode: {mode}")


def _clip_fiber_length(
    shape: tuple[int, int, int],
    center: np.ndarray,
    direction: np.ndarray,
    length_um: float,
    voxel_spacing_um: tuple[float, float, float],
) -> float:
    """Return the in-bounds length for a fiber centered at *center*."""
    diag = float(np.linalg.norm(np.array(shape) * np.array(voxel_spacing_um)))
    if length_um <= 0 or length_um > diag:
        return diag
    return length_um


def _generate_finite_fiber(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    direction: np.ndarray,
    radius_voxels: float,
    length_um: float,
    voxel_spacing_um: tuple[float, float, float],
    intensity: float = 1.0,
) -> np.ndarray:
    """Draw a finite-length cylinder (caps included) in a binary volume."""
    direction = direction / np.linalg.norm(direction)
    z, y, x = np.indices(shape, dtype=float)
    coords = np.stack([z, y, x], axis=-1)
    center_vec = np.array(center)
    to_center = coords - center_vec
    projection = np.dot(to_center, direction)
    perpendicular = to_center - projection[:, :, :, None] * direction
    distance = np.linalg.norm(perpendicular, axis=-1)
    half_length_voxels = 0.5 * length_um / min(voxel_spacing_um)
    volume = np.zeros(shape, dtype=float)
    mask = (distance <= radius_voxels) & (np.abs(projection) <= half_length_voxels)
    volume[mask] = intensity
    return volume


def generate_fiber_phantom(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_fibers: int = 10,
    fiber_diameter_um: float = 4.0,
    fiber_length_um: float = -1.0,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 0.02,
    orientation_mode: str = "random",
    broken_fraction: float = 0.0,
    n_broken_pieces: int = 2,
    porosity: float = 0.0,
    seed: Optional[int] = None,
) -> FiberPhantom:
    """Generate a phantom with configurable fiber architecture.

    Parameters
    ----------
    orientation_mode:
        ``random`` (default), ``aligned``, ``in_plane``, ``orthogonal``,
        ``woven`` (0/90), or ``twill`` (±45).
    fiber_length_um:
        Fiber length in micrometres.  Negative values use the volume diagonal.
    broken_fraction:
        Fraction of fibers that are broken into *n_broken_pieces* segments.
    n_broken_pieces:
        Number of segments per broken fiber.
    porosity:
        Probability of inserting spherical voids (porosity) into the matrix.
    """
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
        direction = _sample_direction(rng, orientation_mode)
        length = _clip_fiber_length(shape, center, direction, fiber_length_um, voxel_spacing_um)

        if rng.random() < broken_fraction:
            # Split the fiber into segments along its axis.
            segment_length = length / n_broken_pieces
            half_total = length / 2
            starts = [-half_total + k * segment_length for k in range(n_broken_pieces)]
            for start in starts:
                seg_center = center + direction * (start + segment_length / 2)
                fiber = _generate_finite_fiber(
                    shape,
                    tuple(seg_center),
                    direction,
                    radius_voxels,
                    segment_length * 0.9,  # tiny gap between pieces
                    voxel_spacing_um,
                )
                mask = fiber > 0
                if np.any(labels[mask] > 0):
                    continue
                labels[mask] = next_label
                next_label += 1
                volume += fiber
                orientations.append(direction)
                diameters.append(fiber_diameter_um)
                lengths.append(segment_length * 0.9)
        else:
            fiber = _generate_finite_fiber(
                shape, tuple(center), direction, radius_voxels, length, voxel_spacing_um
            )
            mask = fiber > 0
            if np.any(labels[mask] > 0):
                continue
            labels[mask] = next_label
            next_label += 1
            volume += fiber
            orientations.append(direction)
            diameters.append(fiber_diameter_um)
            lengths.append(length)

    # Add spherical voids to simulate porosity / matrix damage.
    if porosity > 0:
        n_voids = int(np.prod(shape) * porosity / (4 / 3 * np.pi * radius_voxels**3))
        for _ in range(max(1, n_voids)):
            void_center = rng.uniform(0, np.array(shape))
            void_radius = rng.uniform(radius_voxels, 2 * radius_voxels)
            zz, yy, xx = np.indices(shape, dtype=float)
            dist = np.sqrt(
                (zz - void_center[0]) ** 2 + (yy - void_center[1]) ** 2 + (xx - void_center[2]) ** 2
            )
            void_mask = dist <= void_radius
            volume[void_mask] = 0.0
            labels[void_mask] = 0

    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=fiber_diameter_um,
    )


def generate_phantom_with_known_orientation(
    shape: tuple[int, int, int] = (64, 64, 64),
    direction: Optional[np.ndarray] = None,
    center: Optional[tuple[float, float, float]] = None,
    fiber_diameter_um: float = 4.0,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
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
    volume = generate_straight_fiber(shape, center, direction, radius_voxels, intensity=intensity)
    labels = (volume > 0).astype(np.int32)
    length = min(shape) * min(voxel_spacing_um)

    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array([direction]),
        diameters_um=np.array([fiber_diameter_um]),
        lengths_um=np.array([length]),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=fiber_diameter_um,
    )
