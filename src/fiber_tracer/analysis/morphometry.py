"""Fiber morphometry computed from ordered centerlines."""

import numpy as np


def ordered_path_length(path: np.ndarray, voxel_spacing: tuple[float, float, float]) -> float:
    """Sum Euclidean distances along an ordered path in physical units."""
    scaled = path * np.array(voxel_spacing)
    return float(np.sum(np.linalg.norm(np.diff(scaled, axis=0), axis=1)))


def tortuosity(path: np.ndarray, voxel_spacing: tuple[float, float, float]) -> float:
    """Arc length divided by endpoint Euclidean distance."""
    if len(path) < 2:
        return 1.0
    arc_length = ordered_path_length(path, voxel_spacing)
    scaled = path * np.array(voxel_spacing)
    chord = np.linalg.norm(scaled[-1] - scaled[0])
    if chord == 0:
        return 1.0
    return float(arc_length / chord)


def equivalent_diameter_from_volume(
    n_voxels: int, voxel_spacing: tuple[float, float, float]
) -> float:
    """Diameter of a sphere with equivalent volume."""
    volume_um3 = float(n_voxels * np.prod(voxel_spacing))
    return float(2.0 * (3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0))


def per_fiber_volumes(labels: np.ndarray) -> dict:
    """Return a mapping label_id -> voxel_count for all foreground labels.

    Background (label 0) is excluded.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts) if label != 0}
