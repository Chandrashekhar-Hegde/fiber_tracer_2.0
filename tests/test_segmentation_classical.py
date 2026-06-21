"""Tests for classical 3D segmentation."""

import numpy as np

from fiber_tracer.segmentation.classical import (
    remove_small_objects,
    segment_otsu_3d,
    segment_watershed_3d,
)


def _sphere(shape, center, radius, intensity=1.0):
    """Return a binary/intensity sphere in a 3D volume."""
    z, y, x = np.indices(shape, dtype=float)
    dz = z - center[0]
    dy = y - center[1]
    dx = x - center[2]
    dist_sq = dz**2 + dy**2 + dx**2
    volume = np.zeros(shape, dtype=float)
    volume[dist_sq <= radius**2] = intensity
    return volume


def test_segment_otsu_3d_returns_boolean_same_shape():
    rng = np.random.default_rng(42)
    volume = rng.random((16, 16, 16))
    mask = segment_otsu_3d(volume)
    assert mask.shape == volume.shape
    assert mask.dtype == bool


def test_segment_otsu_3d_separates_bimodal_sphere():
    rng = np.random.default_rng(42)
    shape = (32, 32, 32)
    background = np.zeros(shape, dtype=float)
    foreground = _sphere(shape, center=(16, 16, 16), radius=6, intensity=0.9)
    volume = background + foreground
    volume = np.clip(volume + rng.normal(0, 0.02, shape), 0, 1)

    mask = segment_otsu_3d(volume)

    # Foreground sphere voxels should mostly be classified as foreground.
    sphere_voxels = foreground > 0
    foreground_fraction = np.sum(mask & sphere_voxels) / np.sum(sphere_voxels)
    assert foreground_fraction > 0.95

    # Background should remain mostly background.
    background_fraction = np.sum(mask & ~sphere_voxels) / np.sum(~sphere_voxels)
    assert background_fraction < 0.05


def test_segment_watershed_3d_splits_touching_spheres():
    shape = (48, 48, 48)
    # Two overlapping/touching spheres.
    sphere_a = _sphere(shape, center=(16, 24, 18), radius=8, intensity=1.0)
    sphere_b = _sphere(shape, center=(32, 24, 30), radius=8, intensity=1.0)
    foreground = ((sphere_a + sphere_b) > 0).astype(bool)

    labels = segment_watershed_3d(foreground, min_distance_voxels=3)

    # There should be at least two distinct foreground labels.
    unique_labels = np.unique(labels[labels > 0])
    assert len(unique_labels) >= 2

    # Each original sphere region should contain a distinct label.
    labels_in_a = np.unique(labels[sphere_a > 0])
    labels_in_b = np.unique(labels[sphere_b > 0])
    labels_in_a = labels_in_a[labels_in_a > 0]
    labels_in_b = labels_in_b[labels_in_b > 0]
    assert len(labels_in_a) >= 1
    assert len(labels_in_b) >= 1


def test_remove_small_objects_keeps_large_removes_small():
    labels = np.zeros((32, 32, 32), dtype=int)
    # Large cube.
    labels[4:14, 4:14, 4:14] = 1
    # Small cube.
    labels[20:22, 20:22, 20:22] = 2

    cleaned = remove_small_objects(labels, min_size_voxels=50)

    unique = np.unique(cleaned)
    assert 1 in unique
    assert 2 not in unique
