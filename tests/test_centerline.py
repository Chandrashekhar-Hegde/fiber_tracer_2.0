"""Tests for 3D skeletonization and optional skan graph analysis."""

import numpy as np
import pytest
from unittest.mock import patch

from fiber_tracer.centerline.skeleton import skeletonize_label_volume
from fiber_tracer.centerline.graph import skeleton_to_skan
from fiber_tracer.exceptions import BackendNotAvailableError


def _cylinder_labels(shape, center, radius, height, label):
    """Return a labeled volume containing a solid cylinder along the z-axis."""
    z, y, x = np.indices(shape, dtype=float)
    dy = y - center[1]
    dx = x - center[2]
    dist_sq = dy**2 + dx**2
    labels = np.zeros(shape, dtype=int)
    z_min = max(0, center[0] - height // 2)
    z_max = min(shape[0], center[0] + height // 2)
    mask = (dist_sq <= radius**2) & (z >= z_min) & (z < z_max)
    labels[mask] = label
    return labels


def test_skeletonize_label_volume_returns_boolean_same_shape():
    labels = _cylinder_labels((20, 20, 20), center=(10, 10, 10), radius=3, height=15, label=1)
    skeleton = skeletonize_label_volume(labels)
    assert skeleton.shape == labels.shape
    assert skeleton.dtype == bool


def test_skeletonize_label_volume_thins_each_label():
    labels = np.zeros((20, 20, 20), dtype=int)
    # Two non-touching thick cylinders.
    labels |= _cylinder_labels(
        (20, 20, 20), center=(10, 6, 6), radius=3, height=15, label=1
    )
    labels |= _cylinder_labels(
        (20, 20, 20), center=(10, 14, 14), radius=3, height=15, label=2
    )

    skeleton = skeletonize_label_volume(labels)
    original_foreground = np.sum(labels > 0)
    skeleton_foreground = np.sum(skeleton)

    assert skeleton_foreground > 0
    assert skeleton_foreground < original_foreground

    # Both labels should still be present in the skeleton.
    assert np.any(skeleton & (labels == 1))
    assert np.any(skeleton & (labels == 2))


def test_skeleton_to_skan_raises_when_skan_missing():
    """BackendNotAvailableError must be raised when skan cannot be imported."""

    def fake_import(name, *args, **kwargs):
        if name == "skan":
            raise ImportError("No module named 'skan'")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    skeleton = np.ones((5, 5, 5), dtype=bool)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(BackendNotAvailableError):
            skeleton_to_skan(skeleton)
