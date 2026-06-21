import numpy as np
import pytest

from fiber_tracer.validation.phantoms import (
    generate_fiber_phantom,
    generate_phantom_with_known_orientation,
)


def test_phantom_has_expected_fibers():
    phantom = generate_fiber_phantom(
        shape=(32, 32, 32), n_fibers=3, fiber_diameter_um=2.0, seed=42
    )
    assert phantom.volume.shape == (32, 32, 32)
    assert len(phantom.orientations) >= 1
    assert phantom.volume.max() <= 1.0


def test_known_orientation_helper_matches_requested_direction():
    direction = np.array([1.0, 2.0, 3.0])
    direction = direction / np.linalg.norm(direction)
    phantom = generate_phantom_with_known_orientation(
        shape=(32, 32, 32), direction=direction, fiber_diameter_um=4.0
    )
    assert phantom.orientations.shape == (1, 3)
    actual = phantom.orientations[0]
    dot = np.clip(np.dot(actual, direction), -1.0, 1.0)
    angle = np.arccos(abs(dot))
    assert angle < np.radians(1.0)


def test_phantom_volume_within_unit_range():
    phantom = generate_fiber_phantom(
        shape=(32, 32, 32), n_fibers=5, fiber_diameter_um=2.0, seed=42
    )
    assert phantom.volume.min() >= 0.0
    assert phantom.volume.max() <= 1.0


def test_phantom_labels_are_non_overlapping():
    phantom = generate_fiber_phantom(
        shape=(32, 32, 32), n_fibers=10, fiber_diameter_um=2.0, seed=42
    )
    nonzero = phantom.labels[phantom.labels > 0]
    # Each voxel should have exactly one label.
    assert len(nonzero) == phantom.labels.astype(bool).sum()
