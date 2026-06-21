import numpy as np

from fiber_tracer.validation.phantoms import (
    generate_fiber_phantom,
    generate_phantom_with_known_orientation,
)


def test_phantom_has_expected_fibers():
    n_fibers = 3
    phantom = generate_fiber_phantom(
        shape=(128, 128, 128), n_fibers=n_fibers, fiber_diameter_um=2.0, seed=42
    )
    assert phantom.volume.shape == (128, 128, 128)
    assert len(phantom.orientations) == n_fibers
    assert len(np.unique(phantom.labels[phantom.labels > 0])) == n_fibers
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
    phantom = generate_fiber_phantom(shape=(32, 32, 32), n_fibers=5, fiber_diameter_um=2.0, seed=42)
    assert phantom.volume.min() >= 0.0
    assert phantom.volume.max() <= 1.0


def test_phantom_labels_are_pairwise_disjoint():
    n_fibers = 5
    phantom = generate_fiber_phantom(
        shape=(128, 128, 128), n_fibers=n_fibers, fiber_diameter_um=2.0, seed=42
    )
    labels = phantom.labels
    present_labels = np.unique(labels[labels > 0])
    assert len(present_labels) == n_fibers
    for i in present_labels:
        for j in present_labels:
            if i == j:
                continue
            overlap = np.logical_and(labels == i, labels == j).sum()
            assert overlap == 0


def test_phantom_ground_truth_alignment():
    phantom = generate_fiber_phantom(
        shape=(32, 32, 32),
        n_fibers=20,
        fiber_diameter_um=4.0,
        seed=42,
    )
    present_labels = np.unique(phantom.labels)
    # Drop background label 0.
    placed_labels = present_labels[present_labels > 0]
    n_placed = len(placed_labels)

    assert n_placed >= 1
    assert len(phantom.orientations) == n_placed
    assert len(phantom.diameters_um) == n_placed
    assert len(phantom.lengths_um) == n_placed
    assert np.array_equal(placed_labels, np.arange(1, n_placed + 1))
