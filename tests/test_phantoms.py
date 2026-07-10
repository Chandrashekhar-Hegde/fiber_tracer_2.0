import numpy as np

from fiber_tracer.validation.phantoms import (
    generate_fiber_phantom,
    generate_phantom_with_known_orientation,
)


def test_supersampling_produces_partial_volume_edges():
    """supersample>1 anti-aliases fibre boundaries into fractional partial-volume voxels."""
    kwargs = dict(
        shape=(32, 32, 32),
        n_fibers=1,
        fiber_diameter_um=8.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="aligned",
        noise_std=0.0,
        seed=1,
    )
    hard = generate_fiber_phantom(supersample=1, **kwargs)
    soft = generate_fiber_phantom(supersample=4, **kwargs)

    # Hard rasterisation is binary; supersampled has intermediate edge values.
    assert not np.any((hard.volume > 0.01) & (hard.volume < 0.99))
    assert np.any((soft.volume > 0.01) & (soft.volume < 0.99))

    # Both recover the same single fibre.
    assert len(np.unique(hard.labels[hard.labels > 0])) == 1
    assert len(np.unique(soft.labels[soft.labels > 0])) == 1


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
