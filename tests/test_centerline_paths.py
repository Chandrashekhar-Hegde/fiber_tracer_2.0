"""Tests for ordered per-fiber centerline extraction."""

import numpy as np

from fiber_tracer.analysis.morphometry import ordered_path_length, tortuosity
from fiber_tracer.centerline.paths import extract_fiber_paths


def test_straight_line_path_is_ordered_end_to_end():
    """A straight 1-voxel-thick fiber yields a contiguous ordered path."""
    labels = np.zeros((3, 3, 10), dtype=np.int32)
    labels[1, 1, 1:9] = 1  # segment along x at z=1, y=1
    skeleton = labels > 0

    paths = extract_fiber_paths(labels, skeleton)

    assert set(paths) == {1}
    path = paths[1]
    assert path.shape == (8, 3)
    # Consecutive voxels are direct neighbors (step magnitude 1 along x).
    diffs = np.abs(np.diff(path, axis=0))
    assert np.all(diffs.sum(axis=1) == 1)
    # Endpoints are the two ends of the segment.
    ends = {tuple(path[0]), tuple(path[-1])}
    assert ends == {(1, 1, 1), (1, 1, 8)}


def test_multiple_labels_yield_multiple_paths():
    labels = np.zeros((3, 12, 12), dtype=np.int32)
    labels[1, 2, 1:9] = 1
    labels[1, 8, 1:9] = 2
    skeleton = labels > 0

    paths = extract_fiber_paths(labels, skeleton)

    assert set(paths) == {1, 2}
    assert paths[1].shape[0] == 8
    assert paths[2].shape[0] == 8


def test_straight_path_has_unit_tortuosity():
    labels = np.zeros((3, 3, 10), dtype=np.int32)
    labels[1, 1, 1:9] = 1
    skeleton = labels > 0

    path = extract_fiber_paths(labels, skeleton)[1]
    spacing = (1.0, 1.0, 1.0)
    assert np.isclose(ordered_path_length(path, spacing), 7.0)
    assert np.isclose(tortuosity(path, spacing), 1.0)


def test_l_shaped_path_has_tortuosity_above_one():
    """An L-shaped fiber's arc length exceeds its endpoint chord."""
    labels = np.zeros((3, 12, 12), dtype=np.int32)
    labels[1, 1, 1:9] = 1  # horizontal arm
    labels[1, 1:9, 8] = 1  # vertical arm sharing the corner (1,1,8)
    skeleton = labels > 0

    path = extract_fiber_paths(labels, skeleton)[1]
    spacing = (1.0, 1.0, 1.0)
    # The path should traverse both arms (about 14 steps end-to-end).
    assert ordered_path_length(path, spacing) > 12.0
    assert tortuosity(path, spacing) > 1.3


def test_single_voxel_label_returns_single_point():
    labels = np.zeros((3, 3, 3), dtype=np.int32)
    labels[1, 1, 1] = 1
    skeleton = labels > 0

    paths = extract_fiber_paths(labels, skeleton)
    assert paths[1].shape == (1, 3)
