"""Tests for centerline-based morphometry."""

import numpy as np
import pytest

from fiber_tracer.analysis.morphometry import (
    equivalent_diameter_from_volume,
    ordered_path_length,
    per_fiber_volumes,
    tortuosity,
)


SPACING = (1.0, 1.0, 1.0)
ANISOTROPIC_SPACING = (2.0, 1.0, 1.0)


def test_ordered_path_length_straight_diagonal():
    """A straight diagonal from (0,0,0) to (2,2,2) has physical length sqrt(12)."""
    path = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    length = ordered_path_length(path, SPACING)
    expected = np.sqrt(12.0)
    assert np.isclose(length, expected)


def test_ordered_path_length_anisotropic():
    """Anisotropic spacing scales the physical distance correctly.

    Voxel coordinates are ordered (z, y, x); the path below moves along x.
    """
    path = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)
    # Only the x-coordinate (last column) changes; x spacing is 1.0.
    length = ordered_path_length(path, ANISOTROPIC_SPACING)
    assert np.isclose(length, 2.0)


def test_tortuosity_straight_line():
    """Tortuosity of a straight line equals 1.0."""
    path = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    assert np.isclose(tortuosity(path, SPACING), 1.0)


def test_tortuosity_zigzag_greater_than_one():
    """A zig-zag path has arc length longer than the endpoint chord."""
    path = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]], dtype=float)
    tau = tortuosity(path, SPACING)
    assert tau > 1.0


def test_tortuosity_single_point():
    """Tortuosity of a degenerate single-point path is defined as 1.0."""
    path = np.array([[5.0, 5.0, 5.0]])
    assert tortuosity(path, SPACING) == 1.0


def test_equivalent_diameter_from_volume_known_sphere():
    """A sphere with diameter 4 µm should yield equivalent diameter ~4 µm."""
    diameter_um = 4.0
    radius_um = diameter_um / 2.0
    volume_um3 = (4.0 / 3.0) * np.pi * radius_um ** 3
    # Under isotropic 1 µm spacing, one voxel is 1 µm³.
    n_voxels = int(round(volume_um3))
    result = equivalent_diameter_from_volume(n_voxels, SPACING)
    # Integer voxel discretisation of a sphere allows a small tolerance.
    assert np.isclose(result, diameter_um, atol=0.05)


def test_per_fiber_volumes_excludes_background():
    """per_fiber_volumes excludes label 0 and returns correct counts."""
    labels = np.array([
        [1, 1, 0, 2],
        [1, 0, 0, 2],
        [3, 3, 0, 2],
    ], dtype=np.int32)
    volumes = per_fiber_volumes(labels)
    assert volumes == {1: 3, 2: 3, 3: 2}
