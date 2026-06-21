"""Tests for Advani-Tucker orientation tensor utilities."""

import numpy as np
import pytest

from fiber_tracer.orientation.tensor import (
    aggregate_direction_tensor,
    direction_tensor,
    fractional_anisotropy,
    windowed_orientation_tensor_field,
)


def _random_unit_directions(rng: np.random.Generator, n: int) -> np.ndarray:
    directions = rng.normal(size=(n, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def test_direction_tensor_is_symmetric_psd():
    """A2 is symmetric and positive semi-definite for random unit directions."""
    rng = np.random.default_rng(42)
    directions = _random_unit_directions(rng, 200)
    a2 = direction_tensor(directions)

    assert a2.shape == (3, 3)
    np.testing.assert_allclose(a2, a2.T, atol=1e-12)

    eigenvalues = np.linalg.eigvalsh(a2)
    assert np.all(eigenvalues >= -1e-12)
    np.testing.assert_allclose(np.trace(a2), 1.0, atol=1e-12)


def test_direction_tensor_perfectly_aligned():
    """Perfectly aligned directions give one eigenvalue near 1 and two near 0."""
    direction = np.array([1.0, 0.0, 0.0])
    directions = np.tile(direction, (50, 1))
    a2 = direction_tensor(directions)

    eigenvalues = np.sort(np.linalg.eigvalsh(a2))
    np.testing.assert_allclose(eigenvalues, [0.0, 0.0, 1.0], atol=1e-12)

    principal_axis = np.linalg.eigh(a2)[1][:, -1]
    np.testing.assert_allclose(np.abs(principal_axis), np.abs(direction), atol=1e-12)


def test_fractional_anisotropy_isotropic_and_aligned():
    """FA is 0 for an isotropic distribution and close to 1 for aligned fibers."""
    isotropic = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Repeat to keep the mean well-defined with equal weights.
    isotropic = np.repeat(isotropic, 20, axis=0)
    a2_iso = direction_tensor(isotropic)
    assert fractional_anisotropy(a2_iso) == pytest.approx(0.0, abs=1e-12)

    aligned = np.tile([0.0, 0.0, 1.0], (100, 1))
    a2_aligned = direction_tensor(aligned)
    assert fractional_anisotropy(a2_aligned) == pytest.approx(1.0, abs=1e-12)


def test_windowed_orientation_tensor_field_shapes_and_anisotropy():
    """Windowed A2 field has expected shapes and lower FA for larger windows."""
    rng = np.random.default_rng(0)
    z_dim, y_dim, x_dim = 16, 16, 16

    # Random unit direction field.
    directions = rng.normal(size=(3, z_dim, y_dim, x_dim))
    directions = directions / np.linalg.norm(directions, axis=0, keepdims=True)

    tensor_field, centers = windowed_orientation_tensor_field(directions, window_size=3, stride=2)
    expected_z = (z_dim - 3) // 2 + 1
    expected_y = (y_dim - 3) // 2 + 1
    expected_x = (x_dim - 3) // 2 + 1
    assert tensor_field.shape == (expected_z, expected_y, expected_x, 3, 3)
    assert centers.shape == (expected_z, expected_y, expected_x, 3)

    # Centers should lie inside the volume and match the first expected center.
    assert centers[0, 0, 0, 0] == 1
    assert np.all(centers[..., 0] >= 1) and np.all(centers[..., 0] < z_dim - 1)
    assert np.all(centers[..., 1] >= 1) and np.all(centers[..., 1] < y_dim - 1)
    assert np.all(centers[..., 2] >= 1) and np.all(centers[..., 2] < x_dim - 1)

    # Every tensor should be symmetric and have trace 1 (unit directions).
    for i in range(expected_z):
        for j in range(expected_y):
            for k in range(expected_x):
                a2 = tensor_field[i, j, k]
                np.testing.assert_allclose(a2, a2.T, atol=1e-12)
                np.testing.assert_allclose(np.trace(a2), 1.0, atol=1e-12)

    # Larger windows average over more random directions, reducing mean FA.
    small, _ = windowed_orientation_tensor_field(directions, window_size=3, stride=4)
    large, _ = windowed_orientation_tensor_field(directions, window_size=7, stride=4)

    fa_small = np.array(
        [
            fractional_anisotropy(small[i, j, k])
            for i in range(small.shape[0])
            for j in range(small.shape[1])
            for k in range(small.shape[2])
        ]
    )
    fa_large = np.array(
        [
            fractional_anisotropy(large[i, j, k])
            for i in range(large.shape[0])
            for j in range(large.shape[1])
            for k in range(large.shape[2])
        ]
    )

    assert fa_large.mean() < fa_small.mean()


def test_aggregate_direction_tensor_alias():
    """Global A2 alias agrees with direction_tensor on flattened directions."""
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(3, 8, 8, 8))
    directions = directions / np.linalg.norm(directions, axis=0, keepdims=True)

    global_a2 = aggregate_direction_tensor(directions)
    expected = direction_tensor(directions.reshape(3, -1).T)

    np.testing.assert_allclose(global_a2, expected, atol=1e-12)

    # Also works with an (N, 3) array.
    flat = directions.reshape(3, -1).T
    np.testing.assert_allclose(aggregate_direction_tensor(flat), expected, atol=1e-12)
