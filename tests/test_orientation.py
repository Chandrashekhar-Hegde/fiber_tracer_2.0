"""Tests for orientation backends."""

import builtins
import sys
from unittest import mock

import numpy as np
import pytest

from fiber_tracer.exceptions import BackendNotAvailableError
from fiber_tracer.orientation.pca import pca_orientation
from fiber_tracer.orientation.structure_tensor import (
    compute_structure_tensor_field,
    orientation_from_smallest_eigenvector,
)


class _BlockImport:
    """Context manager that blocks import of a module name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._original_import = builtins.__import__
        self._modules_to_restore = {}

    def __enter__(self) -> "_BlockImport":
        def _blocked_import(name: str, *args, **kwargs):
            if name == self.name or name.startswith(self.name + "."):
                raise ImportError(f"No module named '{name}'")
            return self._original_import(name, *args, **kwargs)

        builtins.__import__ = _blocked_import
        # Remove any cached modules so the blocked import path is exercised.
        for key in list(sys.modules.keys()):
            if key == self.name or key.startswith(self.name + "."):
                self._modules_to_restore[key] = sys.modules.pop(key)
        return self

    def __exit__(self, *exc) -> None:
        builtins.__import__ = self._original_import
        sys.modules.update(self._modules_to_restore)


def test_compute_structure_tensor_field_missing_backend():
    """BackendNotAvailableError is raised when structure_tensor is absent."""
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    with _BlockImport("structure_tensor"):
        with pytest.raises(BackendNotAvailableError):
            compute_structure_tensor_field(volume, sigma=1.0, rho=2.0)


def test_orientation_from_smallest_eigenvector():
    """First eigenvector slice is returned with shape (3, D, H, W)."""
    eigenvectors = np.zeros((3, 3, 4, 4, 4), dtype=np.float64)
    eigenvectors[0, 0, ...] = 1.0
    direction = orientation_from_smallest_eigenvector(eigenvectors)
    assert direction.shape == (3, 4, 4, 4)
    np.testing.assert_array_equal(direction, eigenvectors[0])


def test_pca_orientation_collinear_3d():
    """Principal axis of collinear 3D points is a unit vector along the line."""
    t = np.linspace(-1.0, 1.0, 20)
    coords = np.column_stack([t, 2.0 * t, 3.0 * t])
    axis = pca_orientation(coords)
    expected = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
    assert np.abs(np.linalg.norm(axis) - 1.0) < 1e-6
    np.testing.assert_allclose(np.abs(axis), np.abs(expected), atol=1e-6)


def test_pca_orientation_2d_in_3d():
    """PCA on points lying in a plane returns a unit vector in that plane."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=50)
    y = rng.uniform(0.0, 1.0, size=50)
    coords = np.column_stack([x, y, np.zeros_like(x)])
    axis = pca_orientation(coords)
    assert np.abs(np.linalg.norm(axis) - 1.0) < 1e-6
    assert np.abs(axis[2]) < 1e-6
