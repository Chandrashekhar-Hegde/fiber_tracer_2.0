# tests/test_tda_backend.py
"""Tests for optional TDA descriptor backends."""

import builtins
import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

from fiber_tracer.backends.tda_gudhi import betti_numbers, persistence_summary
from fiber_tracer.exceptions import BackendNotAvailableError

_ORIGINAL_IMPORT = builtins.__import__


def _import_side_effect(name, *args, **kwargs):
    if name == "gudhi":
        raise ImportError(f"No module named '{name}'")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def test_betti_numbers_raises_when_gudhi_unavailable():
    volume = np.ones((3, 3, 3), dtype=bool)
    with patch.object(builtins, "__import__", side_effect=_import_side_effect):
        with pytest.raises(BackendNotAvailableError, match="Install tda extra"):
            betti_numbers(volume)


def test_persistence_summary_raises_when_gudhi_unavailable():
    volume = np.ones((3, 3, 3), dtype=bool)
    with patch.object(builtins, "__import__", side_effect=_import_side_effect):
        with pytest.raises(BackendNotAvailableError, match="Install tda extra"):
            persistence_summary(volume)


def test_betti_numbers_rejects_non_3d_volume():
    with patch("fiber_tracer.backends.tda_gudhi._import_gudhi") as mock_import:
        mock_import.return_value = object()
        with pytest.raises(ValueError, match="Expected 3D binary volume"):
            betti_numbers(np.ones((3, 3), dtype=bool))


@pytest.mark.skipif(importlib.util.find_spec("gudhi") is None, reason="gudhi not installed")
def test_betti_numbers_solid_cube():
    volume = np.ones((5, 5, 5), dtype=bool)
    result = betti_numbers(volume)
    assert result == {"b0": 1, "b1": 0, "b2": 0}


@pytest.mark.skipif(importlib.util.find_spec("gudhi") is None, reason="gudhi not installed")
def test_betti_numbers_hollow_cube_shell():
    volume = np.ones((5, 5, 5), dtype=bool)
    volume[1:-1, 1:-1, 1:-1] = False
    result = betti_numbers(volume)
    assert result["b0"] == 1
    assert result["b2"] == 1
