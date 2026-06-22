# tests/test_ml_backend.py
"""Tests for the optional ML segmentation backend adapter."""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fiber_tracer.backends.ml_segmentation import MLSegmentationBackend
from fiber_tracer.exceptions import BackendNotAvailableError


_ORIGINAL_IMPORT = builtins.__import__


def _import_side_effect(name, *args, **kwargs):
    if name == "torch":
        raise ImportError("No module named 'torch'")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def _fake_torch_import(name, *args, **kwargs):
    if name == "torch":
        return SimpleNamespace(__version__="0.0.0")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def test_ml_backend_raises_when_torch_unavailable():
    with patch.object(builtins, "__import__", side_effect=_import_side_effect):
        with pytest.raises(BackendNotAvailableError, match="Install ml extra"):
            MLSegmentationBackend()


def test_ml_backend_segment_raises_when_model_not_loaded():
    with patch.object(builtins, "__import__", side_effect=_fake_torch_import):
        backend = MLSegmentationBackend()

    volume = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="No model is loaded"):
        backend.segment(volume)
