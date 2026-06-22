# tests/test_ml_backend.py
"""Tests for the optional ML segmentation backend adapter."""

import builtins
import importlib.util
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


def test_ml_backend_segment_raises_when_no_checkpoint_configured():
    with patch.object(builtins, "__import__", side_effect=_fake_torch_import):
        backend = MLSegmentationBackend()

    volume = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(RuntimeError, match="No model checkpoint"):
        backend.segment(volume)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_unet3d_forward_shape():
    import torch

    from fiber_tracer.backends.unet3d import UNet3D

    model = UNet3D(in_channels=1, out_channels=1, features=(8, 16, 32))
    x = torch.randn(1, 1, 32, 32, 32)
    y = model(x)
    assert y.shape == (1, 1, 32, 32, 32)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_ml_backend_loads_checkpoint_and_segments(tmp_path):
    import torch

    from fiber_tracer.backends.unet3d import UNet3D

    checkpoint = tmp_path / "model.pt"
    model = UNet3D(in_channels=1, out_channels=1, features=(8, 16, 32))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": (8, 16, 32),
            "patch_size": (16, 16, 16),
        },
        checkpoint,
    )

    backend = MLSegmentationBackend.from_checkpoint(checkpoint)
    volume = np.random.rand(32, 32, 32).astype(np.float32)
    mask = backend.segment(volume)
    assert mask.shape == volume.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
