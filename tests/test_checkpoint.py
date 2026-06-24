import tempfile
from pathlib import Path

import pytest
import torch

from fiber_tracer.backends.unet3d import UNet3D
from fiber_tracer.training.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint():
    model = UNet3D(features=(8, 16))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, metadata={"epoch": 2})
        loaded = load_checkpoint(path)
        assert loaded["metadata"]["epoch"] == 2
        assert "model_state_dict" in loaded


def test_state_dict_round_trip():
    model = UNet3D(features=(8, 16))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model)
        loaded = load_checkpoint(path)

        fresh_model = UNet3D(features=(8, 16))
        fresh_model.load_state_dict(loaded["model_state_dict"])

        for param, fresh_param in zip(model.parameters(), fresh_model.parameters(), strict=True):
            assert torch.equal(param, fresh_param)


def test_metadata_defaults_to_empty_dict():
    model = UNet3D(features=(8, 16))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, metadata=None)
        loaded = load_checkpoint(path)
        assert loaded["metadata"] == {}


def test_load_checkpoint_missing_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "missing.pt"
        with pytest.raises(FileNotFoundError):
            load_checkpoint(path)
