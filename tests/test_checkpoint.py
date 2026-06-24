import tempfile
from pathlib import Path

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
