"""Optional ML segmentation backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fiber_tracer.backends.base import SegmentationBackend
from fiber_tracer.exceptions import BackendNotAvailableError

if TYPE_CHECKING:
    from fiber_tracer.backends.unet3d import UNet3D


class MLSegmentationBackend(SegmentationBackend):
    """Segmentation backend that lazy-loads PyTorch and a 3D U-Net checkpoint.

    The backend does not ship a trained model. Train one with
    ``scripts/train_unet_phantoms.py`` and pass the checkpoint path to
    ``model_path``.
    """

    def __init__(self, model_path: str | None = None, batch_size: int = 1):
        try:
            import torch
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Install ml extra: pip install fiber-tracer[ml]"
            ) from exc
        self.torch = torch
        self.model_path = model_path
        self.batch_size = batch_size
        self.model: UNet3D | None = None
        self._checkpoint: dict | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> MLSegmentationBackend:
        """Create a backend and load the checkpoint at *path*."""
        backend = cls(model_path=str(path))
        backend._load_model()
        return backend

    def _load_model(self) -> None:
        """Load the PyTorch checkpoint and rebuild the U-Net architecture."""
        if self.model is not None:
            return
        if self.model_path is None:
            raise RuntimeError(
                "No model checkpoint configured. Train a model with "
                "`python scripts/train_unet_phantoms.py` and set "
                "`segmentation.model_path` to the resulting .pt file."
            )
        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. "
                "Train one with `python scripts/train_unet_phantoms.py`."
            )

        from fiber_tracer.backends.unet3d import UNet3D

        checkpoint = self.torch.load(path, map_location="cpu", weights_only=True)
        features = checkpoint.get("features", (8, 16, 32))
        net = UNet3D(in_channels=1, out_channels=1, features=features)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        self.model = net
        self._checkpoint = checkpoint

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Return a binary segmentation mask for the input volume.

        Parameters
        ----------
        volume : np.ndarray
            3D grayscale volume. It is normalized internally before inference.

        Returns
        -------
        np.ndarray
            Binary uint8 mask with the same shape as *volume*.
        """
        self._load_model()
        assert self.model is not None
        if self._checkpoint:
            patch_size = self._checkpoint.get("patch_size", (32, 32, 32))
        else:
            patch_size = (32, 32, 32)
        prob: np.ndarray = self.model.predict_volume(
            volume, patch_size=patch_size, overlap=16, batch_size=self.batch_size
        )
        return (prob > 0.5).astype(np.uint8)
