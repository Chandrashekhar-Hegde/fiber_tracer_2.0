"""Optional ML segmentation backend."""

from typing import Callable, Optional

import numpy as np

from fiber_tracer.backends.base import SegmentationBackend
from fiber_tracer.exceptions import BackendNotAvailableError


class MLSegmentationBackend(SegmentationBackend):
    """Segmentation backend that lazy-loads PyTorch.

    This adapter does not ship a trained model. Users must either subclass it
    or load a checkpoint before calling `segment`.
    """

    def __init__(self, model_path: Optional[str] = None):
        try:
            import torch
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Install ml extra: pip install fiber-tracer[ml]"
            ) from exc
        self.torch = torch
        self.model_path = model_path
        self.model: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Raise NotImplementedError until a model is loaded."""
        if self.model is None:
            raise NotImplementedError(
                "No model is loaded. Subclass MLSegmentationBackend, load a checkpoint, "
                "or implement a model before calling segment()."
            )
        return np.asarray(self.model(volume))
