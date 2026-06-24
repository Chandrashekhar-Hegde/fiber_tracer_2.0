"""Backend adapter base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SegmentationBackend(ABC):
    """Abstract base class for optional segmentation backends."""

    @abstractmethod
    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Return a binary or labeled segmentation for the input volume."""
        ...
