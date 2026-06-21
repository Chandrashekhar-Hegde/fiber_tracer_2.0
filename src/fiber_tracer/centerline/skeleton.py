"""Skeletonization and graph analysis using scikit-image and skan."""

from typing import List
import numpy as np
from skimage.morphology import skeletonize_3d


def skeletonize_label_volume(labels: np.ndarray) -> np.ndarray:
    """Skeletonize each labeled fiber separately to avoid bridging."""
    skeleton = np.zeros_like(labels, dtype=bool)
    for label in np.unique(labels)[1:]:
        mask = labels == label
        skel = skeletonize_3d(mask)
        skeleton |= skel
    return skeleton
