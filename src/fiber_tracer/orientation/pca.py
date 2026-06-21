"""PCA-based orientation from voxel coordinates."""

import numpy as np
from scipy import linalg


def pca_orientation(coords: np.ndarray) -> np.ndarray:
    """Return principal axis from voxel coordinates."""
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    return axis / np.linalg.norm(axis)
