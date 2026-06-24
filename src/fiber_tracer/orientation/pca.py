"""PCA-based orientation from voxel coordinates."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def pca_orientation(coords: np.ndarray) -> np.ndarray:
    """Return principal axis from voxel coordinates."""
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be an Nx3 array")
    if len(coords) == 0:
        raise ValueError("coords must contain at least one point")
    if len(coords) == 1:
        return np.array([0.0, 0.0, 1.0])
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.eye(3) * np.maximum(cov, 1e-12)
    evals, evecs = linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(axis / norm, dtype=np.float64)
