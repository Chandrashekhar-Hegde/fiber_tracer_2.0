"""3D augmentations for fiber segmentation training patches."""

from __future__ import annotations

import random

import numpy as np


def augment_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random flips, rotations, gamma, and noise to a 3D patch.

    Both *volume* and *mask* must have the same shape. The returned arrays
    are contiguous copies.
    """
    rng = random.Random(seed)

    # Random axis flips.
    for axis in range(3):
        if rng.random() > 0.5:
            volume = np.flip(volume, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

    # Random 90-degree rotations around the z-axis.
    k = rng.randint(0, 3)
    if k:
        volume = np.rot90(volume, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

    # Random gamma intensity scaling.
    if rng.random() > 0.5:
        gamma = rng.uniform(0.8, 1.2)
        volume = np.clip(volume**gamma, 0.0, 1.0)

    # Random additive Gaussian noise.
    if rng.random() > 0.5:
        noise = rng.gauss(0.0, 0.02)
        volume = np.clip(volume + noise, 0.0, 1.0)

    return volume, mask
