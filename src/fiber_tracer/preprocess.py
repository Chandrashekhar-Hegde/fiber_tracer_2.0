"""Preprocessing: denoising, normalization, anisotropy handling."""

import numpy as np
from scipy import ndimage

from fiber_tracer.config import VoxelSpacing


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = volume.min(), volume.max()
    if vmax == vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return np.asarray((volume - vmin) / (vmax - vmin), dtype=np.float32)


def gaussian_denoise(
    volume: np.ndarray, sigma_um: float, voxel_spacing: VoxelSpacing
) -> np.ndarray:
    """3D Gaussian denoising with physical sigma in micrometers."""
    sigma_voxels = (
        sigma_um / voxel_spacing.z,
        sigma_um / voxel_spacing.y,
        sigma_um / voxel_spacing.x,
    )
    return np.asarray(
        ndimage.gaussian_filter(volume.astype(np.float32, copy=False), sigma=sigma_voxels),
        dtype=np.float32,
    )


def resample_to_isotropic(
    volume: np.ndarray,
    voxel_spacing: VoxelSpacing,
    order: int = 1,
) -> tuple[np.ndarray, VoxelSpacing]:
    """Resample anisotropic volume to isotropic voxels at the smallest spacing."""
    target = min(voxel_spacing.z, voxel_spacing.y, voxel_spacing.x)
    zoom = (
        voxel_spacing.z / target,
        voxel_spacing.y / target,
        voxel_spacing.x / target,
    )
    resampled = ndimage.zoom(volume.astype(np.float32, copy=False), zoom, order=order)
    return resampled, VoxelSpacing(target, target, target)
