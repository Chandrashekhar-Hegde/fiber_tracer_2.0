"""Classical segmentation: thresholding and watershed separation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def segment_otsu_3d(volume: np.ndarray) -> np.ndarray:
    """Global 3D Otsu thresholding."""
    threshold = filters.threshold_otsu(volume)
    return np.asarray(volume > threshold, dtype=bool)


def binarize_volume(
    volume: np.ndarray,
    method: str = "otsu",
    threshold_value: float | None = None,
    adaptive_block_size: int = 51,
    adaptive_offset: float = 0.02,
    multiotsu_classes: int = 3,
) -> np.ndarray:
    """Binarize *volume* into a foreground mask using the selected method.

    Parameters
    ----------
    volume:
        Input intensity volume.
    method:
        ``"otsu"`` (global Otsu), ``"manual"`` (fixed ``threshold_value``),
        ``"adaptive"`` (local-mean / Bradley thresholding over a cubic window of
        ``adaptive_block_size`` voxels, requiring intensities to exceed the local
        mean by ``adaptive_offset``), or ``"multiotsu"`` (multi-level Otsu keeping
        the brightest class).

    Returns
    -------
    np.ndarray
        Boolean foreground mask with the same shape as *volume*.
    """
    if method == "otsu":
        return segment_otsu_3d(volume)
    if method == "manual":
        if threshold_value is None:
            raise ValueError("threshold_value is required for method='manual'")
        return np.asarray(volume > threshold_value, dtype=bool)
    if method == "adaptive":
        local_mean = ndimage.uniform_filter(volume.astype(float), size=adaptive_block_size)
        return np.asarray(volume > local_mean + adaptive_offset, dtype=bool)
    if method == "multiotsu":
        thresholds = filters.threshold_multiotsu(volume, classes=multiotsu_classes)
        return np.asarray(volume > thresholds[-1], dtype=bool)
    raise ValueError(
        f"unknown threshold method {method!r}; expected one of "
        "'otsu', 'manual', 'adaptive', 'multiotsu'"
    )


def segment_connected_components_3d(foreground_mask: np.ndarray) -> np.ndarray:
    """Label each connected component of the foreground mask with a unique ID.

    This is useful for the resolved regime when fibers are already well
    separated; it avoids the over-segmentation that distance-transform
    watershed can produce on elongated objects.
    """
    labels, _ = ndimage.label(foreground_mask)
    return np.asarray(labels, dtype=np.int32)


def segment_watershed_3d(
    foreground_mask: np.ndarray,
    min_distance_voxels: int = 3,
) -> np.ndarray:
    """3D marker-controlled watershed on distance transform.

    Parameters
    ----------
    foreground_mask : np.ndarray
        Binary 3D mask of the foreground object(s).
    min_distance_voxels : int, optional
        Minimum distance (in voxels) between watershed markers, by default 3.

    Returns
    -------
    np.ndarray
        Label image with separated foreground objects.
    """
    distance = ndimage.distance_transform_edt(foreground_mask)
    peak_coords = peak_local_max(
        distance,
        min_distance=min_distance_voxels,
        exclude_border=False,
    )
    local_max = np.zeros_like(distance, dtype=bool)
    local_max[tuple(peak_coords.T)] = True
    markers = ndimage.label(local_max)[0]
    labels = watershed(-distance, markers, mask=foreground_mask)
    return np.asarray(labels, dtype=np.int32)


def remove_small_objects(labels: np.ndarray, min_size_voxels: int) -> np.ndarray:
    """Remove connected components below minimum size."""
    return np.asarray(
        morphology.remove_small_objects(labels, min_size=min_size_voxels), dtype=labels.dtype
    )
