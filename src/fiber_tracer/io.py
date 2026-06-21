"""I/O for TIFF stacks, HDF5, and metadata."""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def load_tiff_stack(path: Union[str, Path]) -> np.ndarray:
    """Load a TIFF stack from a file or directory of TIFFs."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.tif*"))
        if not files:
            raise FileNotFoundError(f"No TIFF files found in {path}")
        logger.info(f"Loading {len(files)} TIFF slices from {path}")
        return tifffile.imread(files)
    return tifffile.imread(path)


def estimate_volume_fraction(volume: np.ndarray, threshold: float = 0.5) -> float:
    """Quick estimate of foreground volume fraction from normalized volume."""
    return float(np.mean(volume > threshold))


def get_shape_info(volume: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> dict:
    """Return human-readable shape and physical size info."""
    dz, dy, dx = voxel_spacing
    z, y, x = volume.shape
    return {
        "shape_voxels": (z, y, x),
        "shape_um": (z * dz, y * dy, x * dx),
        "voxel_spacing_um": (dz, dy, dx),
        "dtype": str(volume.dtype),
        "size_gb": volume.nbytes / (1024**3),
    }


def _safe_to_writable_dtype(volume: np.ndarray) -> np.ndarray:
    """Convert an array to a dtype suitable for TIFF writing.

    uint8, uint16, and float32 are passed through.  Other integer dtypes are
    cast to uint16 when values fit, otherwise to float32.  Other float dtypes
    are assumed to be normalized in [0, 1] and converted to float32.
    """
    if volume.dtype in (np.uint8, np.uint16, np.float32):
        return volume

    if np.issubdtype(volume.dtype, np.integer):
        if volume.min() >= 0 and volume.max() <= np.iinfo(np.uint16).max:
            return volume.astype(np.uint16)
        # Values do not fit in uint16; fall back to float32.
        scaled = volume.astype(np.float32)
        max_val = scaled.max()
        if max_val > 0:
            scaled = scaled / max_val
        return scaled

    if np.issubdtype(volume.dtype, np.floating):
        return volume.astype(np.float32)

    raise TypeError(f"Unsupported dtype for TIFF writing: {volume.dtype}")


def save_tiff_stack(path: Union[str, Path], volume: np.ndarray) -> None:
    """Write a 3D volume to a single TIFF file.

    The volume is converted to a TIFF-compatible dtype and written with
    ImageJ-compatible metadata when the volume is 3D and the dtype is one of
    uint8, uint16, or float32.
    """
    path = Path(path)
    writable = _safe_to_writable_dtype(volume)

    kwargs = {}
    if writable.ndim == 3 and writable.dtype in (np.uint8, np.uint16, np.float32):
        kwargs["imagej"] = True

    tifffile.imwrite(path, writable, **kwargs)
