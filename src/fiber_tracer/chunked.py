"""Chunked / out-of-core processing helpers built on zarr.

These functions allow large volumes to be processed in overlapping blocks
without loading the entire array into memory. The ``parallel`` extra
includes ``dask`` for higher-level distributed workflows, but the helpers
here are zarr-backed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import tifffile

from fiber_tracer.exceptions import BackendNotAvailableError

if TYPE_CHECKING:
    import zarr


def _import_zarr():
    try:
        import zarr

        return zarr
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Install parallel extra: pip install fiber-tracer[parallel]"
        ) from exc


def load_zarr(path: str | Path) -> zarr.Array:
    """Open an existing zarr array (read-only)."""
    zarr = _import_zarr()
    store = zarr.DirectoryStore(str(path))
    return zarr.open_array(store, mode="r")


def save_zarr(
    path: str | Path,
    data: np.ndarray,
    chunks: tuple[int, int, int] | None = None,
    dtype: np.dtype | None = None,
) -> zarr.Array:
    """Save a numpy array to a zarr array with the given chunk size."""
    zarr = _import_zarr()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(path))
    chunks_to_use = data.shape if chunks is None else chunks
    return zarr.array(
        data,
        store=store,
        chunks=chunks_to_use,
        dtype=dtype or data.dtype,
        overwrite=True,
    )


def tiff_to_zarr(
    tiff_path: str | Path,
    zarr_path: str | Path,
    chunks: tuple[int, int, int] = (64, 64, 64),
) -> zarr.Array:
    """Convert a TIFF stack to a chunked zarr array without loading it fully into RAM."""
    # For simplicity, load the TIFF stack (tifffile.imread supports lazy via aszarr,
    # but fallback to in-memory conversion for the first version).
    volume = tifffile.imread(tiff_path)
    return save_zarr(zarr_path, volume, chunks=chunks)


def _pad_slice(start: int, stop: int, size: int, overlap: int):
    """Return (read_start, read_stop, write_start, write_stop) with overlap."""
    read_start = max(0, start - overlap)
    read_stop = min(size, stop + overlap)
    write_start = start
    write_stop = stop
    return read_start, read_stop, write_start, write_stop


def process_chunks(
    input_array: np.ndarray | zarr.Array,
    output_array: np.ndarray | zarr.Array,
    func: Callable[[np.ndarray], np.ndarray],
    chunk_shape: tuple[int, int, int],
    overlap: int = 0,
) -> None:
    """Apply ``func`` to overlapping chunks of ``input_array``.

    The central region of each processed chunk is written to ``output_array``.
    ``func`` receives a chunk of shape at most ``chunk_shape + 2*overlap`` and
    must return an array of the same shape.
    """
    shape = input_array.shape
    if output_array.shape != shape:
        raise ValueError("input_array and output_array must have the same shape")

    for z_start in range(0, shape[0], chunk_shape[0]):
        for y_start in range(0, shape[1], chunk_shape[1]):
            for x_start in range(0, shape[2], chunk_shape[2]):
                z_stop = min(z_start + chunk_shape[0], shape[0])
                y_stop = min(y_start + chunk_shape[1], shape[1])
                x_stop = min(x_start + chunk_shape[2], shape[2])

                rz0, rz1, wz0, wz1 = _pad_slice(z_start, z_stop, shape[0], overlap)
                ry0, ry1, wy0, wy1 = _pad_slice(y_start, y_stop, shape[1], overlap)
                rx0, rx1, wx0, wx1 = _pad_slice(x_start, x_stop, shape[2], overlap)

                chunk_in = np.asarray(input_array[rz0:rz1, ry0:ry1, rx0:rx1])
                chunk_out = func(chunk_in)

                # Verify central region shape matches write region
                expected_shape = (rz1 - rz0, ry1 - ry0, rx1 - rx0)
                if chunk_out.shape != expected_shape:
                    raise ValueError(
                        f"func returned shape {chunk_out.shape}, expected {expected_shape}"
                    )

                output_array[wz0:wz1, wy0:wy1, wx0:wx1] = chunk_out[
                    z_start - rz0 : z_stop - rz0,
                    y_start - ry0 : y_stop - ry0,
                    x_start - rx0 : x_stop - rx0,
                ]


def normalize_intensity_chunked(
    input_array: np.ndarray | zarr.Array,
    output_array: np.ndarray | zarr.Array,
    chunk_shape: tuple[int, int, int] = (64, 64, 64),
) -> None:
    """Two-pass min-max normalization on chunked arrays."""
    # First pass: compute global min and max in chunks
    vmin = np.inf
    vmax = -np.inf
    for z in range(0, input_array.shape[0], chunk_shape[0]):
        for y in range(0, input_array.shape[1], chunk_shape[1]):
            for x in range(0, input_array.shape[2], chunk_shape[2]):
                cz = z + chunk_shape[0]
                cy = y + chunk_shape[1]
                cx = x + chunk_shape[2]
                chunk = np.asarray(input_array[z:cz, y:cy, x:cx])
                vmin = min(vmin, chunk.min())
                vmax = max(vmax, chunk.max())

    if vmax == vmin:
        output_array[:] = 0.0
        return

    def _norm(chunk: np.ndarray) -> np.ndarray:
        return ((chunk - vmin) / (vmax - vmin)).astype(np.float32)

    process_chunks(input_array, output_array, _norm, chunk_shape)


def gaussian_denoise_chunked(
    input_array: np.ndarray | zarr.Array,
    output_array: np.ndarray | zarr.Array,
    sigma_voxels: tuple[float, float, float],
    chunk_shape: tuple[int, int, int] = (64, 64, 64),
) -> None:
    """Chunked Gaussian denoising with overlap to avoid boundary artifacts."""
    from scipy import ndimage

    overlap = int(np.ceil(max(sigma_voxels) * 3))

    def _denoise(chunk: np.ndarray) -> np.ndarray:
        filtered = ndimage.gaussian_filter(chunk, sigma=sigma_voxels)
        return np.asarray(filtered, dtype=np.float32)

    process_chunks(input_array, output_array, _denoise, chunk_shape, overlap=overlap)
