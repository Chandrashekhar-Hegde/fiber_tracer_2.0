# tests/test_io.py

import numpy as np
import pytest

from fiber_tracer.io import (
    get_shape_info,
    load_tiff_stack,
    save_tiff_stack,
)


def test_load_tiff_stack_from_dir(tmp_path):
    volume = np.random.rand(5, 8, 8).astype(np.float32)
    for i, slice_ in enumerate(volume):
        save_tiff_stack(tmp_path / f"slice_{i:03d}.tif", slice_[np.newaxis, ...])
    loaded = load_tiff_stack(tmp_path)
    assert loaded.shape == volume.shape


def test_load_tiff_stack_from_single_file(tmp_path):
    volume = np.random.rand(5, 8, 8).astype(np.float32)
    path = tmp_path / "stack.tif"
    save_tiff_stack(path, volume)
    loaded = load_tiff_stack(path)
    assert loaded.shape == volume.shape


def test_get_shape_info():
    volume = np.zeros((10, 20, 30), dtype=np.uint16)
    spacing = (2.0, 1.0, 1.0)
    info = get_shape_info(volume, spacing)
    assert info["shape_voxels"] == (10, 20, 30)
    assert info["shape_um"] == (20.0, 20.0, 30.0)
    assert info["voxel_spacing_um"] == spacing
    assert info["dtype"] == "uint16"
    assert info["size_gb"] == volume.nbytes / (1024**3)


def test_save_tiff_stack_round_trip(tmp_path):
    volume = np.random.randint(0, 4096, size=(4, 6, 8)).astype(np.uint16)
    path = tmp_path / "round_trip.tif"
    save_tiff_stack(path, volume)
    loaded = load_tiff_stack(path)
    assert loaded.shape == volume.shape
    np.testing.assert_array_equal(loaded, volume)


def test_load_tiff_stack_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_tiff_stack(empty_dir)
