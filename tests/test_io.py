# tests/test_io.py

import numpy as np
import pytest

from fiber_tracer.io import (
    estimate_volume_fraction,
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


def test_load_tiff_stack_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.tif"
    with pytest.raises(FileNotFoundError):
        load_tiff_stack(missing)


def test_get_shape_info():
    volume = np.zeros((10, 20, 30), dtype=np.uint16)
    spacing = (2.0, 1.0, 1.0)
    info = get_shape_info(volume, spacing)
    assert info["shape_voxels"] == (10, 20, 30)
    assert info["shape_um"] == (20.0, 20.0, 30.0)
    assert info["voxel_spacing_um"] == spacing
    assert info["dtype"] == "uint16"
    assert info["size_gb"] == volume.nbytes / (1024**3)


def test_get_shape_info_2d_raises():
    array = np.zeros((10, 20), dtype=np.uint16)
    with pytest.raises(ValueError, match="Expected 3D volume, got 2D"):
        get_shape_info(array, (1.0, 1.0, 1.0))


def test_save_tiff_stack_round_trip_uint16(tmp_path):
    volume = np.random.randint(0, 4096, size=(4, 6, 8)).astype(np.uint16)
    path = tmp_path / "round_trip_uint16.tif"
    save_tiff_stack(path, volume)
    loaded = load_tiff_stack(path)
    assert loaded.shape == volume.shape
    np.testing.assert_array_equal(loaded, volume)


def test_save_tiff_stack_round_trip_uint8(tmp_path):
    volume = np.random.randint(0, 256, size=(4, 6, 8)).astype(np.uint8)
    path = tmp_path / "round_trip_uint8.tif"
    save_tiff_stack(path, volume)
    loaded = load_tiff_stack(path)
    assert loaded.shape == volume.shape
    np.testing.assert_array_equal(loaded, volume)


def test_save_tiff_stack_round_trip_float32(tmp_path):
    volume = np.random.rand(4, 6, 8).astype(np.float32)
    path = tmp_path / "round_trip_float32.tif"
    save_tiff_stack(path, volume)
    loaded = load_tiff_stack(path)
    assert loaded.shape == volume.shape
    np.testing.assert_allclose(loaded, volume)


def test_save_tiff_stack_negative_int_raises(tmp_path):
    volume = np.array([-1, 0, 1], dtype=np.int16).reshape((1, 1, 3))
    path = tmp_path / "negative.tif"
    with pytest.raises(ValueError, match="Negative intensities cannot be safely written"):
        save_tiff_stack(path, volume)


def test_load_tiff_stack_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_tiff_stack(empty_dir)


def test_estimate_volume_fraction_known_threshold():
    volume = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Default threshold of 0.5: values strictly greater than 0.5 are 0.75 and 1.0.
    assert estimate_volume_fraction(volume) == 0.4
    assert estimate_volume_fraction(volume, threshold=0.25) == 0.6
