"""Tests for chunked / out-of-core processing helpers."""

import numpy as np
import pytest

from fiber_tracer.chunked import (
    gaussian_denoise_chunked,
    load_zarr,
    normalize_intensity_chunked,
    process_chunks,
    save_zarr,
)


@pytest.fixture
def tmp_zarr_path(tmp_path):
    return tmp_path / "test.zarr"


def test_save_and_load_zarr_round_trip(tmp_zarr_path):
    data = np.random.rand(16, 16, 16).astype(np.float32)
    save_zarr(tmp_zarr_path, data, chunks=(8, 8, 8))
    loaded = load_zarr(tmp_zarr_path)
    np.testing.assert_array_equal(data, loaded[:])


def test_process_chunks_identity():
    data = np.arange(64).reshape((4, 4, 4)).astype(np.float32)
    output = np.zeros_like(data)
    process_chunks(data, output, lambda x: x, chunk_shape=(2, 2, 2))
    np.testing.assert_array_equal(data, output)


def test_process_chunks_with_overlap():
    data = np.zeros((24, 24, 24), dtype=np.float32)
    output = np.zeros_like(data)
    process_chunks(data, output, lambda x: x, chunk_shape=(8, 8, 8), overlap=2)
    np.testing.assert_array_equal(data, output)


def test_normalize_intensity_chunked():
    data = np.linspace(0, 255, 64).reshape((4, 4, 4)).astype(np.uint8)
    output = np.zeros((4, 4, 4), dtype=np.float32)
    normalize_intensity_chunked(data, output, chunk_shape=(2, 2, 2))
    assert output.min() == pytest.approx(0.0, abs=1e-6)
    assert output.max() == pytest.approx(1.0, abs=1e-6)


def test_gaussian_denoise_chunked_preserves_shape():
    data = np.random.rand(8, 8, 8).astype(np.float32)
    output = np.zeros_like(data)
    gaussian_denoise_chunked(data, output, sigma_voxels=(1.0, 1.0, 1.0), chunk_shape=(4, 4, 4))
    assert output.shape == data.shape
