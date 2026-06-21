import numpy as np
import pytest

from fiber_tracer.config import VoxelSpacing
from fiber_tracer.preprocess import (
    gaussian_denoise,
    normalize_intensity,
    resample_to_isotropic,
)


def test_normalize_intensity_maps_to_zero_one_and_preserves_shape():
    volume = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    normalized = normalize_intensity(volume)
    assert normalized.shape == volume.shape
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_normalize_intensity_returns_zeros_for_constant_input():
    volume = np.full((4, 4, 4), 5.0, dtype=np.float32)
    normalized = normalize_intensity(volume)
    assert normalized.shape == volume.shape
    np.testing.assert_array_equal(normalized, np.zeros_like(volume, dtype=float))


def test_gaussian_denoise_preserves_shape_and_reduces_noise():
    rng = np.random.default_rng(42)
    signal = np.ones((16, 16, 16), dtype=np.float32) * 10.0
    noise = rng.standard_normal(signal.shape).astype(np.float32)
    volume = signal + noise
    spacing = VoxelSpacing(1.0, 1.0, 1.0)
    denoised = gaussian_denoise(volume, sigma_um=2.0, voxel_spacing=spacing)
    assert denoised.shape == volume.shape
    assert denoised.std() < volume.std()


def test_resample_to_isotropic_reduces_spacing_and_changes_shape():
    volume = np.zeros((10, 20, 30), dtype=np.float32)
    spacing = VoxelSpacing(2.0, 1.0, 1.5)
    resampled, new_spacing = resample_to_isotropic(volume, spacing)
    assert new_spacing.is_isotropic()
    assert new_spacing.x == pytest.approx(1.0)
    assert new_spacing.y == pytest.approx(1.0)
    assert new_spacing.z == pytest.approx(1.0)
    expected_shape = (
        int(round(volume.shape[0] * spacing.z / new_spacing.z)),
        int(round(volume.shape[1] * spacing.y / new_spacing.y)),
        int(round(volume.shape[2] * spacing.x / new_spacing.x)),
    )
    assert resampled.shape == expected_shape
