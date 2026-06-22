# tests/test_pipeline_backends.py
import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import load_tiff_stack, save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def test_unet_method_routes_to_ml_backend(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(64, 64, 64),
        n_fibers=3,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=4.0,
        regime="resolved",
    )
    config.segmentation.method = "unet"

    pipeline = FiberAnalysisPipeline(config)
    with patch("fiber_tracer.pipeline.MLSegmentationBackend") as mock_backend_cls:
        mock_backend_cls.return_value.segment.return_value = phantom.labels > 0
        summary = pipeline.run()

    mock_backend_cls.assert_called_once_with(model_path=None)
    mock_backend_cls.return_value.segment.assert_called_once()
    assert summary["n_labels"] > 0
    assert summary["regime"] == "resolved"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_unet_pipeline_with_real_checkpoint(tmp_path):
    import torch

    from fiber_tracer.backends.unet3d import UNet3D

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(64, 64, 64),
        n_fibers=3,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    checkpoint = tmp_path / "model.pt"
    model = UNet3D(in_channels=1, out_channels=1, features=(4, 8, 16))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": (4, 8, 16),
            "patch_size": (32, 32, 32),
        },
        checkpoint,
    )

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=4.0,
        regime="resolved",
    )
    config.segmentation.method = "unet"
    config.segmentation.model_path = str(checkpoint)

    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()
    assert summary["regime"] == "resolved"
    assert (out_dir / "labels.tif").exists()
    assert (out_dir / "summary.json").exists()


def test_invalid_segmentation_method_raises(tmp_path):
    config = Config()
    config.data_path = str(tmp_path / "data")
    config.output_dir = str(tmp_path / "out")
    config.segmentation.method = "invalid"
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="segmentation.method"):
        config.validate()


def test_tda_descriptors_added_to_summary(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(64, 64, 64),
        n_fibers=3,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=4.0,
        regime="resolved",
    )
    config.analysis.compute_tda_descriptors = True

    expected_betti = {"b0": 3, "b1": 0, "b2": 0}
    expected_persistence = {"n_features": 3, "n_finite": 0, "max_persistence": 0.0}

    known_mask = np.zeros((64, 64, 64), dtype=bool)
    known_mask[10:20, 10:20, 10:20] = True
    known_mask[30:40, 30:40, 30:40] = True

    pipeline = FiberAnalysisPipeline(config)
    with patch("fiber_tracer.pipeline.segment_otsu_3d", return_value=known_mask) as mock_otsu:
        with patch(
            "fiber_tracer.pipeline.betti_numbers", return_value=expected_betti
        ) as mock_betti:
            with patch(
                "fiber_tracer.pipeline.persistence_summary",
                return_value=expected_persistence,
            ) as mock_persistence:
                summary = pipeline.run()

    mock_otsu.assert_called_once()
    mock_betti.assert_called_once()
    mock_persistence.assert_called_once()
    assert summary["tda"]["betti_numbers"] == expected_betti
    assert summary["tda"]["persistence_summary"] == expected_persistence

    # TDA descriptors must be computed on the cleaned mask (labels > 0),
    # not on the raw Otsu mask.
    labels = load_tiff_stack(out_dir / "labels.tif")
    cleaned_mask = labels > 0
    passed_mask = mock_betti.call_args[0][0]
    assert passed_mask.dtype == bool
    np.testing.assert_array_equal(passed_mask, cleaned_mask)
