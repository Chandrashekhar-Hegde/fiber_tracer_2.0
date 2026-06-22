# tests/test_pipeline_backends.py
from unittest.mock import patch

import pytest

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
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


def test_invalid_segmentation_method_raises():
    config = Config()
    config.segmentation.method = "invalid"
    with pytest.raises(ValueError):
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

    pipeline = FiberAnalysisPipeline(config)
    with patch("fiber_tracer.pipeline.betti_numbers", return_value=expected_betti) as mock_betti:
        with patch(
            "fiber_tracer.pipeline.persistence_summary",
            return_value=expected_persistence,
        ) as mock_persistence:
            summary = pipeline.run()

    mock_betti.assert_called_once()
    mock_persistence.assert_called_once()
    assert summary["tda"]["betti_numbers"] == expected_betti
    assert summary["tda"]["persistence_summary"] == expected_persistence
