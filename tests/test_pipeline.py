import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def test_pipeline_creates_outputs_for_resolved_regime(tmp_path):
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

    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "labels.tif").exists()
    assert (out_dir / "skeleton.tif").exists()
    assert (out_dir / "normalized_input.tif").exists()
    assert summary["n_labels"] > 0
    assert summary["regime"] == "resolved"
    assert len(summary["fibers"]) == summary["n_labels"]

    with open(out_dir / "summary.json") as f:
        disk_summary = json.load(f)
    assert disk_summary["n_labels"] == summary["n_labels"]


def test_auto_regime_uses_detect_regime(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(32, 32, 32),
        n_fibers=2,
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
        regime="auto",
    )

    with patch("fiber_tracer.pipeline.detect_regime") as mock_detect:
        mock_detect.return_value = "resolved"
        pipeline = FiberAnalysisPipeline(config)
        summary = pipeline.run()

    mock_detect.assert_called_once_with(config)
    assert summary["regime"] == "resolved"


def test_invalid_regime_raises_during_validation(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(32, 32, 32),
        n_fibers=1,
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
        regime="invalid",
    )

    with pytest.raises(ValueError):
        config.validate()
