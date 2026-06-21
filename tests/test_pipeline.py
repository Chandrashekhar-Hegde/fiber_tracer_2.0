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


def test_auto_regime_selects_marginal_and_subvoxel_by_ratio(tmp_path):
    """Auto regime maps physical voxel/fiber ratio to marginal or subvoxel."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Marginal: ratio = 1.0 / 2.0 = 0.5 (between 0.3 and 3.0).
    phantom_marginal = generate_fiber_phantom(
        shape=(48, 48, 48),
        n_fibers=3,
        fiber_diameter_um=2.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path_marginal = data_dir / "marginal_input.tif"
    save_tiff_stack(stack_path_marginal, phantom_marginal.volume)

    config_marginal = Config(
        data_path=str(stack_path_marginal),
        output_dir=str(tmp_path / "out_marginal"),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=2.0,
        regime="auto",
    )
    summary_marginal = FiberAnalysisPipeline(config_marginal).run()
    assert summary_marginal["regime"] == "marginal"
    assert "a2_map" in summary_marginal

    # Subvoxel: ratio = 5.0 / 1.0 = 5.0 (> 3.0).
    phantom_subvoxel = generate_fiber_phantom(
        shape=(32, 32, 32),
        n_fibers=3,
        fiber_diameter_um=1.0,
        voxel_spacing_um=(5.0, 5.0, 5.0),
        seed=43,
    )
    stack_path_subvoxel = data_dir / "subvoxel_input.tif"
    save_tiff_stack(stack_path_subvoxel, phantom_subvoxel.volume)

    config_subvoxel = Config(
        data_path=str(stack_path_subvoxel),
        output_dir=str(tmp_path / "out_subvoxel"),
        voxel_spacing_um=VoxelSpacing(5.0, 5.0, 5.0),
        fiber_diameter_um=1.0,
        regime="auto",
    )
    summary_subvoxel = FiberAnalysisPipeline(config_subvoxel).run()
    assert summary_subvoxel["regime"] == "subvoxel"
    assert "a2" in summary_subvoxel
