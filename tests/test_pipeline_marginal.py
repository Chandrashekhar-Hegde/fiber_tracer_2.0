"""Tests for the marginal-regime pipeline."""

import numpy as np

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def test_pipeline_marginal_creates_a2_map_and_centers(tmp_path):
    """Marginal regime writes a2_map.npy and a2_centers.npy."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(64, 64, 64),
        n_fibers=3,
        fiber_diameter_um=1.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=1.0,
        regime="marginal",
    )

    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()

    assert summary["regime"] == "marginal"
    assert "a2_map" in summary
    assert "a2_map_shape" in summary
    assert (out_dir / "a2_map.npy").exists()
    assert (out_dir / "a2_centers.npy").exists()


def test_pipeline_marginal_constant_volume_returns_gracefully(tmp_path):
    """A constant (fiber-less) volume yields a zero-voxel marginal summary."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    stack_path = data_dir / "constant.tif"
    save_tiff_stack(stack_path, np.full((32, 32, 32), 128, dtype=np.uint8))

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=1.0,
        regime="marginal",
    )

    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()

    assert summary["regime"] == "marginal"
    assert summary["n_voxels"] == 0
    assert (out_dir / "a2_map.npy").exists()
    assert (out_dir / "a2_centers.npy").exists()
