"""Tests for the subvoxel-regime pipeline."""

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def test_pipeline_subvoxel_returns_global_tensor_and_fa(tmp_path):
    """Subvoxel regime returns a global A2 tensor, FA, and orientation distribution."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(48, 48, 48),
        n_fibers=5,
        fiber_diameter_um=1.0,
        voxel_spacing_um=(5.0, 5.0, 5.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(5.0, 5.0, 5.0),
        fiber_diameter_um=1.0,
        regime="subvoxel",
    )

    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()

    assert summary["regime"] == "subvoxel"
    assert "a2" in summary
    assert "fa" in summary
    assert "orientation_distribution" in summary
    assert isinstance(summary["fa"], float)
    assert 0.0 <= summary["fa"] <= 1.0
