import json
from unittest.mock import patch

import pytest

from fiber_tracer.cli import main
from fiber_tracer.config import Config, DVCConfig, VoxelSpacing
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


def test_resolved_pipeline_respects_analysis_flags(tmp_path):
    """Resolved pipeline skips per-fiber morphometry/orientation when flags are False."""
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
    config.analysis.compute_morphometry = False
    config.analysis.compute_orientation_tensor = False

    summary = FiberAnalysisPipeline(config).run()
    assert summary["n_labels"] > 0
    for fiber in summary["fibers"]:
        assert "equivalent_diameter_um" not in fiber
        assert "orientation" not in fiber


def test_resolved_pipeline_reports_tracking_metrics(tmp_path):
    """Resolved pipeline reports per-fiber centerline length and tortuosity."""
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

    summary = FiberAnalysisPipeline(config).run()
    assert summary["n_labels"] > 0
    for fiber in summary["fibers"]:
        assert "length_um" in fiber
        assert "tortuosity" in fiber
        assert fiber["length_um"] >= 0.0
        assert fiber["tortuosity"] >= 1.0 - 1e-6


def test_tracking_can_be_disabled(tmp_path):
    """Setting analysis.compute_tracking=False omits centerline metrics."""
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
    config.analysis.compute_tracking = False

    summary = FiberAnalysisPipeline(config).run()
    assert summary["n_labels"] > 0
    for fiber in summary["fibers"]:
        assert "length_um" not in fiber
        assert "tortuosity" not in fiber


def test_resolved_pipeline_respects_normalize_flag(tmp_path):
    """Resolved pipeline respects processing.normalize=False by preserving raw scale."""
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
    raw_max = phantom.volume.max()
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=4.0,
        regime="resolved",
    )
    config.processing.normalize = False

    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()
    assert summary["n_labels"] > 0
    assert pipeline.volume.max() == pytest.approx(raw_max, rel=1e-5)


def test_pipeline_runs_with_config_file_only(tmp_path):
    """CLI can run with --config alone when data_path and output_dir are in the file."""
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
    config_path = tmp_path / "config.yaml"
    config.save(config_path)

    rc = main(["--config", str(config_path)])

    assert rc == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "labels.tif").exists()
    assert (out_dir / "skeleton.tif").exists()
    assert (out_dir / "normalized_input.tif").exists()


def test_dvc_disabled_omits_dvc_summary(tmp_path):
    """dvc.enabled=False (default) means no DVC step runs and no key is added."""
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

    summary = FiberAnalysisPipeline(config).run()
    assert "dvc" not in summary
    assert not (out_dir / "dvc_summary.json").exists()


def test_dvc_enabled_writes_dvc_reports(tmp_path):
    """dvc.enabled=True runs local DVC on the reference/deformed pair and writes reports."""
    pytest.importorskip("spam")
    import numpy as np
    from scipy.ndimage import affine_transform

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    main_phantom = generate_fiber_phantom(
        shape=(64, 64, 64),
        n_fibers=3,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, main_phantom.volume)

    # 80^3 with node_spacing=20/half_window_size=10 keeps every grid node's
    # window (+ search margin) inside the volume; a smaller volume leaves most
    # nodes out-of-bounds (see fiber_tracer.correlation.dvc.OUT_OF_BOUNDS_STATUS).
    dvc_phantom = generate_fiber_phantom(
        shape=(80, 80, 80),
        n_fibers=200,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=1,
    )
    reference = dvc_phantom.volume.astype(np.float32)
    deformed = affine_transform(
        reference, np.eye(3), offset=[1.0, 0.0, 0.0], order=1, mode="nearest"
    ).astype(np.float32)
    reference_path = data_dir / "dvc_reference.tif"
    deformed_path = data_dir / "dvc_deformed.tif"
    save_tiff_stack(reference_path, reference)
    save_tiff_stack(deformed_path, deformed)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=4.0,
        regime="resolved",
        dvc=DVCConfig(
            enabled=True,
            reference_path=str(reference_path),
            deformed_path=str(deformed_path),
            node_spacing_voxels=20,
            half_window_size_voxels=10,
        ),
    )

    summary = FiberAnalysisPipeline(config).run()

    assert "dvc" in summary
    assert summary["dvc"]["n_windows"] > 0
    assert "noise_floor" in summary["dvc"]
    assert (out_dir / "dvc_summary.json").exists()
    assert (out_dir / "dvc_report.csv").exists()
    assert (out_dir / "dvc_report.html").exists()

    with open(out_dir / "dvc_summary.json") as f:
        dvc_json = json.load(f)
    assert dvc_json["n_windows"] == summary["dvc"]["n_windows"]
