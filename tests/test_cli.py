import json
from unittest.mock import MagicMock

import pytest

from fiber_tracer.cli import main
from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def _make_resolved_output(tmp_path, shape=(32, 32, 32)):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=shape,
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
        regime="resolved",
    )
    FiberAnalysisPipeline(config).run()
    return stack_path, out_dir


def test_cli_threshold_method_runs_end_to_end(tmp_path):
    """The --threshold-method flag drives the pipeline and produces outputs."""
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

    rc = main(
        [
            "run",
            "--data",
            str(stack_path),
            "--output",
            str(out_dir),
            "--voxel-spacing",
            "1.0",
            "1.0",
            "1.0",
            "--fiber-diameter",
            "4.0",
            "--regime",
            "resolved",
            "--threshold-method",
            "multiotsu",
        ]
    )
    assert rc == 0
    assert (out_dir / "summary.json").exists()


def test_cli_manual_threshold_requires_value(tmp_path):
    """Manual thresholding without a value fails validation."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    phantom = generate_fiber_phantom(shape=(16, 16, 16), n_fibers=1, seed=1)
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    with pytest.raises(ValueError):
        main(
            [
                "run",
                "--data",
                str(stack_path),
                "--output",
                str(tmp_path / "out"),
                "--regime",
                "resolved",
                "--threshold-method",
                "manual",
            ]
        )


def test_cli_version_command(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "fiber-tracer" in captured.out


def test_cli_view_command_runs_napari_viewer(tmp_path, monkeypatch):
    mock_run = MagicMock(return_value=0)
    monkeypatch.setattr("fiber_tracer.viz.napari_viewer.run_napari_viewer", mock_run)

    rc = main(["view", "--data", "data.tif", "--output", str(tmp_path / "out")])

    assert rc == 0
    mock_run.assert_called_once_with("data.tif", str(tmp_path / "out"))


def test_cli_default_no_subcommand_runs_pipeline(tmp_path):
    stack_path, out_dir = _make_resolved_output(tmp_path)

    rc = main(
        [
            "--data",
            str(stack_path),
            "--output",
            str(out_dir / "cli_out"),
            "--regime",
            "resolved",
            "--voxel-spacing",
            "1.0",
            "1.0",
            "1.0",
            "--fiber-diameter",
            "4.0",
        ]
    )

    assert rc == 0
    assert (out_dir / "cli_out" / "summary.json").exists()
    assert (out_dir / "cli_out" / "labels.tif").exists()


def test_cli_run_subcommand_runs_pipeline(tmp_path):
    stack_path, out_dir = _make_resolved_output(tmp_path)

    rc = main(
        [
            "run",
            "--data",
            str(stack_path),
            "--output",
            str(out_dir / "run_out"),
            "--regime",
            "resolved",
            "--voxel-spacing",
            "1.0",
            "1.0",
            "1.0",
            "--fiber-diameter",
            "4.0",
        ]
    )

    assert rc == 0
    assert (out_dir / "run_out" / "summary.json").exists()


def test_report_viz_command(tmp_path, monkeypatch):
    summary = {"regime": "resolved", "fibers": []}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    mock_generate = MagicMock(return_value=None)
    monkeypatch.setattr("fiber_tracer.viz.plotly_plots.generate_interactive_report", mock_generate)

    rc = main(
        [
            "report-viz",
            "--summary",
            str(summary_path),
            "--output",
            str(tmp_path / "report.html"),
        ]
    )

    assert rc == 0
    mock_generate.assert_called_once_with(summary, str(tmp_path / "report.html"))


def test_batch_command(tmp_path, monkeypatch):
    config_path = tmp_path / "batch.yaml"
    config_path.write_text("dummy")

    mock_process = MagicMock(return_value=[])
    monkeypatch.setattr("fiber_tracer.batch.process_batch", mock_process)

    rc = main(
        [
            "batch",
            "--config",
            str(config_path),
            "--aggregate-csv",
            str(tmp_path / "aggregate.csv"),
        ]
    )

    assert rc == 0
    mock_process.assert_called_once_with(
        str(config_path), aggregate_csv=str(tmp_path / "aggregate.csv")
    )


def test_cli_config_file_not_overridden_by_defaults(tmp_path, monkeypatch):
    """CLI defaults for --regime and --batch-size must not override config file values."""
    stack_path, out_dir = _make_resolved_output(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"data_path: {stack_path}\n"
        f"output_dir: {out_dir / 'config_out'}\n"
        "voxel_spacing_um: [1.0, 1.0, 1.0]\n"
        "fiber_diameter_um: 4.0\n"
        "regime: resolved\n"
        "segmentation:\n"
        "  method: otsu\n"
        "  batch_size: 8\n"
    )

    mock_pipeline_cls = MagicMock()
    monkeypatch.setattr("fiber_tracer.cli.FiberAnalysisPipeline", mock_pipeline_cls)

    rc = main(["--config", str(config_path)])

    assert rc == 0
    config = mock_pipeline_cls.call_args.args[0]
    assert config.regime == "resolved"
    assert config.segmentation.batch_size == 8
