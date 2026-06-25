"""In-process tests for train checkpoint registration and --model-path resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

from fiber_tracer.cli import main
from fiber_tracer.models.registry import ModelRegistry


def test_train_registers_new_model(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(config_dir))

    output_dir = tmp_path / "output"
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("dummy")

    mock_trainer_cls = MagicMock()
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.train.return_value = {"val_dice": 0.9}
    monkeypatch.setattr("fiber_tracer.training.trainer.UNetTrainer", mock_trainer_cls)

    rc = main(
        [
            "train",
            "--dataset-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output_dir),
            "--model-id",
            "my-new-model",
            "--epochs",
            "1",
            "--batch-size",
            "1",
        ]
    )

    assert rc == 0
    registry = ModelRegistry()
    model = registry.get_model("my-new-model")
    assert model is not None
    assert model.path == str(checkpoint_path)


def test_train_updates_existing_model(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(config_dir))

    old_path = tmp_path / "old.pt"
    old_path.write_text("old")
    registry = ModelRegistry()
    registry.add_model(
        model_id="existing-model",
        name="Existing Model",
        path=str(old_path),
    )

    output_dir = tmp_path / "output"
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("dummy")

    mock_trainer_cls = MagicMock()
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.train.return_value = {"val_dice": 0.9}
    monkeypatch.setattr("fiber_tracer.training.trainer.UNetTrainer", mock_trainer_cls)

    rc = main(
        [
            "train",
            "--dataset-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output_dir),
            "--model-id",
            "existing-model",
            "--epochs",
            "1",
            "--batch-size",
            "1",
        ]
    )

    assert rc == 0
    updated_registry = ModelRegistry()
    model = updated_registry.get_model("existing-model")
    assert model.path == str(checkpoint_path)


def test_run_resolves_registry_model_path(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(config_dir))

    data_path = tmp_path / "data.tif"
    data_path.write_text("dummy")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"data_path: {data_path}\n"
        "output_dir: output\n"
        "voxel_spacing_um: [1.0, 1.0, 1.0]\n"
        "fiber_diameter_um: 6.0\n"
    )

    mock_pipeline_cls = MagicMock()
    monkeypatch.setattr("fiber_tracer.cli.FiberAnalysisPipeline", mock_pipeline_cls)

    rc = main(
        [
            "run",
            "--config",
            str(config_path),
            "--segmentation-method",
            "unet",
            "--model-path",
            "unet-v3.2",
        ]
    )

    assert rc == 0
    config = mock_pipeline_cls.call_args.args[0]
    assert config.segmentation.model_path == "models/fiber_unet_v2_full.pt"
