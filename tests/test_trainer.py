import json

import numpy as np
import pytest

from fiber_tracer.experiments.store import ExperimentStore
from fiber_tracer.training.trainer import UNetTrainer


def _make_dataset(tmp_path, n_patches: int = 4) -> tuple:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    registry = [{"patch_dir": "patches"}]
    (dataset_dir / "datasets.json").write_text(json.dumps(registry))
    patch_dir = dataset_dir / "patches"
    patch_dir.mkdir()
    for i in range(n_patches):
        np.savez(
            patch_dir / f"patch_{i}.npz",
            volume=np.random.rand(32, 32, 32).astype(np.float32),
            mask=(np.random.rand(32, 32, 32) > 0.5).astype(np.float32),
        )
    return dataset_dir


def _create_experiment(store: ExperimentStore, dataset_dir):
    return store.create(
        name="smoke",
        type="train",
        model_id="unet-v3.2",
        dataset=str(dataset_dir),
    )


def test_trainer_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    dataset_dir = _make_dataset(tmp_path, n_patches=4)
    store = ExperimentStore()
    exp = _create_experiment(store, dataset_dir)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    trainer = UNetTrainer(
        dataset_dir=str(dataset_dir),
        output_dir=str(artifact_dir),
        epochs=1,
        batch_size=2,
        device="cpu",
        features=(8, 16),
    )
    metrics = trainer.train(experiment_id=exp.id)
    assert "train_loss" in metrics
    assert (artifact_dir / "checkpoint.pt").exists()
    assert store.get_experiment(exp.id).status == "completed"
    assert store.get_experiment(exp.id).history


def test_progress_callback_receives_events(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    dataset_dir = _make_dataset(tmp_path, n_patches=4)
    store = ExperimentStore()
    exp = _create_experiment(store, dataset_dir)

    events = []
    trainer = UNetTrainer(
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "artifacts"),
        epochs=1,
        batch_size=2,
        device="cpu",
        features=(8, 16),
        progress_callback=events.append,
    )
    trainer.train(experiment_id=exp.id)

    assert any(e.get("stage") == "train" and "percent" in e for e in events)
    complete_events = [e for e in events if e.get("stage") == "complete"]
    assert complete_events
    assert complete_events[-1]["percent"] == 100


def test_json_progress_emitted_when_env_var_set(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("FIBER_TRACER_JSON_PROGRESS", "1")
    dataset_dir = _make_dataset(tmp_path, n_patches=4)
    store = ExperimentStore()
    exp = _create_experiment(store, dataset_dir)

    trainer = UNetTrainer(
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "artifacts"),
        epochs=1,
        batch_size=2,
        device="cpu",
        features=(8, 16),
    )
    trainer.train(experiment_id=exp.id)

    out = capsys.readouterr().out
    payloads = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payloads.append(json.loads(line))

    assert any(p["stage"] == "train" for p in payloads)
    assert any(p["stage"] == "complete" for p in payloads)


def test_experiment_status_transitions(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    dataset_dir = _make_dataset(tmp_path, n_patches=4)
    store = ExperimentStore()
    exp = _create_experiment(store, dataset_dir)
    assert store.get_experiment(exp.id).status == "pending"

    seen_running = []

    def callback(payload):
        if payload.get("stage") == "train":
            seen_running.append(store.get_experiment(exp.id).status == "running")

    trainer = UNetTrainer(
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "artifacts"),
        epochs=1,
        batch_size=2,
        device="cpu",
        features=(8, 16),
        progress_callback=callback,
    )
    trainer.train(experiment_id=exp.id)

    assert seen_running
    assert all(seen_running)
    assert store.get_experiment(exp.id).status == "completed"


def test_empty_validation_loader_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    dataset_dir = _make_dataset(tmp_path, n_patches=1)
    store = ExperimentStore()
    exp = _create_experiment(store, dataset_dir)

    trainer = UNetTrainer(
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "artifacts"),
        epochs=1,
        batch_size=1,
        device="cpu",
        features=(8, 16),
    )
    with pytest.raises(ValueError, match="validation loader is empty"):
        trainer.train(experiment_id=exp.id)

    assert store.get_experiment(exp.id).status == "failed"
    assert "validation loader is empty" in store.get_experiment(exp.id).error_message
