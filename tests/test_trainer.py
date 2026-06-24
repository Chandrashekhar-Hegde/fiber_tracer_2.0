import json

import numpy as np

from fiber_tracer.experiments.store import ExperimentStore
from fiber_tracer.training.trainer import UNetTrainer


def test_trainer_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    registry = [{"patch_dir": "patches"}]
    (dataset_dir / "datasets.json").write_text(json.dumps(registry))
    patch_dir = dataset_dir / "patches"
    patch_dir.mkdir()
    for i in range(4):
        np.savez(
            patch_dir / f"patch_{i}.npz",
            volume=np.random.rand(32, 32, 32).astype(np.float32),
            mask=(np.random.rand(32, 32, 32) > 0.5).astype(np.float32),
        )

    store = ExperimentStore()
    exp = store.create(
        name="smoke",
        type="train",
        model_id="unet-v3.2",
        dataset=str(dataset_dir),
    )
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
