"""Benchmark runner for task-aware evaluation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fiber_tracer.benchmark.tasks import SEGMENTATION_TASK, TaskDefinition
from fiber_tracer.training.checkpoint import load_checkpoint
from fiber_tracer.training.models.fibertracer_x import FiberTracerX
from fiber_tracer.training.synthetic_dataset import (
    SyntheticCorpusDataset,
    synthetic_collate,
)


class BenchmarkRunner:
    """Run a benchmark task and persist JSON results.

    Parameters
    ----------
    model:
        A PyTorch model with a task-specific forward method.
    task:
        Benchmark task definition.
    device:
        Device to run inference on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task: TaskDefinition,
        device: torch.device,
    ) -> None:
        self.model = model
        self.task = task
        self.device = device
        self.model.eval()

    @classmethod
    def from_fibertracer_x_checkpoint(
        cls,
        checkpoint_path: str | Path,
        task_name: str = "segment",
        device: str = "auto",
    ) -> BenchmarkRunner:
        """Build a runner from a FiberTracer-X checkpoint."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        device_obj = torch.device(device)

        checkpoint = load_checkpoint(checkpoint_path)
        metadata = checkpoint.get("metadata", {})
        features = tuple(metadata.get("features", (16, 32, 64)))
        n_classes = metadata.get("n_classes", 3)

        # Always load both adapters present in a pre-training checkpoint; the
        # requested benchmark task only selects which head is used for scoring.
        tasks = {
            "segment": {"out_channels": n_classes},
            "orient": {},
        }
        model = FiberTracerX(tasks=tasks, features=features).to(device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])

        task = SEGMENTATION_TASK if task_name == "segment" else SEGMENTATION_TASK
        return cls(model, task, device_obj)

    def _predict_segment(self, volume: torch.Tensor) -> np.ndarray:
        """Predict semantic labels for a batch of volumes."""
        with torch.no_grad():
            logits = self.model(volume, task="segment")
        return torch.argmax(logits, dim=1).cpu().numpy()

    def _predict_orient(self, volume: torch.Tensor) -> np.ndarray:
        """Predict A2 tensor for a batch of volumes."""
        with torch.no_grad():
            components = self.model(volume, task="orient")
        matrices = self.model.adapters["orient"].components_to_matrix(components)
        return matrices.cpu().numpy()

    def run(
        self,
        loader: DataLoader,
        output_path: str | Path | None = None,
        run_name: str = "run",
    ) -> dict[str, Any]:
        """Run inference over *loader* and compute task metrics."""
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        start = time.perf_counter()

        for batch in loader:
            volume = batch["volume"].to(self.device)
            if self.task.name == "segmentation":
                pred = self._predict_segment(volume)
                target = batch["semantic"].numpy()
            elif self.task.name == "orientation":
                pred = self._predict_orient(volume)
                target = batch["a2"].numpy()
            else:
                raise ValueError(f"Unsupported task: {self.task.name}")
            predictions.append(pred)
            targets.append(target)

        elapsed = time.perf_counter() - start

        # Flatten across batches for metric computation.
        if self.task.name == "segmentation":
            pred_all = np.concatenate([p.reshape(-1, *p.shape[-3:]) for p in predictions], axis=0)
            target_all = np.concatenate([t.reshape(-1, *t.shape[-3:]) for t in targets], axis=0)
        else:
            pred_all = np.concatenate(predictions, axis=0)
            target_all = np.concatenate(targets, axis=0)

        metrics = self.task.compute_fn(pred_all[0], target_all[0])
        # For segmentation, average per-sample metrics would be better, but
        # for a quick leaderboard we report the whole-volume metric.
        if self.task.name == "segmentation":
            sample_metrics = [
                self.task.compute_fn(pred_all[i], target_all[i]) for i in range(len(pred_all))
            ]
            metrics = {
                key: float(np.mean([m[key] for m in sample_metrics])) for key in sample_metrics[0]
            }

        result = {
            "run_name": run_name,
            "task": self.task.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(pred_all),
            "inference_seconds": elapsed,
            "metrics": metrics,
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        return result


def build_synthetic_loader(
    corpus_dir: str | Path,
    split: str = "val",
    batch_size: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> DataLoader:
    """Convenience helper to build a DataLoader for the synthetic corpus."""
    dataset = SyntheticCorpusDataset(
        corpus_dir=corpus_dir,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=synthetic_collate,
        num_workers=0,
    )
