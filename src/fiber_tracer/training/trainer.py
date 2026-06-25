"""Reusable 3D U-Net trainer with JSON progress emission."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fiber_tracer.backends.unet3d import UNet3D
from fiber_tracer.experiments.store import ExperimentStore
from fiber_tracer.training.checkpoint import save_checkpoint
from fiber_tracer.training.dataset import FiberVolumeDataset, numpy_collate


class BCEDiceLoss(nn.Module):
    """Combined binary cross-entropy and Dice loss."""

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        dice = 1.0 - _dice_score(pred, target).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def _dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return per-sample Dice coefficients for a batch of predictions."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (2.0 * intersection + eps) / (union + eps)


class UNetTrainer:
    """Train a 3D U-Net on a ``FiberVolumeDataset``."""

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-3,
        val_fraction: float = 0.1,
        device: str = "auto",
        features: tuple[int, ...] = (8, 16, 32),
        seed: int = 42,
        split_mode: str = "patch",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the trainer.

        The ``seed`` parameter is kept for reference and passed to the dataset
        for deterministic train/val splits. It intentionally does **not** set
        global RNG state; future versions may use a local generator.
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.seed = seed
        self.split_mode = split_mode
        self.features = features
        self.progress_callback = progress_callback

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _emit(
        self,
        stage: str,
        percent: float,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "stage": stage,
            "percent": round(percent, 2),
            "message": message,
            "metrics": metrics or {},
        }
        if os.environ.get("FIBER_TRACER_JSON_PROGRESS"):
            print(json.dumps(payload), flush=True)
        if self.progress_callback:
            self.progress_callback(payload)

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        registry = self.dataset_dir / "datasets.json"
        train_set = FiberVolumeDataset(
            registry_path=registry,
            processed_root=self.dataset_dir,
            split="train",
            val_fraction=self.val_fraction,
            augment=True,
            seed=self.seed,
            split_mode=self.split_mode,
        )
        val_set = FiberVolumeDataset(
            registry_path=registry,
            processed_root=self.dataset_dir,
            split="val",
            val_fraction=self.val_fraction,
            augment=False,
            seed=self.seed,
            split_mode=self.split_mode,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
            num_workers=0,
        )
        return train_loader, val_loader

    def _numpy_to_tensor(
        self,
        batch: tuple[np.ndarray, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        volumes, masks = batch
        return (
            torch.from_numpy(volumes).float().to(self.device),
            torch.from_numpy(masks).float().to(self.device),
        )

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        model.eval()
        criterion = BCEDiceLoss().to(self.device)
        total_loss = 0.0
        total_dice = 0.0
        total_samples = 0
        for batch in loader:
            inputs, targets = self._numpy_to_tensor(batch)
            batch_size = inputs.size(0)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * batch_size
            total_dice += _dice_score(outputs, targets).mean().item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("validation loader is empty")

        return {
            "loss": total_loss / total_samples,
            "dice": total_dice / total_samples,
        }

    def train(self, experiment_id: str) -> dict[str, Any]:
        """Run training and return final metrics."""
        store = ExperimentStore()
        store.update(experiment_id, status="running")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self._build_loaders()

        model = UNet3D(
            in_channels=1,
            out_channels=1,
            features=self.features,
        ).to(self.device)
        criterion = BCEDiceLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_dice = -1.0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
        }

        try:
            for epoch in range(1, self.epochs + 1):
                model.train()
                epoch_loss = 0.0
                for batch in train_loader:
                    inputs, targets = self._numpy_to_tensor(batch)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_loss = epoch_loss / max(len(train_loader), 1)
                val_metrics = self._evaluate(model, val_loader)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_dice"].append(val_metrics["dice"])

                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    save_checkpoint(
                        self.output_dir / "checkpoint.pt",
                        model,
                        metadata={
                            "epoch": epoch,
                            "val_dice": best_dice,
                            "features": self.features,
                        },
                    )

                percent = 100 * epoch / self.epochs
                self._emit(
                    "train",
                    percent,
                    f"epoch {epoch}/{self.epochs}",
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_dice": val_metrics["dice"],
                    },
                )

                store.update(
                    experiment_id,
                    history=history,
                    metrics={},
                )

            final_metrics = {
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
                "val_dice": history["val_dice"][-1],
                "best_val_dice": best_dice,
            }
            store.update(
                experiment_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                history=history,
                metrics=final_metrics,
                artifact_dir=str(self.output_dir),
            )
            self._emit("complete", 100, "Training complete", final_metrics)
            return final_metrics

        except Exception as exc:
            store.update(
                experiment_id,
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error_message=str(exc),
            )
            self._emit("error", 0, f"Training failed: {exc}", {})
            raise
