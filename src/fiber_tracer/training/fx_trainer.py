"""Multi-task trainer for the FiberTracer-X foundation model."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fiber_tracer.experiments.store import ExperimentStore
from fiber_tracer.training.checkpoint import save_checkpoint
from fiber_tracer.training.models.fibertracer_x import (
    FiberTracerX,
    OrientationRegressorAdapter,
)
from fiber_tracer.training.synthetic_dataset import (
    SyntheticCorpusDataset,
    synthetic_collate,
)


def _a2_to_components(a2: torch.Tensor) -> torch.Tensor:
    """Extract the 6 unique components of a symmetric 3x3 A2 tensor.

    Input shape: (B, 3, 3).  Output shape: (B, 6) with
    (a11, a12, a13, a22, a23, a33).
    """
    return torch.stack(
        [a2[:, 0, 0], a2[:, 0, 1], a2[:, 0, 2], a2[:, 1, 1], a2[:, 1, 2], a2[:, 2, 2]],
        dim=1,
    )


class SegmentationLoss(nn.Module):
    """Cross-entropy + Dice loss for multi-class semantic segmentation."""

    def __init__(self, n_classes: int = 3, ce_weight: float = 0.5) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce: torch.Tensor = self.ce(logits, target)
        dice = 1.0 - _multiclass_dice(logits, target, self.n_classes).mean()
        return self.ce_weight * ce + (1.0 - self.ce_weight) * dice


def _multiclass_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-sample multiclass Dice coefficients."""
    probs = torch.softmax(logits, dim=1)
    dice_per_class = []
    for c in range(n_classes):
        pred = probs[:, c]
        true = (target == c).float()
        intersection = (pred * true).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + true.sum(dim=(1, 2, 3))
        dice_per_class.append((2.0 * intersection + eps) / (union + eps))
    return torch.stack(dice_per_class, dim=1).mean(dim=1)


class OrientationLoss(nn.Module):
    """MSE on the unique A2 components plus a Frobenius matrix penalty."""

    def __init__(self, matrix_weight: float = 0.5) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.matrix_weight = matrix_weight

    def forward(self, pred_components: torch.Tensor, target_a2: torch.Tensor) -> torch.Tensor:
        target_components = _a2_to_components(target_a2)
        comp_loss: torch.Tensor = self.mse(pred_components, target_components)
        pred_matrix = OrientationRegressorAdapter.components_to_matrix(pred_components)
        matrix_loss: torch.Tensor = torch.linalg.matrix_norm(
            pred_matrix - target_a2, ord="fro"
        ).mean()
        return comp_loss + self.matrix_weight * matrix_loss


class FiberTracerXTrainer:
    """Train FiberTracer-X on the synthetic corpus.

    Parameters
    ----------
    corpus_dir:
        Directory containing ``corpus.json`` and ``patches/``.
    output_dir:
        Where to save checkpoints and logs.
    epochs:
        Number of training epochs.
    batch_size:
        Batch size for training and validation.
    lr:
        Learning rate.
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    features:
        Encoder feature channels.
    n_classes:
        Number of semantic classes (default 3: matrix, fiber, void).
    seg_weight:
        Weight of segmentation loss in the multi-task objective.
    orient_weight:
        Weight of orientation loss in the multi-task objective.
    """

    def __init__(
        self,
        corpus_dir: str,
        output_dir: str,
        epochs: int = 10,
        batch_size: int = 2,
        lr: float = 1e-3,
        val_fraction: float = 0.1,
        device: str = "auto",
        features: tuple[int, ...] = (16, 32, 64),
        n_classes: int = 3,
        seg_weight: float = 0.7,
        orient_weight: float = 0.3,
        seed: int = 42,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.seed = seed
        self.features = features
        self.n_classes = n_classes
        self.seg_weight = seg_weight
        self.orient_weight = orient_weight
        self.progress_callback = progress_callback

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = FiberTracerX(
            tasks={
                "segment": {"out_channels": n_classes},
                "orient": {"hidden_dim": 128},
            },
            in_channels=1,
            features=features,
        ).to(self.device)

        self.seg_criterion = SegmentationLoss(n_classes=n_classes).to(self.device)
        self.orient_criterion = OrientationLoss().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

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
        train_set = SyntheticCorpusDataset(
            self.corpus_dir, split="train", val_fraction=self.val_fraction, seed=self.seed
        )
        val_set = SyntheticCorpusDataset(
            self.corpus_dir, split="val", val_fraction=self.val_fraction, seed=self.seed
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=synthetic_collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=synthetic_collate,
            num_workers=0,
        )
        return train_loader, val_loader

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        inputs = batch["volume"].to(self.device)
        semantic = batch["semantic"].to(self.device)
        a2 = batch["a2"].to(self.device)

        seg_logits = self.model(inputs, task="segment")
        seg_loss = self.seg_criterion(seg_logits, semantic)

        orient_pred = self.model(inputs, task="orient")
        orient_loss = self.orient_criterion(orient_pred, a2)

        loss = self.seg_weight * seg_loss + self.orient_weight * orient_loss
        return {
            "loss": loss,
            "seg_loss": seg_loss.item(),
            "orient_loss": orient_loss.item(),
        }

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_orient_loss = 0.0
        total_seg_dice = 0.0
        total_samples = 0
        for batch in loader:
            batch_size = batch["volume"].size(0)
            metrics = self._run_batch(batch)
            seg_logits = self.model(batch["volume"].to(self.device), task="segment")
            seg_dice = _multiclass_dice(
                seg_logits, batch["semantic"].to(self.device), self.n_classes
            ).mean()
            total_loss += metrics["loss"].item() * batch_size
            total_seg_loss += metrics["seg_loss"] * batch_size
            total_orient_loss += metrics["orient_loss"] * batch_size
            total_seg_dice += seg_dice.item() * batch_size
            total_samples += batch_size
        if total_samples == 0:
            raise ValueError("validation loader is empty")
        return {
            "loss": total_loss / total_samples,
            "seg_loss": total_seg_loss / total_samples,
            "orient_loss": total_orient_loss / total_samples,
            "seg_dice": total_seg_dice / total_samples,
        }

    def train(self, experiment_id: str) -> dict[str, Any]:
        """Run multi-task pre-training and return final metrics."""
        store = ExperimentStore()
        store.update(experiment_id, status="running")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self._build_loaders()

        best_loss = float("inf")
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_seg_dice": [],
        }

        try:
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_loss = 0.0
                epoch_seg_loss = 0.0
                epoch_orient_loss = 0.0
                for batch in train_loader:
                    self.optimizer.zero_grad()
                    metrics = self._run_batch(batch)
                    metrics["loss"].backward()
                    self.optimizer.step()
                    epoch_loss += metrics["loss"].item()
                    epoch_seg_loss += metrics["seg_loss"]
                    epoch_orient_loss += metrics["orient_loss"]

                train_loss = epoch_loss / max(len(train_loader), 1)
                val_metrics = self._evaluate(val_loader)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_seg_dice"].append(val_metrics["seg_dice"])

                if val_metrics["loss"] < best_loss:
                    best_loss = val_metrics["loss"]
                    save_checkpoint(
                        self.output_dir / "checkpoint.pt",
                        self.model,
                        metadata={
                            "epoch": epoch,
                            "val_loss": best_loss,
                            "features": self.features,
                            "n_classes": self.n_classes,
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
                        "train_seg_loss": epoch_seg_loss / max(len(train_loader), 1),
                        "train_orient_loss": epoch_orient_loss / max(len(train_loader), 1),
                        "val_loss": val_metrics["loss"],
                        "val_seg_loss": val_metrics["seg_loss"],
                        "val_orient_loss": val_metrics["orient_loss"],
                        "val_seg_dice": val_metrics["seg_dice"],
                    },
                )
                store.update(experiment_id, history=history, metrics={})

            final_metrics = {
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
                "val_seg_dice": history["val_seg_dice"][-1],
            }
            store.update(
                experiment_id,
                status="completed",
                finished_at=datetime.now(UTC).isoformat(),
                history=history,
                metrics=final_metrics,
                artifact_dir=str(self.output_dir),
            )
            self._emit("complete", 100, "Pre-training complete", final_metrics)
            return final_metrics

        except Exception as exc:
            store.update(
                experiment_id,
                status="failed",
                finished_at=datetime.now(UTC).isoformat(),
                error_message=str(exc),
            )
            self._emit("error", 0, f"Pre-training failed: {exc}", {})
            raise
