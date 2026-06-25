"""Benchmark task definitions and metric functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class TaskDefinition:
    """Description of a benchmark task and its metrics."""

    def __init__(
        self,
        name: str,
        metric_names: list[str],
        compute_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    ) -> None:
        self.name = name
        self.metric_names = metric_names
        self.compute_fn = compute_fn


def _one_hot(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer label map to one-hot (C, D, H, W)."""
    return np.stack([mask == c for c in range(n_classes)], axis=0).astype(np.float32)


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    n_classes: int = 3,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute Dice, IoU, pixel accuracy, and class-wise Dice.

    Parameters
    ----------
    pred:
        Integer predicted labels of shape (D, H, W).
    target:
        Integer ground-truth labels of shape (D, H, W).
    n_classes:
        Number of semantic classes.

    Returns
    -------
    metrics:
        Dictionary with ``mean_dice``, ``mean_iou``, ``pixel_accuracy``,
        and per-class Dice keys.
    """
    pred_oh = _one_hot(pred, n_classes)
    target_oh = _one_hot(target, n_classes)
    intersection = (pred_oh * target_oh).sum(axis=(1, 2, 3))
    union = pred_oh.sum(axis=(1, 2, 3)) + target_oh.sum(axis=(1, 2, 3))
    dice_per_class = (2.0 * intersection + eps) / (union + eps)
    iou_per_class = (intersection + eps) / (
        pred_oh.sum(axis=(1, 2, 3)) + target_oh.sum(axis=(1, 2, 3)) - intersection + eps
    )
    pixel_accuracy = float((pred == target).mean())
    metrics: dict[str, float] = {
        "mean_dice": float(dice_per_class.mean()),
        "mean_iou": float(iou_per_class.mean()),
        "pixel_accuracy": pixel_accuracy,
    }
    for c in range(n_classes):
        metrics[f"dice_class_{c}"] = float(dice_per_class[c])
    return metrics


def orientation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute orientation tensor regression errors.

    Parameters
    ----------
    pred:
        Predicted A2 tensor of shape (3, 3).
    target:
        Ground-truth A2 tensor of shape (3, 3).
    """
    diff = pred - target
    frobenius = float(np.linalg.norm(diff, ord="fro"))
    mse = float((diff**2).mean())
    return {"frobenius_error": frobenius, "mse": mse}


SEGMENTATION_TASK = TaskDefinition(
    name="segmentation",
    metric_names=["mean_dice", "mean_iou", "pixel_accuracy"],
    compute_fn=lambda p, t: segmentation_metrics(p, t, n_classes=3),
)

ORIENTATION_TASK = TaskDefinition(
    name="orientation",
    metric_names=["frobenius_error", "mse"],
    compute_fn=orientation_metrics,
)
