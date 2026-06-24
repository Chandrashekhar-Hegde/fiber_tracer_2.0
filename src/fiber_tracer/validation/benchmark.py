"""Reusable benchmark utilities."""

from __future__ import annotations

import numpy as np


def mean_dice_per_label(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Mean Dice over all foreground labels in true_labels."""
    true_ids = np.setdiff1d(np.unique(true_labels), [0])
    if len(true_ids) == 0:
        return 0.0
    scores = []
    for tid in true_ids:
        pred_mask = pred_labels == tid
        true_mask = true_labels == tid
        intersection = np.sum(pred_mask & true_mask)
        denom = np.sum(pred_mask) + np.sum(true_mask)
        scores.append(2.0 * intersection / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def _align_labels(
    pred_labels: np.ndarray, true_labels: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Remap predicted label IDs to match ground-truth IDs by overlap.

    Returns a relabeled prediction volume and the mapping used.  Each true
    foreground label is matched to the predicted label with which it shares the
    most voxels.
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    true_ids = np.setdiff1d(np.unique(true_labels), [0])
    pred_ids = np.setdiff1d(np.unique(pred_labels), [0])

    mapping: dict[int, int] = {}
    used_pred_ids: set = set()
    for true_id in true_ids:
        true_mask = true_labels == true_id
        best_pred = None
        best_overlap = 0
        for pred_id in pred_ids:
            if pred_id in used_pred_ids:
                continue
            overlap = int(np.sum(true_mask & (pred_labels == pred_id)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred = pred_id
        if best_pred is None:
            raise RuntimeError(f"Could not find a predicted label overlapping true label {true_id}")
        mapping[best_pred] = int(true_id)
        used_pred_ids.add(best_pred)

    aligned = np.zeros_like(pred_labels)
    for pred_id, true_id in mapping.items():
        aligned[pred_labels == pred_id] = true_id
    return aligned, mapping
