"""Validation metrics against ground truth."""

from __future__ import annotations

import numpy as np


def angular_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Smallest angle between two directions in degrees."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if np.linalg.norm(pred) == 0 or np.linalg.norm(true) == 0:
        raise ValueError("Direction vector must be non-zero")
    pred = pred / np.linalg.norm(pred)
    true = true / np.linalg.norm(true)
    dot = np.clip(np.abs(np.dot(pred, true)), 0, 1)
    return float(np.degrees(np.arccos(dot)))


def orientation_tensor_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Frobenius norm of A2 difference."""
    return float(np.linalg.norm(pred - true, ord="fro"))


def dice_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Dice coefficient between two binary masks.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask. Should be a boolean array or contain only 0/1.
    true : np.ndarray
        Ground-truth binary mask. Should be a boolean array or contain only 0/1.

    Returns
    -------
    float
        Dice coefficient.
    """
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    if np.sum(pred) + np.sum(true) == 0:
        return 1.0
    intersection = np.sum(pred & true)
    return float(2.0 * intersection / (np.sum(pred) + np.sum(true)))


def mean_angular_error(pred_directions: np.ndarray, true_directions: np.ndarray) -> float:
    """Mean angular error over corresponding direction rows.

    Parameters
    ----------
    pred_directions : np.ndarray
        Array of shape (N, 3).
    true_directions : np.ndarray
        Array of shape (N, 3).

    Returns
    -------
    float
        Mean angular error in degrees.
    """
    pred_directions = np.asarray(pred_directions)
    true_directions = np.asarray(true_directions)
    if pred_directions.shape != true_directions.shape:
        raise ValueError("pred_directions and true_directions must have the same shape")
    if pred_directions.ndim != 2 or pred_directions.shape[1] != 3:
        raise ValueError("expected arrays of shape (N, 3)")
    if pred_directions.shape[0] == 0:
        return 0.0
    errors = [angular_error(p, t) for p, t in zip(pred_directions, true_directions)]
    return float(np.mean(errors))


def mean_dice_score(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Mean Dice score over foreground labels present in the ground truth.

    For each unique foreground label in *true_labels* (excluding 0), the Dice
    coefficient is computed between the binary mask for that label in both
    *pred_labels* and *true_labels*.  The label IDs are assumed to be aligned.

    Parameters
    ----------
    pred_labels : np.ndarray
        Predicted label volume.
    true_labels : np.ndarray
        Ground-truth label volume.

    Returns
    -------
    float
        Mean Dice score across foreground labels.
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    true_foreground = np.setdiff1d(np.unique(true_labels), [0])
    if len(true_foreground) == 0:
        pred_foreground = np.setdiff1d(np.unique(pred_labels), [0])
        return 1.0 if len(pred_foreground) == 0 else 0.0
    scores = []
    for label in true_foreground:
        pred_mask = pred_labels == label
        true_mask = true_labels == label
        scores.append(dice_score(pred_mask, true_mask))
    return float(np.mean(scores))
