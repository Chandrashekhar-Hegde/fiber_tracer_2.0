"""Tests for validation metrics."""

import numpy as np
import pytest

from fiber_tracer.validation.metrics import (
    angular_error,
    dice_score,
    mean_angular_error,
    mean_dice_score,
)


def test_angular_error_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert angular_error(v, v) == pytest.approx(0.0)


def test_angular_error_perpendicular():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert angular_error(a, b) == pytest.approx(90.0)


def test_angular_error_antiparallel():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert angular_error(a, b) == pytest.approx(0.0)


def test_dice_score_identical():
    mask = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    assert dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_score_non_overlapping():
    a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    b = np.array([[0, 0, 0], [1, 1, 0]], dtype=bool)
    assert dice_score(a, b) == pytest.approx(0.0)


def test_mean_angular_error_multiple():
    pred = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    assert mean_angular_error(pred, true) == pytest.approx(0.0)


def test_mean_angular_error_shape_mismatch():
    pred = np.array([[1.0, 0.0, 0.0]])
    true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        mean_angular_error(pred, true)


def test_mean_dice_score_multiple_labels():
    true = np.zeros((6, 6), dtype=np.int32)
    true[:2, :2] = 1
    true[2:4, 2:4] = 2
    true[4:6, 4:6] = 3

    pred = np.zeros((6, 6), dtype=np.int32)
    pred[:2, :2] = 1
    pred[2:4, 2:4] = 2
    pred[4:6, 4:6] = 3

    assert mean_dice_score(pred, true) == pytest.approx(1.0)


def test_mean_dice_score_missing_label():
    true = np.zeros((6, 6), dtype=np.int32)
    true[:2, :2] = 1
    true[2:4, 2:4] = 2

    pred = np.zeros((6, 6), dtype=np.int32)
    pred[:2, :2] = 1

    score = mean_dice_score(pred, true)
    assert score == pytest.approx(0.5)
