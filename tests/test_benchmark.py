import numpy as np
import pytest

from fiber_tracer.validation.benchmark import _align_labels, mean_dice_per_label


def test_mean_dice_per_label_perfect():
    """Identical labels should yield a mean Dice of 1.0."""
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[:2, :2] = 1
    labels[2:4, 2:4] = 2
    labels[4:6, 4:6] = 3

    assert mean_dice_per_label(labels, labels) == pytest.approx(1.0)


def test_mean_dice_per_label_non_overlapping():
    """Non-overlapping labels should yield a mean Dice of 0.0."""
    true = np.zeros((6, 6), dtype=np.int32)
    true[:2, :2] = 1

    pred = np.zeros((6, 6), dtype=np.int32)
    pred[4:6, 4:6] = 1

    assert mean_dice_per_label(pred, true) == pytest.approx(0.0)


def test_align_labels():
    """_align_labels remaps predicted IDs to maximize overlap with ground truth."""
    true = np.zeros((6, 6), dtype=np.int32)
    true[:2, :2] = 1
    true[2:4, 2:4] = 2

    pred = np.zeros((6, 6), dtype=np.int32)
    pred[:2, :2] = 2
    pred[2:4, 2:4] = 1

    aligned, mapping = _align_labels(pred, true)

    assert np.array_equal(aligned, true)
    assert mapping == {1: 2, 2: 1}
