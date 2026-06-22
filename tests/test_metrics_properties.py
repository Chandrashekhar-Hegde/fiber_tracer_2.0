"""Property-based tests for validation metrics using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fiber_tracer.validation.metrics import (
    angular_error,
    dice_score,
    mean_angular_error,
    orientation_tensor_error,
)

non_zero_vector = arrays(np.float64, (3,), elements=st.floats(-1e3, 1e3)).filter(
    lambda v: np.linalg.norm(v) > 1e-12
)


@given(non_zero_vector, non_zero_vector)
@settings(deadline=2000)
def test_angular_error_range(v1, v2):
    assert 0.0 <= angular_error(v1, v2) <= 90.0


@given(non_zero_vector)
@settings(deadline=2000)
def test_angular_error_identical(v):
    assert angular_error(v, v) == pytest.approx(0.0, abs=1e-5)


@given(non_zero_vector)
@settings(deadline=2000)
def test_angular_error_sign_ambiguity(v):
    assert angular_error(v, -v) == pytest.approx(0.0, abs=1e-5)


@given(mask=arrays(np.bool_, (10, 10)))
@settings(deadline=2000)
def test_dice_score_identical_masks(mask):
    assert dice_score(mask, mask) == pytest.approx(1.0)


@given(arrays(np.bool_, (10, 10)), arrays(np.bool_, (10, 10)))
@settings(deadline=2000)
def test_dice_score_symmetric(a, b):
    assert dice_score(a, b) == pytest.approx(dice_score(b, a))


def _normalize_rows(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


unit_vectors_5x3 = (
    arrays(np.float64, (5, 3), elements=st.floats(-1.0, 1.0))
    .filter(lambda a: np.all(np.linalg.norm(a, axis=1) > 1e-12))
    .map(_normalize_rows)
)


@given(unit_vectors_5x3, unit_vectors_5x3)
@settings(deadline=2000)
def test_mean_angular_error_range(pred_directions, true_directions):
    assert 0.0 <= mean_angular_error(pred_directions, true_directions) <= 90.0


@given(
    arrays(np.float64, (3, 3), elements=st.floats(-10.0, 10.0)),
)
@settings(deadline=2000)
def test_orientation_tensor_error_identical(tensor):
    assert orientation_tensor_error(tensor, tensor) == pytest.approx(0.0, abs=1e-9)
