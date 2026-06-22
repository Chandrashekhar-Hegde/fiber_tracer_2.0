"""Property-based tests for validation metrics using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fiber_tracer.validation.metrics import (
    angular_error,
    dice_score,
    mean_angular_error,
    orientation_tensor_error,
)

float_array = arrays(
    np.float64,
    (3,),
    elements=st.floats(-1e3, 1e3),
)


@given(float_array, float_array)
def test_angular_error_range(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        pytest.skip("zero vector")
    assert 0.0 <= angular_error(v1, v2) <= 90.0


@given(float_array)
def test_angular_error_identical(v):
    if np.linalg.norm(v) == 0:
        pytest.skip("zero vector")
    assert angular_error(v, v) == pytest.approx(0.0, abs=1e-9)


@given(float_array)
def test_angular_error_sign_ambiguity(v):
    if np.linalg.norm(v) == 0:
        pytest.skip("zero vector")
    assert angular_error(v, -v) == pytest.approx(0.0, abs=1e-9)


@given(mask=arrays(np.bool_, (10, 10)))
def test_dice_score_identical_masks(mask):
    assert dice_score(mask, mask) == pytest.approx(1.0)


@given(arrays(np.bool_, (10, 10)), arrays(np.bool_, (10, 10)))
def test_dice_score_symmetric(a, b):
    assert dice_score(a, b) == pytest.approx(dice_score(b, a))


@given(
    arrays(
        np.float64,
        (5, 3),
        elements=st.floats(-1.0, 1.0),
    ),
    arrays(
        np.float64,
        (5, 3),
        elements=st.floats(-1.0, 1.0),
    ),
)
def test_mean_angular_error_range(pred_directions, true_directions):
    for directions in (pred_directions, true_directions):
        for i in range(directions.shape[0]):
            norm = np.linalg.norm(directions[i])
            if norm < 1e-12:
                pytest.skip("zero direction row")
            directions[i] = directions[i] / norm
    assert 0.0 <= mean_angular_error(pred_directions, true_directions) <= 90.0


@given(
    arrays(np.float64, (3, 3), elements=st.floats(-10.0, 10.0)),
)
def test_orientation_tensor_error_identical(tensor):
    assert orientation_tensor_error(tensor, tensor) == pytest.approx(0.0, abs=1e-9)
