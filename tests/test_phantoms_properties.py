"""Property-based tests for synthetic fiber phantoms using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fiber_tracer.validation.phantoms import generate_fiber_phantom

SHAPES = st.sampled_from([(32, 32, 32), (48, 48, 48), (64, 64, 64)])
N_FIBERS = st.integers(min_value=1, max_value=5)


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_volume_shape(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    assert phantom.volume.shape == shape


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_volume_range(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    assert np.all(phantom.volume >= 0.0)
    assert np.all(phantom.volume <= 1.0)


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_labels_shape(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    assert phantom.labels.shape == shape


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_placed_fiber_count(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    placed = phantom.orientations.shape[0]
    assert 1 <= placed <= n_fibers


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_orientations_are_unit_vectors(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    for direction in phantom.orientations:
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-9)


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_labels_are_nonnegative_integers(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    assert phantom.labels.dtype.kind == "i"
    assert np.all(phantom.labels >= 0)


@given(shape=SHAPES, n_fibers=N_FIBERS)
@settings(max_examples=10, deadline=None)
def test_phantom_max_label_equals_placed_fiber_count(shape, n_fibers):
    phantom = generate_fiber_phantom(shape=shape, n_fibers=n_fibers, seed=42)
    placed = phantom.orientations.shape[0]
    if placed == 0:
        pytest.skip("no fibers placed")
    assert phantom.labels.max() == placed
