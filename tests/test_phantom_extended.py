import numpy as np
import pytest

from fiber_tracer.validation.phantoms import (
    apply_xct_domain_randomization,
    compute_orientation_tensor,
    generate_recycled_fiber_phantom,
    generate_short_fiber_phantom,
    generate_woven_bundle_phantom,
    semantic_mask_from_phantom,
)


@pytest.mark.parametrize(
    "generator,kwargs",
    [
        (generate_short_fiber_phantom, {"n_fibers": 20, "seed": 0}),
        (generate_woven_bundle_phantom, {"n_bundles": 4, "seed": 0}),
        (generate_recycled_fiber_phantom, {"n_fibers": 20, "seed": 0}),
    ],
)
def test_extended_phantom_shapes_and_ranges(generator, kwargs):
    phantom = generator(shape=(64, 64, 64), **kwargs)
    assert phantom.volume.shape == (64, 64, 64)
    assert phantom.labels.shape == (64, 64, 64)
    assert 0.0 <= phantom.volume.min() <= phantom.volume.max() <= 1.0
    assert len(phantom.orientations) >= 1


def test_semantic_mask_has_expected_classes():
    phantom = generate_short_fiber_phantom(shape=(64, 64, 64), n_fibers=20, porosity=0.01, seed=0)
    semantic = semantic_mask_from_phantom(phantom)
    assert semantic.shape == phantom.volume.shape
    assert set(np.unique(semantic)).issubset({0, 1, 2})


def test_orientation_tensor_is_symmetric_and_normalized():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a2 = compute_orientation_tensor(directions)
    assert a2.shape == (3, 3)
    assert np.allclose(a2, a2.T)
    # Trace of A2 equals 1 for unit-weighted directions.
    assert np.isclose(np.trace(a2), 1.0)


def test_domain_randomization_preserves_shape_and_range():
    volume = np.random.rand(32, 32, 32).astype(np.float32)
    augmented = apply_xct_domain_randomization(volume, seed=42)
    assert augmented.shape == volume.shape
    assert 0.0 <= augmented.min() <= augmented.max() <= 1.0
