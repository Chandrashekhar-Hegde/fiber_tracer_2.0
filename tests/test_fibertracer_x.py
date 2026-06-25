import pytest
import torch

from fiber_tracer.training.models import FiberTracerX


@pytest.mark.parametrize("device", ["cpu"])
def test_fibertracer_x_forward_shapes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = FiberTracerX(
        tasks={"segment": {"out_channels": 3}, "orient": {}},
        features=(8, 16, 16),
    ).to(device)
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    seg = model(x, "segment")
    orient = model(x, "orient")
    assert seg.shape == (1, 3, 32, 32, 32)
    assert orient.shape == (1, 6)


def test_fibertracer_x_predict_segment_probabilities():
    model = FiberTracerX(
        tasks={"segment": {"out_channels": 1}},
        features=(8, 16),
    )
    x = torch.randn(1, 1, 32, 32, 32)
    probs = model.predict_segment(x)
    assert probs.shape == (1, 1, 32, 32, 32)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_unknown_task_raises():
    model = FiberTracerX(tasks={"segment": {"out_channels": 1}})
    x = torch.randn(1, 1, 16, 16, 16)
    with pytest.raises(KeyError):
        model(x, "orient")
