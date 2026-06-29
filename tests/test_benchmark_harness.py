from fiber_tracer.benchmark import ORIENTATION_TASK, SEGMENTATION_TASK
from fiber_tracer.benchmark.runner import BenchmarkRunner
from fiber_tracer.training.checkpoint import save_checkpoint
from fiber_tracer.training.models import FiberTracerX


def _write_checkpoint(path):
    features = (8, 16, 16)
    n_classes = 3
    model = FiberTracerX(
        tasks={"segment": {"out_channels": n_classes}, "orient": {}},
        features=features,
    )
    save_checkpoint(path, model, metadata={"features": features, "n_classes": n_classes})


def test_runner_selects_segmentation_task(tmp_path):
    ckpt = tmp_path / "fx.pt"
    _write_checkpoint(ckpt)
    runner = BenchmarkRunner.from_fibertracer_x_checkpoint(ckpt, task_name="segment", device="cpu")
    assert runner.task is SEGMENTATION_TASK
    assert runner.task.name == "segmentation"


def test_runner_selects_orientation_task(tmp_path):
    """Requesting the orient task must use ORIENTATION_TASK, not segmentation."""
    ckpt = tmp_path / "fx.pt"
    _write_checkpoint(ckpt)
    runner = BenchmarkRunner.from_fibertracer_x_checkpoint(ckpt, task_name="orient", device="cpu")
    assert runner.task is ORIENTATION_TASK
    assert runner.task.name == "orientation"
