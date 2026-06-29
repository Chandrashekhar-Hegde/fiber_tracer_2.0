"""Benchmark harness for task-aware model evaluation."""

from fiber_tracer.benchmark.runner import BenchmarkRunner
from fiber_tracer.benchmark.tasks import (
    ORIENTATION_TASK,
    SEGMENTATION_TASK,
    TaskDefinition,
)

__all__ = ["BenchmarkRunner", "TaskDefinition", "SEGMENTATION_TASK", "ORIENTATION_TASK"]
