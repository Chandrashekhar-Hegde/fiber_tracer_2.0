"""Optional backend adapters for ML and TDA."""

from fiber_tracer.backends.base import SegmentationBackend
from fiber_tracer.backends.ml_segmentation import MLSegmentationBackend
from fiber_tracer.backends.tda_gudhi import betti_numbers, persistence_summary

__all__ = [
    "SegmentationBackend",
    "MLSegmentationBackend",
    "betti_numbers",
    "persistence_summary",
]
