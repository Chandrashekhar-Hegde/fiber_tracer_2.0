"""Local Digital Image Correlation (DIC) -- thin re-export of the shared
correlation engine in `fiber_tracer.correlation.core`, called with 2D
(shape (1, H, W)) images.

See `core.py` for the algorithm and accuracy methodology; see
docs/superpowers/specs/2026-08-04-dic-spike-design.md for the spike that
validated this is the same engine as DVC, unchanged.
"""

from __future__ import annotations

from fiber_tracer.correlation.core import (
    CONVERGED_STATUS,
    OUT_OF_BOUNDS_STATUS,
    displacement_and_strain_per_node,
    estimate_noise_floor,
)
from fiber_tracer.correlation.core import run_local_correlation as run_local_dic

__all__ = [
    "CONVERGED_STATUS",
    "OUT_OF_BOUNDS_STATUS",
    "displacement_and_strain_per_node",
    "estimate_noise_floor",
    "run_local_dic",
]
