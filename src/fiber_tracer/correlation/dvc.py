"""Local Digital Volume Correlation (DVC) -- thin re-export of the shared
correlation engine in `fiber_tracer.correlation.core`.

See `core.py` for the algorithm, its accuracy methodology, and the bugs
found and fixed while building it (boundary nodes, fork deadlock, PhiCentre
convention) -- all apply identically to `fiber_tracer.correlation.dic`.
"""

from __future__ import annotations

from fiber_tracer.correlation.core import (
    CONVERGED_STATUS,
    OUT_OF_BOUNDS_STATUS,
    displacement_and_strain_per_node,
    estimate_noise_floor,
)
from fiber_tracer.correlation.core import run_local_correlation as run_local_dvc

__all__ = [
    "CONVERGED_STATUS",
    "OUT_OF_BOUNDS_STATUS",
    "displacement_and_strain_per_node",
    "estimate_noise_floor",
    "run_local_dvc",
]
