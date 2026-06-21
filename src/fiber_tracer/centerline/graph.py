"""Graph analysis of skeletons using optional skan backend."""

import numpy as np

from fiber_tracer.exceptions import BackendNotAvailableError


def skeleton_to_skan(skeleton: np.ndarray):
    """Return a skan Skeleton object for graph analysis if available."""
    try:
        from skan import Skeleton
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Install skeleton extra: pip install fiber-tracer[skeleton]"
        ) from exc
    return Skeleton(skeleton)
