"""Optional TDA descriptors using ripser."""

import numpy as np

from fiber_tracer.exceptions import BackendNotAvailableError


def _import_ripser():
    try:
        import ripser

        return ripser
    except ImportError as exc:
        raise BackendNotAvailableError("Install tda extra: pip install fiber-tracer[tda]") from exc


def ripser_persistence(
    points: np.ndarray, max_dim: int = 1
) -> dict[str, list[tuple[float, float]]]:
    """Compute Vietoris-Rips persistence diagrams for a point cloud."""
    ripser = _import_ripser()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point cloud, got shape {points.shape}")
    result = ripser.ripser(points, maxdim=max_dim)
    diagrams = result.get("dgms", [])
    return {
        f"h{i}": [(float(b), float(d)) for b, d in diagram] for i, diagram in enumerate(diagrams)
    }
