"""Optional TDA descriptors using gudhi."""

import numpy as np

from fiber_tracer.exceptions import BackendNotAvailableError


def _import_gudhi():
    try:
        import gudhi

        return gudhi
    except ImportError as exc:
        raise BackendNotAvailableError("Install tda extra: pip install fiber-tracer[tda]") from exc


def betti_numbers(binary_volume: np.ndarray) -> dict[str, int]:
    """Compute Betti numbers b0, b1, b2 of a binary 3D volume.

    Uses a cubical complex with foreground voxels at filtration value 0
    and background at infinity.
    """
    gudhi = _import_gudhi()
    if binary_volume.ndim != 3:
        raise ValueError(f"Expected 3D binary volume, got {binary_volume.ndim}D")
    filtration = np.where(binary_volume, 0.0, np.inf).astype(np.float64).flatten()
    cc = gudhi.CubicalComplex(
        dimensions=binary_volume.shape,
        top_dimensional_cells=filtration,
    )
    persistence = cc.persistence()
    betti: dict[str, int] = {"b0": 0, "b1": 0, "b2": 0}
    for dim, (birth, death) in persistence:
        if np.isinf(death):
            key = f"b{int(dim)}"
            betti[key] = betti.get(key, 0) + 1
    return betti


def persistence_summary(binary_volume: np.ndarray) -> dict[str, float]:
    """Return summary statistics of finite persistence for a binary volume."""
    gudhi = _import_gudhi()
    if binary_volume.ndim != 3:
        raise ValueError(f"Expected 3D binary volume, got {binary_volume.ndim}D")
    filtration = np.where(binary_volume, 0.0, np.inf).astype(np.float64).flatten()
    cc = gudhi.CubicalComplex(
        dimensions=binary_volume.shape,
        top_dimensional_cells=filtration,
    )
    persistence = cc.persistence()
    finite = [death - birth for _, (birth, death) in persistence if not np.isinf(death)]
    return {
        "n_features": len(persistence),
        "n_finite": len(finite),
        "max_persistence": float(max(finite)) if finite else 0.0,
        "mean_persistence": float(np.mean(finite)) if finite else 0.0,
    }
