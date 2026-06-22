"""Optional TDA descriptors using gudhi."""

from typing import Union

import numpy as np
from scipy import ndimage

from fiber_tracer.exceptions import BackendNotAvailableError


def _import_gudhi():
    try:
        import gudhi

        return gudhi
    except ImportError as exc:
        raise BackendNotAvailableError("Install tda extra: pip install fiber-tracer[tda]") from exc


def _validate_binary_volume(binary_volume: np.ndarray) -> None:
    if binary_volume.ndim != 3:
        raise ValueError(f"Expected 3D binary volume, got {binary_volume.ndim}D")


def persistence_diagram(binary_volume: np.ndarray) -> list[dict[str, Union[int, float]]]:
    """Compute the persistence diagram of a binary 3D volume.

    Uses the Euclidean distance transform of the foreground as the filtration.
    Each point in the diagram is returned as a dict with keys:
    ``dimension`` (int), ``birth`` (float), and ``death`` (float).
    Essential features (surviving to the full complex) have ``death == inf``.

    Parameters
    ----------
    binary_volume : np.ndarray
        3D boolean array where ``True`` denotes foreground.

    Returns
    -------
    list[dict]
        Persistence diagram entries.
    """
    gudhi = _import_gudhi()
    _validate_binary_volume(binary_volume)

    # Distance transform gives the distance from each foreground voxel to the
    # nearest background voxel. Setting background voxels to infinity keeps them
    # out of finite sublevel sets, so the resulting persistence diagram describes
    # the foreground topology and feature sizes.
    distance = ndimage.distance_transform_edt(binary_volume).astype(np.float64)
    distance[~binary_volume] = np.inf
    cc = gudhi.CubicalComplex(
        dimensions=binary_volume.shape,
        top_dimensional_cells=distance.flatten(),
    )
    persistence = cc.persistence()
    diagram: list[dict[str, Union[int, float]]] = []
    for dim, (birth, death) in persistence:
        diagram.append(
            {
                "dimension": int(dim),
                "birth": float(birth),
                "death": float("inf") if np.isinf(death) else float(death),
            }
        )
    return diagram


def betti_numbers(binary_volume: np.ndarray) -> dict[str, int]:
    """Compute Betti numbers b0, b1, b2 of a binary 3D volume.

    Betti numbers are read from the essential (infinite-lifetime) features of
    the distance-transform persistence diagram.
    """
    _validate_binary_volume(binary_volume)
    diagram = persistence_diagram(binary_volume)
    betti: dict[str, int] = {"b0": 0, "b1": 0, "b2": 0}
    for point in diagram:
        if np.isinf(point["death"]):
            key = f"b{point['dimension']}"
            betti[key] = betti.get(key, 0) + 1
    return betti


def persistence_summary(binary_volume: np.ndarray) -> dict[str, float]:
    """Return summary statistics of the distance-transform persistence diagram."""
    _validate_binary_volume(binary_volume)
    diagram = persistence_diagram(binary_volume)
    finite = [point["death"] - point["birth"] for point in diagram if not np.isinf(point["death"])]
    return {
        "n_features": len(diagram),
        "n_finite": len(finite),
        "max_persistence": float(max(finite)) if finite else 0.0,
        "mean_persistence": float(np.mean(finite)) if finite else 0.0,
    }
