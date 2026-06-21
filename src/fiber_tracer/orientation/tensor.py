"""Advani-Tucker second-order orientation tensor."""

import numpy as np


def direction_tensor(directions: np.ndarray) -> np.ndarray:
    """Compute A2 = <p p^T> for an array of unit directions."""
    directions = np.atleast_2d(directions)
    return np.asarray(
        np.mean(np.einsum("bi,bj->bij", directions, directions), axis=0), dtype=np.float64
    )


def fractional_anisotropy(tensor: np.ndarray) -> float:
    """Scalar anisotropy measure from A2 eigenvalues."""
    evals = np.linalg.eigvalsh(tensor)
    mean = evals.mean()
    if mean == 0:
        return 0.0
    return float(np.sqrt(1.5 * np.sum((evals - mean) ** 2) / np.sum(evals**2)))


def windowed_orientation_tensor_field(
    directions: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a spatially-windowed second-order orientation tensor field.

    Parameters
    ----------
    directions:
        Unit direction array of shape ``(3, Z, Y, X)``.
    window_size:
        Requested cubic window size. Even values are rounded up to the next odd
        integer so that windows have a well-defined integer center.
    stride:
        Step size between neighboring window centers.

    Returns
    -------
    tensor_field:
        Array of shape ``(Z', Y', X', 3, 3)`` containing the Advani-Tucker A2
        tensor for each window.
    centers:
        Integer array of shape ``(Z', Y', X', 3)`` with the ``(z, y, x)``
        center coordinate of each window in the original volume.

    Notes
    -----
    Windows are kept fully inside the volume, so output spatial dimensions are
    ``floor((D - window_size) / stride) + 1`` for each spatial dimension ``D``.
    """
    directions = np.asarray(directions)
    if directions.ndim != 4 or directions.shape[0] != 3:
        raise ValueError("directions must have shape (3, Z, Y, X)")

    _, z_dim, y_dim, x_dim = directions.shape

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    # Force odd window size so every window has an unambiguous integer center.
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    radius = window_size // 2

    min_dim = min(z_dim, y_dim, x_dim)
    if window_size > min_dim:
        raise ValueError(
            f"window_size {window_size} exceeds the smallest spatial dimension {min_dim}"
        )

    def _centers(dim: int) -> np.ndarray:
        start = radius
        stop = dim - radius  # exclusive
        return np.arange(start, stop, stride)

    z_centers = _centers(z_dim)
    y_centers = _centers(y_dim)
    x_centers = _centers(x_dim)

    z_out = len(z_centers)
    y_out = len(y_centers)
    x_out = len(x_centers)

    tensor_field = np.zeros((z_out, y_out, x_out, 3, 3), dtype=np.float64)
    centers = np.zeros((z_out, y_out, x_out, 3), dtype=np.int64)

    for i, cz in enumerate(z_centers):
        for j, cy in enumerate(y_centers):
            for k, cx in enumerate(x_centers):
                window = directions[
                    :,
                    cz - radius : cz + radius + 1,
                    cy - radius : cy + radius + 1,
                    cx - radius : cx + radius + 1,
                ]
                vectors = window.reshape(3, -1).T
                tensor_field[i, j, k] = direction_tensor(vectors)
                centers[i, j, k] = (cz, cy, cx)

    return tensor_field, centers


def aggregate_direction_tensor(directions: np.ndarray) -> np.ndarray:
    """Compute the global second-order orientation tensor A2.

    Accepts either a ``(3, Z, Y, X)`` spatial direction field or an array of
    shape ``(N, 3)`` containing individual direction vectors.
    """
    directions = np.asarray(directions)
    if directions.ndim == 4 and directions.shape[0] == 3:
        vectors = directions.reshape(3, -1).T
    else:
        vectors = np.atleast_2d(directions)
    return direction_tensor(vectors)
