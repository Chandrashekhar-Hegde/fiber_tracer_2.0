"""Ordered per-fiber centerline extraction from a labeled skeleton.

Each fiber's skeleton voxels are connected with 26-neighbourhood adjacency and
ordered into a single end-to-end path. The path is the longest geodesic through
the skeleton (its two most distant endpoints), found with a double breadth-first
search. This keeps the implementation dependency-free; optional ``skan`` graph
metrics can be layered on top for branched skeletons.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# 26-neighbourhood offsets (all non-zero combinations of -1/0/1).
_NEIGHBOUR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def _farthest_node(
    start: tuple[int, int, int],
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> tuple[tuple[int, int, int], dict[tuple[int, int, int], tuple[int, int, int] | None]]:
    """BFS from *start*; return the farthest node and the parent map for backtracking."""
    parents: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    queue = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        farthest = node  # last node popped at the deepest BFS level
        for neighbour in adjacency[node]:
            if neighbour not in parents:
                parents[neighbour] = node
                queue.append(neighbour)
    return farthest, parents


def _order_component(coords: np.ndarray) -> np.ndarray:
    """Order a connected set of skeleton voxels into an end-to-end path."""
    if len(coords) <= 1:
        return coords

    points: list[tuple[int, int, int]] = [(int(c[0]), int(c[1]), int(c[2])) for c in coords]
    point_set = set(points)
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {p: [] for p in points}
    for p in points:
        pz, py, px = p
        for dz, dy, dx in _NEIGHBOUR_OFFSETS:
            neighbour = (pz + dz, py + dy, px + dx)
            if neighbour in point_set:
                adjacency[p].append(neighbour)

    # Double BFS: farthest node from an arbitrary start, then farthest from there.
    source, _ = _farthest_node(points[0], adjacency)
    target, parents = _farthest_node(source, adjacency)

    path: list[tuple[int, int, int]] = []
    node: tuple[int, int, int] | None = target
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return np.array(path, dtype=np.int64)


def extract_fiber_paths(labels: np.ndarray, skeleton: np.ndarray) -> dict[int, np.ndarray]:
    """Return an ordered centerline path per fiber label.

    Parameters
    ----------
    labels:
        Integer label volume (0 = background).
    skeleton:
        Boolean skeleton volume aligned with *labels*.

    Returns
    -------
    paths:
        Mapping ``label_id -> (N, 3)`` ordered voxel coordinates (z, y, x).
        Labels whose skeleton is empty are omitted.
    """
    paths: dict[int, np.ndarray] = {}
    skeleton_bool = np.asarray(skeleton, dtype=bool)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = skeleton_bool & (labels == label)
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        paths[int(label)] = _order_component(coords)
    return paths
