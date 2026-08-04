"""Local Digital Volume Correlation (DVC) via the optional `spam` backend.

Accuracy note: DVC measurement error is scan/window/speckle-density dependent,
not a fixed number (RESEARCH_FOUNDATION.md ref 60, Croom et al.). Every real
run should be accompanied by `estimate_noise_floor` on the same volume/window
parameters, and non-converged nodes (`converged=False`) excluded from any
aggregate displacement/strain statistic.
"""

from __future__ import annotations

import numpy as np

from fiber_tracer.exceptions import BackendNotAvailableError

# spam.DIC.register centers Phi on the image centre; per-node results below
# must instead be decomposed about each node's own position, or strain leaks
# into apparent displacement (verified empirically: a zero-strain
# self-correlation gives exactly 0.0 displacement/strain only with this
# centering).
CONVERGED_STATUS = 2

# getImagettes' search margin (voxels beyond the half-window on each side,
# needed so register() has room to find a shifted match). Nodes whose window
# (half_window_size + this margin) extends past the volume boundary get a
# degenerate, near-constant imagette from edge padding -- verified this makes
# spam.DIC.register falsely report CONVERGED_STATUS with garbage displacement
# (observed: -78 voxels reported as "converged" for a true -1 voxel shift, on
# a node whose window ran past the volume edge). These nodes are excluded
# before correlation, not just filtered after, so they can never masquerade
# as converged.
_SEARCH_MARGIN = 3
OUT_OF_BOUNDS_STATUS = -8

# spam.DIC.ldic()'s multiprocessing.Pool is not safe to call from here: spam
# unconditionally forces multiprocessing.set_start_method("fork") on import
# (across a dozen of its own modules), and forking while spam's own
# rich.progress.Progress live-refresh thread is active deadlocked reproducibly
# under pytest (observed hanging indefinitely). Forcing "spawn" instead is not
# a fix either: ldic()'s per-node worker is a dynamically-defined closure
# (`global _multiprocessingCorrelateOneNode`) that only exists via fork's
# copy-on-write semantics -- spawned workers re-import the module fresh and
# crash with AttributeError. So this module bypasses ldic() entirely and
# replicates its per-node algorithm sequentially (extract imagette pair via
# getImagettes, then register()), which is exactly what each of ldic()'s
# worker processes does -- just without the unsafe concurrency wrapper.
# ponytail: sequential, not parallel; revisit if per-run correlation time
# becomes a bottleneck (would need a process-pool strategy that survives
# spam's forced fork context, e.g. a subprocess-per-call design).


def _import_spam():
    try:
        import spam.deformation
        import spam.DIC
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Install the dvc extra (`pip install fiber-tracer[dvc]`); on macOS "
            "spam's compiled extension also needs `brew install gmp`."
        ) from exc
    return spam.DIC, spam.deformation


def _correlate_one_node(dic, reference, deformed, node_position, hws) -> tuple:
    """Sequential equivalent of spam.DIC.ldic's internal per-node worker."""
    phi_init = np.eye(4)
    imagette_returns = dic.getImagettes(
        reference,
        node_position,
        hws,
        phi_init.copy(),
        deformed,
        [-3, 3, -3, 3, -3, 3],
        applyF="no",
    )
    if imagette_returns["returnStatus"] != 1:
        bad_phi = np.eye(4)
        bad_phi[0:3, 3] = np.nan
        return bad_phi, imagette_returns["returnStatus"], np.inf, 0

    initial_displacement = np.round(phi_init[0:3, 3]).astype(int)
    phi_init[0:3, -1] -= initial_displacement
    register_returns = dic.register(
        imagette_returns["imagette1"],
        imagette_returns["imagette2"],
        im1mask=imagette_returns["imagette1mask"],
        PhiInit=phi_init,
        margin=1,
        interpolationOrder=1,
        verbose=False,
    )
    good_phi = register_returns["Phi"]
    good_phi[0:3, -1] += initial_displacement
    return (
        good_phi,
        register_returns["returnStatus"],
        register_returns["error"],
        register_returns["iterations"],
    )


def _fits_in_bounds(node_position: np.ndarray, half_window_size: int, shape: tuple) -> bool:
    clearance = half_window_size + _SEARCH_MARGIN
    shape_arr = np.array(shape)
    return bool(
        np.all(node_position - clearance >= 0) and np.all(node_position + clearance < shape_arr)
    )


def run_local_dvc(
    reference: np.ndarray,
    deformed: np.ndarray,
    node_spacing: int,
    half_window_size: int,
) -> dict:
    """Run grid-based local DVC between *reference* and *deformed* volumes.

    Nodes whose correlation window would extend past the volume boundary are
    excluded from correlation (see OUT_OF_BOUNDS_STATUS) rather than passed to
    spam, which was observed to falsely report them as converged.

    Returns a dict with keys: node_positions (N,3), phi_field (N,4,4),
    return_status (N,) float array (2 == converged).
    """
    dic, _ = _import_spam()
    node_positions, _nodes_dim = dic.makeGrid(reference.shape, nodeSpacing=node_spacing)
    hws = np.array([half_window_size, half_window_size, half_window_size])

    n_nodes = len(node_positions)
    phi_field = np.zeros((n_nodes, 4, 4))
    return_status = np.zeros(n_nodes)
    error = np.zeros(n_nodes)
    iterations = np.zeros(n_nodes)
    for i in range(n_nodes):
        if not _fits_in_bounds(node_positions[i], half_window_size, reference.shape):
            bad_phi = np.eye(4)
            bad_phi[0:3, 3] = np.nan
            phi_field[i] = bad_phi
            return_status[i] = OUT_OF_BOUNDS_STATUS
            error[i] = np.inf
            iterations[i] = 0
            continue
        phi, status, err, its = _correlate_one_node(
            dic, reference, deformed, node_positions[i], hws
        )
        phi_field[i] = phi
        return_status[i] = status
        error[i] = err
        iterations[i] = its

    return {
        "node_positions": node_positions,
        "phi_field": phi_field,
        "return_status": return_status,
        "error": error,
        "iterations": iterations,
    }


def displacement_and_strain_per_node(
    phi_field: np.ndarray, node_positions: np.ndarray, return_status: np.ndarray
) -> list[dict]:
    """Decompose each node's Phi into displacement (voxels) and strain (zoom - 1).

    `converged` is True only for return_status == CONVERGED_STATUS; non-converged
    nodes still get a best-effort decomposition (for inspection) but callers must
    exclude them from aggregate statistics.
    """
    _, spam_defo = _import_spam()
    results = []
    for i in range(len(node_positions)):
        decomposed = spam_defo.decomposePhi(phi_field[i], PhiCentre=node_positions[i])
        results.append(
            {
                "node_position": node_positions[i].tolist(),
                "displacement_voxels": np.asarray(decomposed["t"]).tolist(),
                "strain": (np.asarray(decomposed["z"]) - 1.0).tolist(),
                "return_status": float(return_status[i]),
                "converged": bool(return_status[i] == CONVERGED_STATUS),
            }
        )
    return results


def estimate_noise_floor(
    volume: np.ndarray, node_spacing: int, half_window_size: int
) -> dict:
    """Correlate *volume* against a copy of itself to measure the DVC
    measurement-noise floor for this volume/window configuration (Croom et al.
    practice, RESEARCH_FOUNDATION.md ref 60): any nonzero displacement/strain
    here is noise, not signal, since no deformation was applied.
    """
    result = run_local_dvc(volume, volume.copy(), node_spacing, half_window_size)
    nodes = displacement_and_strain_per_node(
        result["phi_field"], result["node_positions"], result["return_status"]
    )
    converged = [n for n in nodes if n["converged"]]
    convergence_rate = len(converged) / len(nodes) if nodes else 0.0
    if converged:
        displacements = np.array([n["displacement_voxels"] for n in converged])
        strains = np.array([n["strain"] for n in converged])
        displacement_std = displacements.std(axis=0).tolist()
        strain_std = strains.std(axis=0).tolist()
    else:
        displacement_std = [float("nan")] * 3
        strain_std = [float("nan")] * 3
    return {
        "convergence_rate": convergence_rate,
        "displacement_std_voxels": displacement_std,
        "strain_std": strain_std,
        "n_nodes": len(nodes),
        "n_converged": len(converged),
    }
