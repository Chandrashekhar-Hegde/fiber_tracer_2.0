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

# spam.DIC.register centers Phi on the image centre; per-node ldic() results
# must instead be decomposed about each node's own position, or strain leaks
# into apparent displacement (verified empirically: a zero-strain
# self-correlation gives exactly 0.0 displacement/strain only with this
# centering).
CONVERGED_STATUS = 2


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


def run_local_dvc(
    reference: np.ndarray,
    deformed: np.ndarray,
    node_spacing: int,
    half_window_size: int,
) -> dict:
    """Run grid-based local DVC between *reference* and *deformed* volumes.

    Returns a dict with keys: node_positions (N,3), phi_field (N,4,4),
    return_status (N,) float array (2 == converged).
    """
    dic, _ = _import_spam()
    node_positions, _nodes_dim = dic.makeGrid(reference.shape, nodeSpacing=node_spacing)
    hws = np.array([half_window_size, half_window_size, half_window_size])
    phi_field, return_status, error, iterations, delta_phi_norm = dic.ldic(
        reference, deformed, node_positions, hws
    )
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
