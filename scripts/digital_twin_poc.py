"""Proof-of-concept: fit synthetic-phantom parameters from a resolved-regime
pipeline summary (standing in for "a real scan" -- this repo has no real
experimental scan to fit against, so the round-trip below validates the
fitting logic against a known-parameter phantom, not the twin concept
itself against real mechanical behavior).

See docs/superpowers/specs/2026-08-07-digital-twin-spike-design.md.

# ponytail: spike script, not part of the installed package; no
# pipeline/config/CLI/TUI wiring.
"""

from __future__ import annotations

import numpy as np

from fiber_tracer.orientation.tensor import aggregate_direction_tensor, fractional_anisotropy
from fiber_tracer.validation.phantoms import FiberPhantom, generate_fiber_phantom

ORIENTATION_MODES = ("aligned", "in_plane", "orthogonal", "woven", "twill", "random")


def fit_twin_parameters(summary: dict) -> dict:
    """Fit phantom-generator parameters from a resolved-regime summary.

    Returns a dict with keys: n_fibers, fiber_diameter_um, orientation_mode,
    fractional_anisotropy (diagnostic, not a generate_fiber_phantom input).
    """
    fibers = summary["fibers"]
    diameters = [f["equivalent_diameter_um"] for f in fibers if "equivalent_diameter_um" in f]
    mean_diameter_um = float(np.mean(diameters)) if diameters else 0.0

    orientations = np.array([f["orientation"] for f in fibers if "orientation" in f])
    if len(orientations) > 0:
        tensor = aggregate_direction_tensor(orientations)
        fa = fractional_anisotropy(tensor)
        principal_axis = int(np.argmax(np.linalg.eigvalsh(tensor)))
    else:
        fa = 0.0
        principal_axis = 2

    # Coarse heuristic mapping fractional anisotropy -> the phantom
    # generator's categorical orientation_mode: high FA means one dominant
    # axis (aligned); low FA means near-isotropic (random). This is exactly
    # the "closest existing mode" mapping the spec flags as a limitation,
    # not a continuous-distribution fit.
    if fa > 0.6:
        orientation_mode = "aligned"
    elif fa < 0.2:
        orientation_mode = "random"
    else:
        orientation_mode = "in_plane"

    return {
        "n_fibers": summary["n_labels"],
        "fiber_diameter_um": mean_diameter_um,
        "orientation_mode": orientation_mode,
        "fractional_anisotropy": float(fa),
        "principal_axis": principal_axis,
    }


def regenerate_twin(
    fitted_params: dict, shape: tuple[int, int, int], voxel_spacing_um: tuple[float, float, float]
) -> FiberPhantom:
    return generate_fiber_phantom(
        shape=shape,
        n_fibers=fitted_params["n_fibers"],
        fiber_diameter_um=fitted_params["fiber_diameter_um"],
        voxel_spacing_um=voxel_spacing_um,
        orientation_mode=fitted_params["orientation_mode"],
        seed=0,
    )


def effective_modulus_halpin_tsai(
    volume_fraction: float,
    fiber_modulus_gpa: float,
    matrix_modulus_gpa: float,
    aspect_ratio: float,
) -> float:
    """Halpin-Tsai longitudinal effective modulus (RESEARCH_FOUNDATION.md ref 67)."""
    xi = 2.0 * aspect_ratio
    ratio = fiber_modulus_gpa / matrix_modulus_gpa
    eta = (ratio - 1.0) / (ratio + xi)
    return matrix_modulus_gpa * (1.0 + xi * eta * volume_fraction) / (1.0 - eta * volume_fraction)
