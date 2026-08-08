"""Digital twin fitting logic: fit synthetic-phantom parameters from a
resolved-regime pipeline summary, regenerate a statistically-matched
"twin" volume, and estimate an effective modulus via Halpin-Tsai.

Scope (see docs/superpowers/specs/2026-08-07-digital-twin-spike-design.md
and docs/superpowers/specs/2026-08-08-digital-twin-feature-design.md): a
parametric fit against the pipeline's own resolved-regime output, not
CAD/nominal-design reconciliation, DVC/DIC deformation coupling, or FE mesh
export -- those are explicitly out of scope for this feature.

Known limitation: see https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/issues/28
-- extract_fiber_paths fails to produce a centerline for most fibers in a
realistic densely-packed unidirectional bundle, so _cross_sectional_diameters_um
falls back to a less-accurate volume-extent estimate for those fibers. This
was found while validating this module's round-trip fit and is not fixed
here.
"""

from __future__ import annotations

import numpy as np

from fiber_tracer.orientation.tensor import aggregate_direction_tensor, fractional_anisotropy
from fiber_tracer.validation.phantoms import FiberPhantom, generate_fiber_phantom

ORIENTATION_MODES = ("aligned", "in_plane", "orthogonal", "woven", "twill", "random")


def _cross_sectional_diameters_um(
    fibers: list[dict], voxel_spacing_um: tuple, fallback_length_um: float
) -> list[float]:
    """Per-fiber cross-sectional diameter, inverting the cylinder volume
    formula (V = pi * (d/2)^2 * L) using each fiber's voxel count and length.

    The pipeline's own "equivalent_diameter_um" is a *sphere*-equivalent
    volume diameter -- appropriate for a roughly isotropic blob, but wrong
    for an elongated fiber: for a cylinder of true diameter 4um and length
    ~64um, the sphere-equivalent diameter comes out to ~11.3um (verified by
    hand), a ~2.8x systematic overestimate.

    `length_um` (from analysis.compute_tracking's centerline extraction) is
    used when available, but was found to be missing for most fibers in a
    realistic densely-packed unidirectional bundle -- extract_fiber_paths
    does not reliably produce a path for every fiber when fibers run close
    and parallel (issue #28, a pre-existing limitation of that module, not
    fixed here). `fallback_length_um` (an estimate from the volume's extent
    along the fitted principal axis, not per-fiber) is used instead when a
    fiber's own length is unavailable -- less accurate than a true per-fiber
    length, but far closer than the sphere-equivalent value.
    """
    voxel_volume_um3 = float(np.prod(voxel_spacing_um))
    diameters = []
    for f in fibers:
        length_um = f.get("length_um") or fallback_length_um
        volume_um3 = f["n_voxels"] * voxel_volume_um3
        diameters.append(2.0 * float(np.sqrt(volume_um3 / (np.pi * length_um))))
    return diameters


def fit_twin_parameters(summary: dict, volume_shape: tuple[int, int, int] | None = None) -> dict:
    """Fit phantom-generator parameters from a resolved-regime summary.

    `volume_shape` (voxels) is optional but recommended -- it grounds the
    fallback-length estimate used for fibers whose centerline tracking
    failed (see _cross_sectional_diameters_um); without it, the fallback is
    the diagonal of a 1-voxel cube, which is not a meaningful estimate.

    Returns a dict with keys: n_fibers, fiber_diameter_um, orientation_mode,
    fractional_anisotropy (diagnostic, not a generate_fiber_phantom input).
    """
    fibers = summary["fibers"]
    voxel_spacing_um = summary["voxel_spacing_um"]

    orientations = np.array([f["orientation"] for f in fibers if "orientation" in f])
    if len(orientations) > 0:
        tensor = aggregate_direction_tensor(orientations)
        fa = fractional_anisotropy(tensor)
        principal_axis = int(np.argmax(np.linalg.eigvalsh(tensor)))
    else:
        fa = 0.0
        principal_axis = 2

    if volume_shape is not None:
        fallback_length_um = volume_shape[principal_axis] * voxel_spacing_um[principal_axis]
    else:
        fallback_length_um = float(np.prod(voxel_spacing_um)) ** (1.0 / 3.0)

    diameters = _cross_sectional_diameters_um(fibers, voxel_spacing_um, fallback_length_um)
    mean_diameter_um = float(np.mean(diameters)) if diameters else 0.0

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
