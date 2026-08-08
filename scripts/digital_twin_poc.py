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
    and parallel (a pre-existing limitation of that module, not something
    this spike fixes). `fallback_length_um` (an estimate from the volume's
    extent along the fitted principal axis, not per-fiber) is used instead
    when a fiber's own length is unavailable -- less accurate than a true
    per-fiber length, but far closer than the sphere-equivalent value.
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


def main() -> None:
    import tempfile
    from pathlib import Path

    from fiber_tracer.config import Config, VoxelSpacing
    from fiber_tracer.io import save_tiff_stack
    from fiber_tracer.pipeline import FiberAnalysisPipeline

    shape = (64, 64, 64)
    voxel_spacing_um = (1.0, 1.0, 1.0)
    # 30 fibers at this diameter/volume touch and merge under segmentation
    # (verified by hand: n_labels came out to 26, not 30, with a badly
    # inflated fitted diameter from the merged blobs) -- 15 is sparse enough
    # to avoid that confound so this test validates the fitting logic, not
    # an unrelated segmentation-density limitation.
    known_n_fibers = 15
    known_diameter_um = 4.0
    known_mode = "aligned"

    known_phantom = generate_fiber_phantom(
        shape=shape,
        n_fibers=known_n_fibers,
        fiber_diameter_um=known_diameter_um,
        voxel_spacing_um=voxel_spacing_um,
        orientation_mode=known_mode,
        seed=1,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "scan.tif"
        save_tiff_stack(data_path, known_phantom.volume)

        config = Config(
            data_path=str(data_path),
            output_dir=str(tmp_path / "out"),
            voxel_spacing_um=VoxelSpacing(*voxel_spacing_um),
            fiber_diameter_um=known_diameter_um,
            regime="resolved",
        )
        summary = FiberAnalysisPipeline(config).run()

    fitted = fit_twin_parameters(summary, volume_shape=shape)
    print(f"known:  n_fibers={known_n_fibers} diameter_um={known_diameter_um} mode={known_mode}")
    print(
        f"fitted: n_fibers={fitted['n_fibers']} diameter_um={fitted['fiber_diameter_um']:.2f} "
        f"mode={fitted['orientation_mode']} (FA={fitted['fractional_anisotropy']:.2f})"
    )

    twin = regenerate_twin(fitted, shape, voxel_spacing_um)
    print(f"twin volume shape: {twin.volume.shape}, twin fiber count: {len(twin.orientations)}")

    volume_fraction = fitted["n_fibers"] * np.pi * (fitted["fiber_diameter_um"] / 2) ** 2 * shape[0]
    volume_fraction /= np.prod(shape) * np.prod(voxel_spacing_um)
    volume_fraction = min(volume_fraction, 0.65)  # physically-plausible packing cap

    modulus = effective_modulus_halpin_tsai(
        volume_fraction, fiber_modulus_gpa=72.0, matrix_modulus_gpa=3.0, aspect_ratio=20.0
    )
    print(f"estimated volume fraction: {volume_fraction:.3f}")
    print(f"Halpin-Tsai effective modulus: {modulus:.2f} GPa")

    # ponytail: generous PoC tolerances -- validates the fitting logic
    # round-trips, not that the twin is mechanically representative of a
    # real specimen (no ground truth for that exists in this repo; see the
    # spec's Validation section).
    diameter_error = abs(fitted["fiber_diameter_um"] - known_diameter_um)
    assert diameter_error < 0.5, f"fitted diameter off by {diameter_error:.2f} um"
    # Not an exact match: generate_fiber_phantom places fibers at random
    # without collision avoidance, so a small fraction touch and merge into
    # a single connected-component label under segmentation (verified by
    # hand: 15 known fibers -> 13 labels here). This is a property of the
    # phantom generator/segmentation, not the fitting logic under test.
    n_fibers_error = abs(fitted["n_fibers"] - known_n_fibers) / known_n_fibers
    assert n_fibers_error < 0.25, (
        f"fitted fiber count {fitted['n_fibers']} vs known {known_n_fibers} "
        f"(error {n_fibers_error:.0%}) exceeds the expected merge-rate tolerance"
    )
    assert (
        fitted["orientation_mode"] == known_mode
    ), f"fitted mode {fitted['orientation_mode']!r} != known mode {known_mode!r}"
    print("PoC self-check passed: fitted parameters round-trip within tolerance.")


if __name__ == "__main__":
    main()
