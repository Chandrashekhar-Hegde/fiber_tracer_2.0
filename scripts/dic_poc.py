"""Proof-of-concept: does `spam.DIC.register` recover a known applied 2D
deformation, using the same call already validated for 3D DVC
(fiber_tracer.correlation.dvc)? Answers epic #20's shared-core question.

See docs/superpowers/specs/2026-08-04-dic-spike-design.md for the design.

# ponytail: spike script, not part of the installed package; no pipeline/CLI
# wiring. Requires the `dvc` extra (`pip install -e .[dvc]`) AND a system
# libgmp (`brew install gmp` on macOS).
"""

from __future__ import annotations

import numpy as np
import spam.deformation
import spam.DIC
from scipy.ndimage import affine_transform

from fiber_tracer.validation.phantoms import generate_fiber_phantom

SHAPE_3D = (20, 80, 80)
SHIFT_PIXELS = np.array([2.5, 0.0])  # (y, x)
STRAIN_AXIS = 1  # x axis
STRAIN_FRACTION = 0.02


def build_reference_and_deformed(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 2D reference image (one slice of a fiber phantom) and a
    known-deformed copy.

    Returns (reference_2d, deformed_2d, applied_phi) where both images are
    shaped (1, H, W) (spam's 2D convention) and applied_phi is a (4, 4) array
    mapping reference coordinates to deformed coordinates.
    """
    phantom = generate_fiber_phantom(
        shape=SHAPE_3D,
        n_fibers=200,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=seed,
    )
    reference_2d = phantom.volume[SHAPE_3D[0] // 2].astype(np.float32)

    zoom = np.ones(2)
    zoom[STRAIN_AXIS] = 1.0 + STRAIN_FRACTION

    applied_phi = np.eye(4)
    applied_phi[1:3, 1:3] = np.diag(zoom)
    applied_phi[1:3, 3] = SHIFT_PIXELS

    matrix = np.diag(1.0 / zoom)
    offset = -SHIFT_PIXELS / zoom
    deformed_2d = affine_transform(
        reference_2d, matrix, offset=offset, order=1, mode="nearest"
    ).astype(np.float32)

    return reference_2d[np.newaxis, ...], deformed_2d[np.newaxis, ...], applied_phi


def run_dic(reference: np.ndarray, deformed: np.ndarray) -> dict:
    result: dict = spam.DIC.register(
        reference,
        deformed,
        margin=10,
        maxIterations=50,
        interpolationOrder=1,
        verbose=False,
    )
    return result


def compare_to_ground_truth(
    recovered_phi: np.ndarray, applied_phi: np.ndarray, image_shape: tuple[int, int]
) -> dict:
    # Mirrors fiber_tracer.correlation.dvc: register()'s Phi is centered on the
    # image center, not the origin -- decomposing about the wrong point leaks
    # strain into apparent displacement (this exact bug, and its fix, is
    # documented in src/fiber_tracer/correlation/dvc.py's compare logic).
    centre = np.zeros(3)
    centre[1:3] = (np.asarray(image_shape) - 1) / 2.0
    recovered = spam.deformation.decomposePhi(recovered_phi, PhiCentre=centre)
    applied = spam.deformation.decomposePhi(applied_phi, PhiCentre=[0.0, 0.0, 0.0])
    recovered_t = np.asarray(recovered["t"])
    applied_t = np.asarray(applied["t"])
    recovered_z = np.asarray(recovered["z"])
    applied_z = np.asarray(applied["z"])
    return {
        "displacement_error_pixels": float(np.linalg.norm(recovered_t - applied_t)),
        "strain_error_fraction": float(np.max(np.abs(recovered_z - applied_z))),
        "recovered_displacement": recovered_t,
        "applied_displacement": applied_t,
        "recovered_zoom": recovered_z,
        "applied_zoom": applied_z,
    }


def main() -> None:
    reference, deformed, applied_phi = build_reference_and_deformed()
    result = run_dic(reference, deformed)
    print(f"spam returnStatus={result['returnStatus']} error={result['error']:.6f}")

    comparison = compare_to_ground_truth(result["Phi"], applied_phi, reference.shape[1:])
    print(f"applied displacement (px):   {comparison['applied_displacement']}")
    print(f"recovered displacement (px): {comparison['recovered_displacement']}")
    print(f"displacement error (px):     {comparison['displacement_error_pixels']:.4f}")
    print(f"applied zoom:                {comparison['applied_zoom']}")
    print(f"recovered zoom:              {comparison['recovered_zoom']}")
    print(f"strain error (fraction):     {comparison['strain_error_fraction']:.4f}")

    # ponytail: generous PoC tolerance, matching the DVC spike's precedent --
    # the question is "does the same backend work for 2D," not accuracy.
    assert comparison["displacement_error_pixels"] < 0.5, (
        "DIC displacement recovery outside PoC tolerance -- "
        "see docs/superpowers/specs/2026-08-04-dic-spike-design.md 'Not (yet) feasible'"
    )
    assert comparison["strain_error_fraction"] < 0.02, (
        "DIC strain recovery outside PoC tolerance -- "
        "see docs/superpowers/specs/2026-08-04-dic-spike-design.md 'Not (yet) feasible'"
    )
    print("PoC self-check passed: DIC recovered the known 2D deformation within tolerance.")


if __name__ == "__main__":
    main()
