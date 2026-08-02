"""Proof-of-concept: does DVC (via `spam`) recover a known applied deformation
on a synthetic fiber phantom?

See docs/superpowers/specs/2026-08-02-dvc-spike-design.md for the design.

# ponytail: spike script, not part of the installed package; no pipeline/CLI
# wiring. Requires the `dvc` extra (`pip install -e .[dvc]`) AND a system
# libgmp (`brew install gmp` on macOS) for spam's compiled extension to import.
"""

from __future__ import annotations

import numpy as np
import spam.DIC as dic
import spam.deformation as spam_defo
from scipy.ndimage import affine_transform

from fiber_tracer.validation.phantoms import generate_fiber_phantom

SHAPE = (80, 80, 80)
SHIFT_VOXELS = np.array([2.5, 0.0, 0.0])  # applied rigid shift (z, y, x)
STRAIN_AXIS = 2  # x axis
STRAIN_FRACTION = 0.02  # 2% uniaxial stretch along STRAIN_AXIS


def build_reference_and_deformed(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reference phantom and a known-deformed copy.

    Returns (reference_volume, deformed_volume, applied_phi). applied_phi maps
    reference coordinates to deformed coordinates: x_deformed = applied_phi[:3,:3] @ x_ref + applied_phi[:3,3].
    """
    phantom = generate_fiber_phantom(
        shape=SHAPE,
        n_fibers=40,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=seed,
    )
    reference = phantom.volume.astype(np.float32)

    zoom = np.ones(3)
    zoom[STRAIN_AXIS] = 1.0 + STRAIN_FRACTION

    applied_phi = np.eye(4)
    applied_phi[:3, :3] = np.diag(zoom)
    applied_phi[:3, 3] = SHIFT_VOXELS

    # affine_transform maps output coords -> input coords, i.e. the inverse of
    # applied_phi: x_ref = (x_deformed - shift) / zoom.
    matrix = np.diag(1.0 / zoom)
    offset = -SHIFT_VOXELS / zoom
    deformed = affine_transform(
        reference, matrix, offset=offset, order=1, mode="nearest"
    ).astype(np.float32)

    return reference, deformed, applied_phi


def run_dvc(reference: np.ndarray, deformed: np.ndarray) -> dict:
    """Whole-volume affine registration: recovers a single Phi mapping
    reference -> deformed. Margin=10 gives room for the 2.5-voxel shift."""
    return dic.register(
        reference,
        deformed,
        margin=10,
        maxIterations=50,
        interpolationOrder=1,
        verbose=False,
    )


def compare_to_ground_truth(
    recovered_phi: np.ndarray, applied_phi: np.ndarray, volume_shape: tuple[int, int, int]
) -> dict:
    # spam.DIC.register's Phi is defined about the image centre (per its
    # docstring), so it must be decomposed about that same centre -- decomposing
    # about the origin (decomposePhi's default) leaks strain into an apparent
    # translation of (zoom - 1) * centre on the strain axis.
    centre = (np.asarray(volume_shape) - 1) / 2.0
    recovered = spam_defo.decomposePhi(recovered_phi, PhiCentre=centre)
    applied = spam_defo.decomposePhi(applied_phi, PhiCentre=[0.0, 0.0, 0.0])
    recovered_t = np.asarray(recovered["t"])
    applied_t = np.asarray(applied["t"])
    recovered_z = np.asarray(recovered["z"])
    applied_z = np.asarray(applied["z"])
    return {
        "displacement_error_voxels": float(np.linalg.norm(recovered_t - applied_t)),
        "strain_error_fraction": float(np.max(np.abs(recovered_z - applied_z))),
        "recovered_displacement": recovered_t,
        "applied_displacement": applied_t,
        "recovered_zoom": recovered_z,
        "applied_zoom": applied_z,
    }


def main() -> None:
    reference, deformed, applied_phi = build_reference_and_deformed()
    result = run_dvc(reference, deformed)
    print(f"spam returnStatus={result['returnStatus']} error={result['error']:.6f}")

    comparison = compare_to_ground_truth(result["Phi"], applied_phi, reference.shape)
    print(f"applied displacement (voxels):   {comparison['applied_displacement']}")
    print(f"recovered displacement (voxels): {comparison['recovered_displacement']}")
    print(f"displacement error (voxels):     {comparison['displacement_error_voxels']:.4f}")
    print(f"applied zoom:                    {comparison['applied_zoom']}")
    print(f"recovered zoom:                  {comparison['recovered_zoom']}")
    print(f"strain error (fraction):         {comparison['strain_error_fraction']:.4f}")

    # ponytail: generous PoC tolerance, not a production accuracy bar -- the
    # question here is "does it work at all," not "how accurate is it."
    assert comparison["displacement_error_voxels"] < 0.5, (
        "DVC displacement recovery outside PoC tolerance (>0.5 voxel) -- "
        "see docs/superpowers/specs/2026-08-02-dvc-spike-design.md 'Not (yet) feasible'"
    )
    assert comparison["strain_error_fraction"] < 0.02, (
        "DVC strain recovery outside PoC tolerance (>2% absolute) -- "
        "see docs/superpowers/specs/2026-08-02-dvc-spike-design.md 'Not (yet) feasible'"
    )
    print("PoC self-check passed: DVC recovered the known deformation within tolerance.")


if __name__ == "__main__":
    main()
