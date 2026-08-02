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
