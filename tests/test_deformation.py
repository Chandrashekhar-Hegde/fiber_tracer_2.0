"""Accuracy tests for local DVC (fiber_tracer.correlation.dvc).

Requires the optional `spam` backend (dvc extra) and, on macOS, a system
libgmp (`brew install gmp`). See docs/superpowers/specs/2026-08-02-dvc-spike-design.md
and RESEARCH_FOUNDATION.md ref 60 for why these thresholds are literature-derived,
not arbitrary.
"""

import numpy as np
import pytest
from scipy.ndimage import affine_transform

pytest.importorskip("spam")

from fiber_tracer.correlation.dic import run_local_dic  # noqa: E402
from fiber_tracer.correlation.dvc import (  # noqa: E402
    displacement_and_strain_per_node,
    estimate_noise_floor,
    run_local_dvc,
)
from fiber_tracer.validation.phantoms import generate_fiber_phantom  # noqa: E402

SHAPE = (80, 80, 80)
NODE_SPACING = 20
HALF_WINDOW_SIZE = 10
# 200 fibers in an 80^3 volume gives ~93% convergence at the default window
# settings with comfortable margin above MIN_CONVERGENCE_RATE (150 fibers
# measures ~89%, too close to the 90% floor to be a stable regression test);
# the sparse 40-fiber phantom used by the earlier spike converges only ~41%
# of nodes and is not representative of a usable configuration.
N_FIBERS = 200
SHIFT_VOXELS = np.array([2.5, 0.0, 0.0])
STRAIN_AXIS = 2
STRAIN_FRACTION = 0.02
MIN_CONVERGENCE_RATE = 0.9
# Croom et al. (RESEARCH_FOUNDATION.md ref 60): 0.012-0.043 voxel displacement
# std dev achievable under good conditions; allow up to the documented
# worst-case machine-specific distortion (0.5 voxel) as the upper bound for a
# synthetic, noise-free phantom (real data will sit closer to the noise floor).
MAX_DISPLACEMENT_ERROR_VOXELS = 0.5
MAX_STRAIN_ERROR = 0.02


def _dense_phantom(seed: int = 1) -> np.ndarray:
    phantom = generate_fiber_phantom(
        shape=SHAPE,
        n_fibers=N_FIBERS,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=seed,
    )
    return phantom.volume.astype(np.float32)


def _deform(reference: np.ndarray) -> np.ndarray:
    zoom = np.ones(3)
    zoom[STRAIN_AXIS] = 1.0 + STRAIN_FRACTION
    matrix = np.diag(1.0 / zoom)
    offset = -SHIFT_VOXELS / zoom
    return affine_transform(reference, matrix, offset=offset, order=1, mode="nearest").astype(
        np.float32
    )


def test_local_dvc_convergence_rate_meets_minimum():
    """A dense phantom with adequate speckle content converges >=90% of nodes."""
    reference = _dense_phantom()
    deformed = _deform(reference)
    result = run_local_dvc(reference, deformed, NODE_SPACING, HALF_WINDOW_SIZE)
    convergence_rate = float(np.mean(result["return_status"] == 2))
    assert convergence_rate >= MIN_CONVERGENCE_RATE, (
        f"convergence rate {convergence_rate:.2f} below {MIN_CONVERGENCE_RATE} -- "
        "default node_spacing/half_window_size no longer match the validated "
        "fiber density; see RESEARCH_FOUNDATION.md ref 54 (Buljac et al.) on "
        "subset-size/speckle-density dependence."
    )


def test_local_dvc_recovers_known_deformation_within_literature_bound():
    """Recovered displacement/strain must fall within the Croom et al. range."""
    reference = _dense_phantom()
    deformed = _deform(reference)
    result = run_local_dvc(reference, deformed, NODE_SPACING, HALF_WINDOW_SIZE)
    nodes = displacement_and_strain_per_node(
        result["phi_field"], result["node_positions"], result["return_status"]
    )
    converged = [n for n in nodes if n["converged"]]
    assert converged, "no nodes converged; cannot assess accuracy"

    displacements = np.array([n["displacement_voxels"] for n in converged])
    strains = np.array([n["strain"] for n in converged])

    applied_strain = np.zeros(3)
    applied_strain[STRAIN_AXIS] = STRAIN_FRACTION

    displacement_error = np.linalg.norm(displacements.mean(axis=0) - SHIFT_VOXELS)
    strain_error = np.max(np.abs(strains.mean(axis=0) - applied_strain))

    assert displacement_error < MAX_DISPLACEMENT_ERROR_VOXELS, (
        f"mean displacement error {displacement_error:.4f} voxels exceeds "
        f"{MAX_DISPLACEMENT_ERROR_VOXELS} (Croom et al. worst-case bound)"
    )
    assert (
        strain_error < MAX_STRAIN_ERROR
    ), f"mean strain error {strain_error:.4f} exceeds {MAX_STRAIN_ERROR}"


def test_noise_floor_is_near_zero_on_self_correlation():
    """estimate_noise_floor on an undeformed self-correlation must report ~0."""
    reference = _dense_phantom()
    noise = estimate_noise_floor(reference, NODE_SPACING, HALF_WINDOW_SIZE)

    assert noise["convergence_rate"] >= MIN_CONVERGENCE_RATE
    assert np.allclose(noise["displacement_std_voxels"], 0.0, atol=1e-6)
    assert np.allclose(noise["strain_std"], 0.0, atol=1e-6)


def test_boundary_nodes_are_excluded_not_falsely_converged():
    """Nodes whose window extends past the volume edge must never report
    converged=True with a bogus displacement.

    Regression test: a 64^3 volume with node_spacing=20/half_window_size=10
    has grid nodes at z/y/x=60, whose window+search-margin (10+3=13) reaches
    voxel 73, past the 64-voxel boundary. Before the out-of-bounds guard,
    spam.DIC.register on the resulting degenerate (edge-padded) imagette
    falsely reported CONVERGED_STATUS with garbage displacement (observed:
    -78 voxels reported as converged for a true -1 voxel shift).
    """
    phantom = generate_fiber_phantom(
        shape=(64, 64, 64),
        n_fibers=100,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=1,
    )
    reference = phantom.volume.astype(np.float32)
    deformed = affine_transform(
        reference, np.eye(3), offset=[1.0, 0.0, 0.0], order=1, mode="nearest"
    ).astype(np.float32)

    result = run_local_dvc(reference, deformed, NODE_SPACING, HALF_WINDOW_SIZE)
    windows = displacement_and_strain_per_node(
        result["phi_field"], result["node_positions"], result["return_status"]
    )

    boundary_node_z60 = [w for w in windows if w["node_position"][0] >= 60]
    assert boundary_node_z60, "expected at least one boundary node in this grid"
    for w in boundary_node_z60:
        assert not w["converged"], (
            f"boundary node {w['node_position']} falsely reported converged "
            f"with displacement {w['displacement_voxels']}"
        )

    converged = [w for w in windows if w["converged"]]
    assert converged, "no interior nodes converged"
    displacements = np.array([w["displacement_voxels"] for w in converged])
    mean_error = np.linalg.norm(displacements.mean(axis=0) - np.array([-1.0, 0.0, 0.0]))
    assert mean_error < 0.01, f"mean displacement error {mean_error:.4f} on interior nodes"


# --- 2D DIC accuracy tests -------------------------------------------------
# Same methodology as the 3D DVC tests above, on a single slice of the same
# phantom. Confirms the DIC spike's finding (docs/superpowers/specs/
# 2026-08-04-dic-spike-design.md): fiber_tracer.correlation.dic.run_local_dic
# is the exact same engine as run_local_dvc, called with a (1, H, W) image.

SHAPE_3D_FOR_SLICE = (20, 80, 80)
SHIFT_PIXELS = np.array([2.5, 0.0])
STRAIN_AXIS_2D = 1
NODE_SPACING_2D = 20
HALF_WINDOW_SIZE_2D = 10


def _dense_phantom_slice(seed: int = 1) -> np.ndarray:
    phantom = generate_fiber_phantom(
        shape=SHAPE_3D_FOR_SLICE,
        n_fibers=200,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=seed,
    )
    slice_2d = phantom.volume[SHAPE_3D_FOR_SLICE[0] // 2].astype(np.float32)
    return slice_2d[np.newaxis, ...]


def _deform_2d(reference: np.ndarray) -> np.ndarray:
    zoom = np.ones(2)
    zoom[STRAIN_AXIS_2D] = 1.0 + STRAIN_FRACTION
    matrix = np.diag(1.0 / zoom)
    offset = -SHIFT_PIXELS / zoom
    deformed_2d = affine_transform(
        reference[0], matrix, offset=offset, order=1, mode="nearest"
    ).astype(np.float32)
    return deformed_2d[np.newaxis, ...]


def test_local_dic_convergence_rate_meets_minimum():
    """A dense phantom slice with adequate speckle content converges >=90% of nodes."""
    reference = _dense_phantom_slice()
    deformed = _deform_2d(reference)
    result = run_local_dic(reference, deformed, NODE_SPACING_2D, HALF_WINDOW_SIZE_2D)
    convergence_rate = float(np.mean(result["return_status"] == 2))
    assert (
        convergence_rate >= MIN_CONVERGENCE_RATE
    ), f"2D convergence rate {convergence_rate:.2f} below {MIN_CONVERGENCE_RATE}"


def test_local_dic_recovers_known_deformation_within_literature_bound():
    """Recovered 2D displacement/strain must fall within the Croom et al. range."""
    reference = _dense_phantom_slice()
    deformed = _deform_2d(reference)
    result = run_local_dic(reference, deformed, NODE_SPACING_2D, HALF_WINDOW_SIZE_2D)
    nodes = displacement_and_strain_per_node(
        result["phi_field"], result["node_positions"], result["return_status"]
    )
    converged = [n for n in nodes if n["converged"]]
    assert converged, "no nodes converged; cannot assess accuracy"

    displacements = np.array([n["displacement_voxels"] for n in converged])
    strains = np.array([n["strain"] for n in converged])

    # node_positions/Phi remain 3-component for a (1, H, W) image (spam treats
    # it as a degenerate 3D grid with a fixed z=0 axis) -- index 0 is the
    # singleton z axis, indices 1-2 are the real (y, x) image axes.
    applied_shift = np.array([0.0, SHIFT_PIXELS[0], SHIFT_PIXELS[1]])
    applied_strain = np.zeros(3)
    applied_strain[STRAIN_AXIS_2D + 1] = STRAIN_FRACTION

    displacement_error = np.linalg.norm(displacements.mean(axis=0) - applied_shift)
    strain_error = np.max(np.abs(strains.mean(axis=0) - applied_strain))

    assert displacement_error < MAX_DISPLACEMENT_ERROR_VOXELS, (
        f"mean 2D displacement error {displacement_error:.4f} px exceeds "
        f"{MAX_DISPLACEMENT_ERROR_VOXELS}"
    )
    assert (
        strain_error < MAX_STRAIN_ERROR
    ), f"mean 2D strain error {strain_error:.4f} exceeds {MAX_STRAIN_ERROR}"


def test_dic_noise_floor_is_near_zero_on_self_correlation():
    """estimate_noise_floor on a 2D undeformed self-correlation must report ~0."""
    reference = _dense_phantom_slice()
    noise = estimate_noise_floor(reference, NODE_SPACING_2D, HALF_WINDOW_SIZE_2D)

    assert noise["convergence_rate"] >= MIN_CONVERGENCE_RATE
    assert np.allclose(noise["displacement_std_voxels"], 0.0, atol=1e-6)
    assert np.allclose(noise["strain_std"], 0.0, atol=1e-6)
