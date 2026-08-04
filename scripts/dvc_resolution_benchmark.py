"""Benchmark: how does DVC accuracy change with phantom voxelization fidelity?

Note on terminology: `supersample` (fiber_tracer.validation.phantoms) does not
increase the voxel grid's resolution/voxel count -- it anti-aliases each fixed
voxel by averaging supersample**3 sub-voxel samples, reproducing the
partial-volume effect of a real detector (supersample=1 gives hard/aliased
fiber edges; higher values give smoother, more realistic boundaries). This
benchmark sweeps that fidelity parameter as a proxy for how partial-volume
blur affects DVC accuracy, since a real scan's effective resolution similarly
trades off against edge sharpness.

Levels are capped at [1, 4] (4x satisfies the "minimum 4x" requirement).
supersample=8 was measured to take >13 minutes at this phantom size (killed as
impractical); shrinking the volume/fiber-count to compensate was tried and
rejected -- the phantom generator's default fiber length spans the volume
diagonal, so a smaller volume also means shorter fibers and starves
correlation windows of texture (measured convergence dropped to ~20%,
independent of the resolution question this benchmark is meant to answer).
This uses the same shape/node/window parameters validated in
tests/test_deformation.py (>=90% convergence at supersample=1).

See docs/superpowers/specs/2026-08-02-dvc-spike-design.md and
docs/DVC_BENCHMARK.md for the accompanying cross-tool comparison.

# ponytail: standalone script, not part of the installed package. Requires the
# dvc extra (spam + libgmp on macOS).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from fiber_tracer.correlation.dvc import displacement_and_strain_per_node, run_local_dvc
from fiber_tracer.validation.phantoms import generate_fiber_phantom

SHAPE = (80, 80, 80)
N_FIBERS = 200
NODE_SPACING = 20
HALF_WINDOW_SIZE = 10
SUPERSAMPLE_LEVELS = [1, 4]
SHIFT_VOXELS = np.array([2.5, 0.0, 0.0])
STRAIN_AXIS = 2
STRAIN_FRACTION = 0.02


def _build_pair(supersample: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    phantom = generate_fiber_phantom(
        shape=SHAPE,
        n_fibers=N_FIBERS,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=seed,
        supersample=supersample,
    )
    reference = phantom.volume.astype(np.float32)
    zoom = np.ones(3)
    zoom[STRAIN_AXIS] = 1.0 + STRAIN_FRACTION
    matrix = np.diag(1.0 / zoom)
    offset = -SHIFT_VOXELS / zoom
    deformed = affine_transform(reference, matrix, offset=offset, order=1, mode="nearest").astype(
        np.float32
    )
    return reference, deformed


def run_one_level(supersample: int) -> dict:
    reference, deformed = _build_pair(supersample)
    result = run_local_dvc(reference, deformed, NODE_SPACING, HALF_WINDOW_SIZE)
    windows = displacement_and_strain_per_node(
        result["phi_field"], result["node_positions"], result["return_status"]
    )
    converged = [w for w in windows if w["converged"]]
    convergence_rate = len(converged) / len(windows) if windows else 0.0

    applied_strain = np.zeros(3)
    applied_strain[STRAIN_AXIS] = STRAIN_FRACTION

    if converged:
        displacements = np.array([w["displacement_voxels"] for w in converged])
        strains = np.array([w["strain"] for w in converged])
        displacement_error = float(np.linalg.norm(displacements.mean(axis=0) - SHIFT_VOXELS))
        strain_error = float(np.max(np.abs(strains.mean(axis=0) - applied_strain)))
    else:
        displacement_error = float("nan")
        strain_error = float("nan")

    return {
        "supersample": supersample,
        "convergence_rate": convergence_rate,
        "n_windows": len(windows),
        "n_converged": len(converged),
        "displacement_error_voxels": displacement_error,
        "strain_error": strain_error,
    }


def main() -> None:
    results = [run_one_level(s) for s in SUPERSAMPLE_LEVELS]

    print(f"{'supersample':>11} {'convergence':>12} {'disp. error (vox)':>18} {'strain error':>13}")
    for r in results:
        print(
            f"{r['supersample']:>11} {r['convergence_rate'] * 100:>11.1f}% "
            f"{r['displacement_error_voxels']:>18.4f} {r['strain_error']:>13.4f}"
        )

    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "dvc_resolution_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    md_lines = [
        "| Supersample | Convergence rate | Displacement error (voxels) | Strain error |",
        "|---|---|---|---|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['supersample']} | {r['convergence_rate'] * 100:.1f}% "
            f"| {r['displacement_error_voxels']:.4f} | {r['strain_error']:.4f} |"
        )
    with open(out_dir / "dvc_resolution_benchmark.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nWrote {out_dir / 'dvc_resolution_benchmark.json'} and .md")


if __name__ == "__main__":
    main()
