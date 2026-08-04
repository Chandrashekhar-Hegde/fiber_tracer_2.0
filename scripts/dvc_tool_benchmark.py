"""Cross-tool DVC comparison: this repo's `spam`-based correlation vs.
literature-reported figures for other tools, on a shared known-deformation
phantom.

Tool selection (see docs/DVC_BENCHMARK.md for the full writeup):
- spam: run live here (this repo's fiber_tracer.correlation.dvc).
- TomoWarp2 (open-source): evaluated, NOT integrated. Its core correlation
  worker (DIC_worker.py) hard-depends on a SWIG-wrapped C extension, not a
  pure-Python fallback; that extension uses distutils.core (removed entirely
  in Python 3.12) and requires the `swig` binary (not installed here) on top
  of ~26 files of Python-2-only syntax. Porting it is a multi-day undertaking,
  not a benchmark-script task -- reported here as evaluated-not-integrated.
- Avizo (commercial): no license available in this environment. Its DVC
  methodology (two-step local-then-FE-global correlation) is described from
  vendor/literature sources, but no specific accuracy number is reported here
  because a fetchable, independently-verifiable source could not be confirmed
  in this session -- see RESEARCH_FOUNDATION.md ref 60 (Croom et al.) for the
  applicable literature accuracy context instead, which is equipment-agnostic.
- DICe and FIDVC were excluded from consideration entirely: DICe is a
  stereo-camera surface-DIC tool (2D + triangulation), not volumetric CT-DVC;
  FIDVC (Bar-Kochba et al., ref 55) is MATLAB-only, no Python.

# ponytail: standalone script, not part of the installed package.
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
SHIFT_VOXELS = np.array([2.5, 0.0, 0.0])
STRAIN_AXIS = 2
STRAIN_FRACTION = 0.02


def _build_pair() -> tuple[np.ndarray, np.ndarray]:
    phantom = generate_fiber_phantom(
        shape=SHAPE,
        n_fibers=N_FIBERS,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        orientation_mode="random",
        seed=1,
    )
    reference = phantom.volume.astype(np.float32)
    zoom = np.ones(3)
    zoom[STRAIN_AXIS] = 1.0 + STRAIN_FRACTION
    matrix = np.diag(1.0 / zoom)
    offset = -SHIFT_VOXELS / zoom
    deformed = affine_transform(
        reference, matrix, offset=offset, order=1, mode="nearest"
    ).astype(np.float32)
    return reference, deformed


def run_spam() -> dict:
    import time

    reference, deformed = _build_pair()
    t0 = time.time()
    result = run_local_dvc(reference, deformed, NODE_SPACING, HALF_WINDOW_SIZE)
    elapsed = time.time() - t0
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
        "tool": "spam (this repo)",
        "status": "measured",
        "convergence_rate": convergence_rate,
        "displacement_error_voxels": displacement_error,
        "strain_error": strain_error,
        "runtime_seconds": elapsed,
    }


def tomowarp2_row() -> dict:
    return {
        "tool": "TomoWarp2",
        "status": "evaluated, not integrated",
        "convergence_rate": None,
        "displacement_error_voxels": None,
        "strain_error": None,
        "runtime_seconds": None,
        "note": (
            "Core correlation worker requires a SWIG-wrapped C extension built "
            "via distutils.core (removed in Python 3.12); requires the `swig` "
            "binary (not installed); ~26 files of Python-2-only syntax. No "
            "pure-Python correlation path exists to fall back to."
        ),
    }


def avizo_row() -> dict:
    return {
        "tool": "Avizo (Thermo Fisher)",
        "status": "literature description, no live run",
        "convergence_rate": None,
        "displacement_error_voxels": None,
        "strain_error": None,
        "runtime_seconds": None,
        "note": (
            "Commercial, no license available in this environment. Methodology "
            "(corroborated across vendor and independent sources): two-step "
            "local-then-FE-global correlation. No tool-specific accuracy number "
            "reported -- a fetchable, independently-verifiable source could not "
            "be confirmed; see RESEARCH_FOUNDATION.md ref 60 (Croom et al.) for "
            "the applicable, verified literature accuracy context instead."
        ),
    }


def main() -> None:
    rows = [run_spam(), tomowarp2_row(), avizo_row()]

    print(f"{'tool':<24} {'status':<28} {'convergence':>12} {'disp. err (vox)':>16}")
    for r in rows:
        rate = r["convergence_rate"]
        disp_err = r["displacement_error_voxels"]
        conv = f"{rate * 100:.1f}%" if rate is not None else "n/a"
        disp = f"{disp_err:.4f}" if disp_err is not None else "n/a"
        print(f"{r['tool']:<24} {r['status']:<28} {conv:>12} {disp:>16}")

    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "dvc_tool_benchmark.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_dir / 'dvc_tool_benchmark.json'}")


if __name__ == "__main__":
    main()
