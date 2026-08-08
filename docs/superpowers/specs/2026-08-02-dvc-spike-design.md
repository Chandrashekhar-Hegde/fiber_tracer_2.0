# Digital Volume Correlation (DVC) Spike — Design Spec

**Goal:** Validate feasibility of DVC for fiber-composite CT with a small proof-of-concept, before committing to any pipeline/CLI/config integration (per `docs/CAPABILITIES.md`'s "Research track (spike first)" item and `docs/RESEARCH_FOUNDATION.md` §8).

**Non-goal:** Production DVC feature. No pipeline, config, or CLI wiring; no strain visualization; no CI gate.

## Background

The research survey (`docs/RESEARCH_FOUNDATION.md` §8) surveys DVC methods and open tooling. `spam` (ref 57) is the most relevant open-source library: local/global/discrete correlation, actively used for granular media and composites, Python-native. This spike uses `spam` rather than hand-rolling correlation, to spend the spike's effort on "does DVC work on our synthetic fiber volumes" rather than on reimplementing a correlation algorithm.

## Approach

1. **Reference volume:** generate a phantom via the existing `fiber_tracer.validation.phantoms.generate_fiber_phantom` — the fiber architecture itself serves as the correlation speckle pattern, matching how DVC is applied to real composites (ref 61, Mehdikhani et al.).
2. **Known deformation:** apply a single combined field — a rigid shift (e.g. 2.5 voxels along one axis) composed with a uniform uniaxial strain (e.g. 2% stretch along a different axis) — via `scipy.ndimage.affine_transform`, producing a "deformed" volume with an exactly known ground-truth displacement field.
3. **Correlation:** run `spam`'s local (subset-based) DVC between reference and deformed volumes to recover a displacement field.
4. **Comparison:** compute mean/max error between recovered and known-applied displacement (in voxels) and strain (in %).

## Components

- **`pyproject.toml`:** add optional extra `dvc = ["spam>=0.9"]`, following the existing `structure`/`skeleton`/`ml` extras pattern. Not added to the `all` extra or to CI — `spam`'s wheel availability across macOS/Linux/Windows is unverified (confirmed to install on macOS arm64 during this spike; not tested on Linux/Windows).
- **`scripts/dvc_poc.py`:** standalone script, not part of the installed package (`src/fiber_tracer/`) or the pipeline. Structure:
  - `build_reference_and_deformed(...)` — phantom generation + known-deformation application.
  - `run_dvc(reference, deformed) -> displacement_field` — thin wrapper around `spam`'s correlation call.
  - `compare_to_ground_truth(recovered, applied) -> dict` — error metrics.
  - `__main__` — runs the above, prints results, and asserts the recovered displacement error is within a generous bound (e.g. <0.5 voxel) as a smoke-test / self-check.

## Success criteria

This is exploratory, so "success" is a documented finding, not a fixed pass/fail bar:
- **Feasible:** DVC recovers the known displacement within a few tenths of a voxel and the known strain within ~0.5% absolute — proceed to scope a real design (config/pipeline integration, phantom-based validation suite, real-scan testing).
- **Not (yet) feasible:** large recovery error, or `spam` fails to install/run on this stack — document why (e.g. speckle contrast too low, subset size mismatch, dependency friction) and report back before further DVC work is scheduled.

## Testing

The script's own `__main__` assertion (error within bound) is the runnable check, per repo convention (no pytest suite for a spike script). Findings — pass/fail, actual error numbers, and any blockers — get reported back in the conversation, not written into a separate report doc.
