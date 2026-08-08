# DIC (Digital Image Correlation) Spike — Design Spec

**Goal:** Validate feasibility of 2D DIC with a small proof-of-concept, and empirically answer epic #20's "how much core is shared with DVC" question, before committing to any pipeline/CLI/config integration (per `docs/CAPABILITIES.md`'s DIC research-track entry).

**Non-goal:** Production DIC feature, grid-based local correlation, pipeline/config/CLI wiring, CI gate.

## Background

Epic #20 (DIC) and epic #19 (DVC, already shipped as `fiber_tracer.correlation.dvc`) both name shared-core as an open decision. Inspecting `spam.DIC.register`'s source (`.venv/lib/python3.12/site-packages/spam/DIC/registration.py:221-222`) shows it auto-detects a 2D image via `im1.shape[0] == 1` and takes a dedicated 2D code path — the same function already used by `fiber_tracer.correlation.dvc` for 3D volumes. This spike tests whether that same backend, called with a `(1, H, W)` image pair instead of a full 3D volume, recovers a known 2D deformation.

## Approach

1. **Test image:** generate a phantom via the existing `fiber_tracer.validation.phantoms.generate_fiber_phantom` (same parameters as the DVC accuracy tests), take a single z-slice as the 2D reference image, reshape to `(1, H, W)` for `spam`.
2. **Known deformation:** apply a 2D rigid shift (e.g. 2.5 pixels) composed with a uniform uniaxial strain (e.g. 2%) via `scipy.ndimage.affine_transform`, producing a "deformed" 2D image with an exactly known ground-truth field.
3. **Correlation:** call `spam.DIC.register()` directly on the reshaped reference/deformed pair (whole-image, not a grid — matching the DVC spike's first validation step).
4. **Comparison:** decompose the recovered `Phi` with `spam.deformation.decomposePhi(Phi, PhiCentre=...)`, using the image-center convention already verified for `register()` in the DVC work, and compare against the known applied displacement/strain.

## Components

- **`scripts/dic_poc.py`:** standalone script, not part of the installed package or the pipeline. Structure mirrors the original DVC spike script:
  - `build_reference_and_deformed(...)` — phantom slice extraction + known-deformation application.
  - `run_dic(reference, deformed) -> dict` — thin wrapper around `spam.DIC.register`.
  - `compare_to_ground_truth(recovered_phi, applied_phi, image_shape) -> dict` — error metrics, reusing the center-correction convention from `fiber_tracer.correlation.dvc`.
  - `__main__` — runs the above, prints results, and asserts the recovered error is within a generous bound as a smoke-test/self-check.

## Success criteria

Exploratory, per the DVC spike's precedent:
- **Feasible + shared-core confirmed:** DIC recovers the known displacement/strain within a similar sub-pixel range to the DVC spike's results, using the same `spam.DIC.register` call as `fiber_tracer.correlation.dvc` — meaning a future real DIC feature can share `correlation/dvc.py`'s helper functions almost unchanged (2D input, not a separate algorithm).
- **Not (yet) feasible:** large recovery error, or the 2D path behaves meaningfully differently from the already-validated 3D path — document why before further DIC work is scheduled.

## Testing

The script's own `__main__` assertion is the runnable check, per repo convention for a spike script (no pytest suite). Findings get reported back in the conversation.
