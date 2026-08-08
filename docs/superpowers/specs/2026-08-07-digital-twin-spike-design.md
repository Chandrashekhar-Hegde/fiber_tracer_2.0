# Digital Twin Spike — Scope Definition & PoC Design

## Context

Epic #21 states the term "digital twin" is currently undefined for this project and explicitly requires a written scope definition plus a minimal PoC proposal, signed off before any real feature work — a different acceptance bar than the DVC/DIC spikes (which targeted a working prototype against known ground truth). This doc is that scope definition.

Of the three candidate scopes the epic lists, the chosen direction is **a parametric synthetic-microstructure model fitted to a real scan, with property export** — reusing the existing phantom generator (`validation/phantoms.py`), orientation-tensor code (`orientation/tensor.py`), and the resolved-regime pipeline's per-fiber morphometry output, rather than building new measurement infrastructure.

## Scope definition

**What "digital twin" means for this tool (this spike's proposal):** a parametric description of a specimen's fiber microstructure — mean fiber diameter, fiber count/volume fraction, and an aggregate orientation tensor — fitted from a real scan's already-computed resolved-regime output, capable of (a) regenerating a statistically-matched synthetic volume via `generate_fiber_phantom`, and (b) producing an effective-stiffness estimate via Halpin-Tsai. This is deliberately narrower than "as-manufactured-to-as-designed reconciliation" (`RESEARCH_FOUNDATION.md` §10, Hearley et al. ref 80) — that requires per-fiber spatial registration against a CAD/nominal model, which this repo has no nominal-model input for and is out of scope here.

**What it explicitly does not mean, for this spike:**
- No CAD/nominal-design comparison (no "as-designed" input exists in this tool).
- No DVC/DIC deformation coupling (a stated candidate scope, deferred — the epic's own third option; revisit once the twin's static microstructure representation is validated).
- No FE mesh export (property export here is a single scalar effective modulus via a closed-form equation, not a meshed geometry for external FE solvers).
- No continuous per-fiber orientation-tensor fitting into the phantom generator's categorical `orientation_mode` — the fit maps a measured aggregate tensor to the closest existing mode (`random`/`aligned`/`in_plane`/`orthogonal`/`woven`/`twill`) via fractional anisotropy and principal axis, not a new continuous sampling model. Extending the generator to accept an arbitrary orientation distribution directly is real future work, not this spike's.

## Reusable primitives (confirmed present, not proposed)

- `fiber_tracer.validation.phantoms.generate_fiber_phantom` — regenerates a synthetic volume from `n_fibers`, `fiber_diameter_um`, `orientation_mode`, `voxel_spacing_um`.
- `fiber_tracer.orientation.tensor.aggregate_direction_tensor` / `fractional_anisotropy` — turns a resolved-regime run's per-fiber orientation vectors into an Advani-Tucker second-order tensor and a scalar anisotropy measure.
- `fiber_tracer.io.estimate_volume_fraction` — foreground fraction from a binarized volume, usable as a volume-fraction fit target.
- The resolved-regime pipeline (`Pipeline._run_resolved`) already computes everything needed as the "real scan" fit target: per-fiber `equivalent_diameter_um`, `orientation`, and `n_labels` in its `summary.json`/`summary["fibers"]`.

## Minimal PoC

**Deliverable:** `scripts/digital_twin_poc.py`, standalone, no pipeline/config/CLI/TUI wiring (matching the DVC/DIC PoC precedent — spike first, feature later if signed off).

1. `fit_twin_parameters(summary: dict) -> dict` — takes a resolved-regime `summary.json`-shaped dict (or the equivalent in-memory summary), computes: mean `fiber_diameter_um` from `summary["fibers"]`; `n_fibers` and volume-fraction proxy from `summary["n_labels"]` and voxel counts; aggregate orientation tensor + fractional anisotropy from per-fiber `orientation` vectors (via `aggregate_direction_tensor`/`fractional_anisotropy`); maps the tensor's anisotropy/principal axis to the closest `generate_fiber_phantom` `orientation_mode` string.
2. `regenerate_twin(fitted_params: dict, shape, voxel_spacing_um) -> FiberPhantom` — calls `generate_fiber_phantom` with the fitted parameters.
3. `effective_modulus_halpin_tsai(volume_fraction, fiber_modulus_gpa, matrix_modulus_gpa, aspect_ratio) -> float` — the Halpin-Tsai closed-form equation (`RESEARCH_FOUNDATION.md` ref 67), a few lines, no new dependency.
4. `main()` — runs the full loop against a **synthetic phantom standing in for "the real scan"** (since there's no real experimental scan in this repo to fit against, matching how the DVC/DIC PoCs first validated against synthetic ground truth before any real-data question was raised): generate a phantom with known parameters, run it through the existing resolved-regime pipeline to get a `summary`, fit twin parameters from that summary, regenerate a twin volume, and report the fitted parameters + effective modulus alongside the original known parameters for comparison.

## Validation (and its explicit limitation)

Unlike DVC/DIC, there is no independent ground truth to check the *twin* against — the twin's whole premise is "reconstruct the generating parameters from measured output," so the natural check is a round-trip: does `fit_twin_parameters` recover parameters close to the ones used to generate the original phantom (known, since step 4 above starts from a known-parameter phantom)? This validates the *fitting* logic, not the *twin concept* (whether a statistically-matched phantom is a meaningful proxy for a real specimen's mechanical behavior) — that broader question is out of scope for a spike and is flagged as future validation work requiring real paired scan/mechanical-test data.

## Recommendation

Proceed with the PoC as scoped above. If the round-trip fit is reasonably accurate (diameter/volume-fraction within the phantom generator's own discretization noise, orientation mode correctly recovered), recommend a follow-on feature scoped to: config/CLI wiring for `twin.enabled` + `twin.source_summary_path`, writing a `twin_summary.json` report (mirroring the DVC/DIC report convention), deferring FE export, DVC/DIC coupling, and CAD reconciliation to later epics as explicitly out of scope above.
