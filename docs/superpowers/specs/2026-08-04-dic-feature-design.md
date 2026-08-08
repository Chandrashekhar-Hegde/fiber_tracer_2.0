# DIC (Digital Image Correlation) Feature — Design Spec

## Context

The DIC spike (`research/dic-spike` branch, `scripts/dic_poc.py`; this validation now lives permanently as the DIC cases in `tests/test_deformation.py`) validated that `spam.DIC.register` — the exact same function `fiber_tracer.correlation.dvc` already uses for 3D volumes — recovers a known 2D deformation on a slice of the fiber phantom (0.04px displacement error, 0.06% strain error), given a `(1, H, W)`-shaped image. This directly answers epic #20's "how much core is shared with DVC" question: the correlation engine needs zero new algorithm code for 2D. This spec turns that into a real feature, mirroring how the DVC spike became `fiber_tracer.correlation.dvc` (PR #25).

## Design

**Module restructure.** `correlation/dvc.py`'s generic engine (`run_local_dvc`, `displacement_and_strain_per_node`, `estimate_noise_floor`, plus the private helpers `_correlate_one_node`, `_fits_in_bounds`, `_import_spam`) is shape-agnostic already — none of it assumes 3D beyond passing `reference.shape` through to `spam.DIC.makeGrid`, which itself handles 2D via the `(1, H, W)` convention. Move it to `correlation/core.py`, renaming `run_local_dvc` to `run_local_correlation` (the other two names are already modality-neutral). `correlation/dvc.py` and the new `correlation/dic.py` become thin re-exports:

```python
# correlation/dvc.py
from fiber_tracer.correlation.core import (
    run_local_correlation as run_local_dvc,
    displacement_and_strain_per_node,
    estimate_noise_floor,
)
```

Existing imports in `pipeline.py` and `tests/test_deformation.py` (`from fiber_tracer.correlation.dvc import run_local_dvc, ...`) keep working unchanged.

**Config.** `DICConfig`, mirroring `DVCConfig` field-for-field but with `_pixels` naming (a real distinction: DIC operates on 2D images/pixels, DVC on 3D volumes/voxels):

```python
@dataclass
class DICConfig:
    enabled: bool = False
    reference_path: str = ""
    deformed_path: str = ""
    node_spacing_pixels: int = 20
    half_window_size_pixels: int = 10
    min_convergence_rate: float = 0.9
```

Added as `Config.dic`, with the same `from_dict`/`validate` treatment `DVCConfig` already has (existence checks on the two paths when enabled, range checks on the numeric fields).

**Pipeline.** New `Pipeline._run_dic(out) -> dict`, gated by `self.config.dic.enabled`, called from `run()` right after `_run_dvc` (same pattern: independent of regime, own report files). Loads the two 2D images via the existing `load_tiff_stack` (which returns a plain `(H, W)` array for a 2D TIFF — confirmed by reading `io.py`), reshapes to `(1, H, W)`, and reuses the exact same noise-floor-and-convergence-gating methodology already built for DVC (§ per `RESEARCH_FOUNDATION.md` ref 60 — the same equipment/scan-dependent-accuracy argument applies to DIC, per Holmes et al., ref 63).

**Reporting.** `reporting/csv.py` and `reporting/html.py` gain a `dic_windows` branch, structurally identical to the existing `dvc_windows` one (2D node positions `(y, x)` instead of 3D `(z, y, x)` — the row shape is a strict subset, not a different shape, so the extension is mechanical).

**CLI/TUI.** No new CLI flag (matches the DVC precedent — config-file-only). TUI: `computeDic`, `dicReferencePath`, `dicDeformedPath` added to `AnalysisConfig`/`buildJson`, and a `computeDic` toggle + two file-picker rows in `Configure`, mirroring the DVC rows already there.

**Critical files:**
- `src/fiber_tracer/correlation/core.py` (new) — moved engine.
- `src/fiber_tracer/correlation/dvc.py` — becomes a 3-line re-export (was the full engine).
- `src/fiber_tracer/correlation/dic.py` (new) — re-export, DIC-specific docstring.
- `src/fiber_tracer/config.py` — `DICConfig`, `Config.dic`, `from_dict`/`validate` wiring (pattern already established for `DVCConfig`).
- `src/fiber_tracer/pipeline.py` — `_run_dic`, call site in `run()`.
- `src/fiber_tracer/reporting/csv.py`, `reporting/html.py` — `dic_windows` branch.
- `src/fiber_tracer/cli.py` — add `dic=file_config.dic` to the `Config(...)` reconstruction in `_run_pipeline` (this exact omission was a real bug caught late in the DVC work — get it right this time from the start).
- `tui/src/types.ts`, `tui/src/bridge.ts`, `tui/src/components/configure.tsx`, `tui/src/components/review.tsx`, `tui/src/app.tsx` — DIC fields/toggle/picker, mirroring the existing DVC rows.
- `tests/test_deformation.py` — extend with 2D accuracy tests (known-deformation recovery + noise-floor self-consistency + convergence-rate gate on a 2D phantom slice), mirroring the existing 3D tests; also a `tests/test_pipeline.py` toggle test and a CLI-level round-trip test (the same class of bug the `dvc` field reconstruction had).
- `docs/CAPABILITIES.md`, `ROADMAP.md` — flip DIC status, mirroring the DVC updates.

## Not doing

- No shared UI component abstraction between the DVC/DIC toggle rows in the TUI — two similar-but-separate row blocks is fine at this scale; abstracting now would be premature.
- No dedicated DIC accuracy/tool-comparison benchmark script — the PoC already validated the shared engine, and `DVC_BENCHMARK.md`'s accuracy story (noise floor, convergence gating, Croom et al. context) applies identically since it's the same code path. A `tests/test_deformation.py` 2D accuracy test is the right level of validation here, not a new benchmark doc.

## Testing

Pytest, following the exact patterns already established for DVC: `tests/test_config.py` (round-trip + validation), `tests/test_deformation.py` (2D accuracy: known-deformation recovery, noise-floor self-consistency, convergence-rate gate — all `pytest.importorskip("spam")`), `tests/test_pipeline.py` (toggle on/off, and the CLI-level `main()` round-trip that catches config-reconstruction omissions).
