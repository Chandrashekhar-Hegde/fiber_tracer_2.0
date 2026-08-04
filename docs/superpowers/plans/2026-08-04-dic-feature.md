# DIC Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a real DIC (2D Digital Image Correlation) feature by extracting the DVC correlation engine into a shared, modality-neutral module and building DIC's config/pipeline/reporting/CLI/TUI surface on top of it, mirroring the already-shipped DVC feature (PR #25).

**Architecture:** `correlation/core.py` holds the generic engine (moved from `correlation/dvc.py`, `run_local_dvc` renamed `run_local_correlation`); `dvc.py` and `dic.py` are thin re-exports. `DICConfig` mirrors `DVCConfig`. `Pipeline._run_dic` mirrors `_run_dvc`. Reporting gains a `dic_windows` branch alongside the existing `dvc_windows` one.

**Tech Stack:** Python, NumPy, SciPy, `spam` (already an optional dependency, no new dependency), TypeScript/Ink for the TUI.

## Global Constraints

- Before any commit: run `black --check`, `ruff check`, `mypy`, and `pytest` — the DVC PR's CI failure was `black` formatting that ruff/mypy didn't catch; do not repeat that gap.
- `cli.py`'s `_run_pipeline` explicitly reconstructs `Config` field-by-field; the DVC feature shipped with `dvc=file_config.dvc` missing from that list (a real bug, caught late via manual CLI testing). Add `dic=file_config.dic` in the same task that adds `dic=file_config.dvc`'s sibling, not as an afterthought.
- Existing imports must keep working: `pipeline.py` and `tests/test_deformation.py` import `run_local_dvc, displacement_and_strain_per_node, estimate_noise_floor` from `fiber_tracer.correlation.dvc` — the re-export must preserve these exact names.
- `tests/test_deformation.py`'s existing 3D tests must still pass unchanged after the `core.py` extraction (it's a pure move + rename, not a behavior change).

---

### Task 1: Extract the shared correlation engine

**Files:**
- Create: `src/fiber_tracer/correlation/core.py`
- Modify: `src/fiber_tracer/correlation/dvc.py` (becomes a re-export)

- [ ] **Step 1:** Move all content of `src/fiber_tracer/correlation/dvc.py` into `src/fiber_tracer/correlation/core.py`, renaming `run_local_dvc` to `run_local_correlation` (including its one internal call site inside `estimate_noise_floor`). Update the module docstring to describe it as the shared local-correlation engine (mention it backs both DVC and DIC), keep all the existing bug-history comments (boundary bug, fork deadlock, PhiCentre convention) verbatim — they're correctness-critical for whoever touches this next.

- [ ] **Step 2:** Replace `src/fiber_tracer/correlation/dvc.py` with:

```python
"""Local Digital Volume Correlation (DVC) — thin re-export of the shared
correlation engine in `fiber_tracer.correlation.core`.

See `core.py` for the algorithm, its accuracy methodology, and the bugs
found and fixed while building it (boundary nodes, fork deadlock, PhiCentre
convention) -- all apply identically to fiber_tracer.correlation.dic.
"""

from __future__ import annotations

from fiber_tracer.correlation.core import (
    CONVERGED_STATUS,
    OUT_OF_BOUNDS_STATUS,
    displacement_and_strain_per_node,
    estimate_noise_floor,
)
from fiber_tracer.correlation.core import run_local_correlation as run_local_dvc

__all__ = [
    "CONVERGED_STATUS",
    "OUT_OF_BOUNDS_STATUS",
    "displacement_and_strain_per_node",
    "estimate_noise_floor",
    "run_local_dvc",
]
```

- [ ] **Step 3:** Run `pytest tests/test_deformation.py -v` — all existing 3D tests must pass unchanged (proves the extraction didn't change behavior).

- [ ] **Step 4:** `black --check src/fiber_tracer/correlation/ && ruff check src/fiber_tracer/correlation/ && mypy src/fiber_tracer/correlation/`, then commit:

```bash
git add src/fiber_tracer/correlation/
git commit -m "Extract shared correlation engine into correlation/core.py"
```

---

### Task 2: `correlation/dic.py`, `DICConfig`

**Files:**
- Create: `src/fiber_tracer/correlation/dic.py`
- Modify: `src/fiber_tracer/config.py`

- [ ] **Step 1:** Create `src/fiber_tracer/correlation/dic.py`:

```python
"""Local Digital Image Correlation (DIC) — thin re-export of the shared
correlation engine in `fiber_tracer.correlation.core`, called with 2D
(shape (1, H, W)) images. See `core.py` for the algorithm and accuracy
methodology; see docs/superpowers/specs/2026-08-04-dic-spike-design.md for
the spike that validated this is the same engine as DVC, unchanged.
"""

from __future__ import annotations

from fiber_tracer.correlation.core import (
    CONVERGED_STATUS,
    OUT_OF_BOUNDS_STATUS,
    displacement_and_strain_per_node,
    estimate_noise_floor,
)
from fiber_tracer.correlation.core import run_local_correlation as run_local_dic

__all__ = [
    "CONVERGED_STATUS",
    "OUT_OF_BOUNDS_STATUS",
    "displacement_and_strain_per_node",
    "estimate_noise_floor",
    "run_local_dic",
]
```

- [ ] **Step 2:** In `src/fiber_tracer/config.py`, add `DICConfig` next to `DVCConfig`:

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

Add `dic: DICConfig = field(default_factory=DICConfig)` to `Config`, immediately after the existing `dvc: DVCConfig` field.

- [ ] **Step 3:** In `Config.validate()`, add a block mirroring the existing `if self.dvc.enabled:` block exactly (existence checks on `reference_path`/`deformed_path`, positivity on the two pixel fields, range on `min_convergence_rate`), immediately after the `dvc` block.

- [ ] **Step 4:** In `Config.from_dict()`, add `if "dic" in data: data["dic"] = _dict_to_dataclass(data["dic"], DICConfig)`, immediately after the existing `dvc` branch.

- [ ] **Step 5:** In `tests/test_config.py`, add `DICConfig` to the import line, and two tests mirroring `test_dvc_config_round_trip_and_validation` / `test_dvc_enabled_requires_existing_reference_path` exactly, with `dic`/`DICConfig` substituted for `dvc`/`DVCConfig`.

- [ ] **Step 6:** `black --check && ruff check && mypy src/fiber_tracer/config.py src/fiber_tracer/correlation/` then `pytest tests/test_config.py -v`, then commit:

```bash
git add src/fiber_tracer/correlation/dic.py src/fiber_tracer/config.py tests/test_config.py
git commit -m "Add DICConfig and correlation/dic.py re-export"
```

---

### Task 3: 2D accuracy tests

**Files:**
- Modify: `tests/test_deformation.py`

- [ ] **Step 1:** Add 2D constants and a `_dense_phantom_slice()` helper near the top of the file (after the existing 3D constants), building on the same `generate_fiber_phantom` used by the spike PoC:

```python
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
```

- [ ] **Step 2:** Add the import for `run_local_dic` alongside the existing `run_local_dvc`-family import (`from fiber_tracer.correlation.dic import run_local_dic`).

- [ ] **Step 3:** Add three tests mirroring the existing 3D ones exactly, substituting the 2D helpers/constants:

```python
def test_local_dic_convergence_rate_meets_minimum():
    reference = _dense_phantom_slice()
    deformed = _deform_2d(reference)
    result = run_local_dic(reference, deformed, NODE_SPACING_2D, HALF_WINDOW_SIZE_2D)
    convergence_rate = float(np.mean(result["return_status"] == 2))
    assert convergence_rate >= MIN_CONVERGENCE_RATE, (
        f"2D convergence rate {convergence_rate:.2f} below {MIN_CONVERGENCE_RATE}"
    )


def test_local_dic_recovers_known_deformation_within_literature_bound():
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
    applied_shift = np.array([0.0, SHIFT_PIXELS[0], SHIFT_PIXELS[1]])
    applied_strain = np.zeros(3)
    applied_strain[STRAIN_AXIS_2D + 1] = STRAIN_FRACTION

    displacement_error = np.linalg.norm(displacements.mean(axis=0) - applied_shift)
    strain_error = np.max(np.abs(strains.mean(axis=0) - applied_strain))

    assert displacement_error < MAX_DISPLACEMENT_ERROR_VOXELS
    assert strain_error < MAX_STRAIN_ERROR


def test_dic_noise_floor_is_near_zero_on_self_correlation():
    reference = _dense_phantom_slice()
    noise = estimate_noise_floor(reference, NODE_SPACING_2D, HALF_WINDOW_SIZE_2D)
    assert noise["convergence_rate"] >= MIN_CONVERGENCE_RATE
    assert np.allclose(noise["displacement_std_voxels"], 0.0, atol=1e-6)
    assert np.allclose(noise["strain_std"], 0.0, atol=1e-6)
```

Note: `node_positions` for a 2D `(1, H, W)` image are still 3-component (z=0 fixed) since `spam.DIC.makeGrid` treats it as a degenerate 3D grid — this matches what the spike PoC observed (`applied_phi` used index `[1:3, ...]`, not `[0:2, ...]`). Verify this by running the test and printing `result["node_positions"]` once before finalizing the shift-vector construction above; adjust indices if the actual axis order differs from this expectation.

- [ ] **Step 4:** Run and verify (adjust tolerances/axis indices per the note above if reality differs from the draft):

```bash
source .venv/bin/activate
pytest tests/test_deformation.py -v -k dic
```

Expected: 3 passing tests, similar magnitude errors to the DVC 3D tests (sub-voxel/sub-pixel).

- [ ] **Step 5:** `black --check && ruff check && mypy` (n/a for test files, run ruff/black only) then commit:

```bash
git add tests/test_deformation.py
git commit -m "Add 2D DIC accuracy tests mirroring the 3D DVC tests"
```

---

### Task 4: Pipeline wiring

**Files:**
- Modify: `src/fiber_tracer/pipeline.py`

- [ ] **Step 1:** Add the import: `from fiber_tracer.correlation.dic import (displacement_and_strain_per_node as _dic_displacement_and_strain_per_node, estimate_noise_floor as _dic_estimate_noise_floor, run_local_dic)` — or, simpler, just reuse the DVC-imported names directly (`displacement_and_strain_per_node`/`estimate_noise_floor` are the exact same functions under both import paths, so no aliasing is needed — only `run_local_dic` needs its own import name alongside the existing `run_local_dvc`).

- [ ] **Step 2:** Add `_run_dic`, modeled directly on `_run_dvc` (read the existing method first): loads `self.config.dic.reference_path`/`deformed_path` via `load_tiff_stack`, reshapes each to `(1, H, W)` if 2D (`if arr.ndim == 2: arr = arr[np.newaxis, ...]`), calls `run_local_dic`/`displacement_and_strain_per_node`/`estimate_noise_floor` with `self.config.dic.node_spacing_pixels`/`half_window_size_pixels`, builds a `dic_summary` dict with `"regime": "dic"` plus the same keys `_run_dvc` produces (`convergence_rate`, `n_windows`, `n_converged`, `mean_displacement_voxels`, `mean_strain`, `noise_floor`, `dic_windows` (not `dvc_windows`), `config`), writes `dic_summary.json`/`dic_report.csv`/`dic_report.html`, warns below `min_convergence_rate` (same log message pattern), returns the dict.

- [ ] **Step 3:** In `run()`, add `if self.config.dic.enabled: summary["dic"] = self._run_dic(out)` immediately after the existing `if self.config.dvc.enabled:` block.

- [ ] **Step 4:** Manual smoke test (adapt the DVC manual-test pattern): build a small script or inline `python3 -c` that constructs a `Config` with `dic.enabled=True` pointing at two saved 2D TIFF slices, run `FiberAnalysisPipeline(config).run()`, confirm `dic_summary.json`/`dic_report.csv`/`dic_report.html` exist and contain sane values.

- [ ] **Step 5:** `black --check && ruff check && mypy src/fiber_tracer/pipeline.py`, then commit:

```bash
git add src/fiber_tracer/pipeline.py
git commit -m "Wire DIC into the pipeline"
```

---

### Task 5: Reporting extensions

**Files:**
- Modify: `src/fiber_tracer/reporting/csv.py`, `src/fiber_tracer/reporting/html.py`

- [ ] **Step 1:** In `csv.py`'s `_records_from_summary`, add a loop over `summary.get("dic_windows", [])` structurally identical to the existing `dvc_windows` loop, with `"regime": "dic"` instead of `"dvc"` and column names unchanged (`node_z`/`node_y`/`node_x` etc. — for a 2D node, `node_z` will just be `0`, which is fine and honest, not worth a special-cased 2D-only column schema).

- [ ] **Step 2:** In `html.py`, rename `_dvc_table` usage pattern: either generalize it to `_correlation_table(summary, key_prefix)` taking `"dvc"` or `"dic"` as a parameter (avoids duplicating the whole table-building function for a one-word difference), or add a parallel `_dic_table` if that reads cleaner given the existing function's structure — read the existing `_dvc_table` first and pick whichever is the smaller, clearer diff. Wire it into `write_html_report`'s dispatch (`elif summary.get("dic_windows"): results = ...`).

- [ ] **Step 3:** Rerun the Task 4 manual smoke test, confirm `dic_report.csv`/`dic_report.html` contain the per-node table and noise-floor comparison, matching the DVC report's shape.

- [ ] **Step 4:** `black --check && ruff check && mypy src/fiber_tracer/reporting/`, then commit:

```bash
git add src/fiber_tracer/reporting/
git commit -m "Extend CSV/HTML reports for dic_windows"
```

---

### Task 6: CLI fix (proactive, not reactive this time)

**Files:**
- Modify: `src/fiber_tracer/cli.py`

- [ ] **Step 1:** In `_run_pipeline`'s explicit `Config(...)` reconstruction, add `dic=file_config.dic,` immediately after the existing `dvc=file_config.dvc,` line.

- [ ] **Step 2:** Add a CLI-level regression test to `tests/test_pipeline.py`, mirroring `test_dvc_config_survives_cli_config_file_round_trip` exactly (`dic`/`DICConfig` substituted for `dvc`/`DVCConfig`, 2D image paths instead of 3D volume paths, asserting `(out_dir / "dic_summary.json").exists()` after `main(["--config", str(config_path)])`).

- [ ] **Step 3:** `pytest tests/test_pipeline.py -v -k dic`, then `black --check && ruff check && mypy src/fiber_tracer/cli.py`, then commit:

```bash
git add src/fiber_tracer/cli.py tests/test_pipeline.py
git commit -m "Wire dic config through cli.py; add CLI-level round-trip test"
```

---

### Task 7: TUI wiring

**Files:**
- Modify: `tui/src/types.ts`, `tui/src/bridge.ts`, `tui/src/app.tsx`, `tui/src/components/configure.tsx`, `tui/src/components/review.tsx`, `tui/src/bridge.test.ts`, `tui/src/history.test.ts`

- [ ] **Step 1:** Add `computeDic: boolean; dicReferencePath: string; dicDeformedPath: string;` to `AnalysisConfig` in `types.ts`, immediately after the existing `dvcDeformedPath` field.

- [ ] **Step 2:** In `bridge.ts`'s `buildJson`, add a `dic: { enabled: config.computeDic, reference_path: config.dicReferencePath, deformed_path: config.dicDeformedPath }` block after the existing `dvc` block.

- [ ] **Step 3:** In `app.tsx`'s `DEFAULT_CONFIG`, add `computeDic: false, dicReferencePath: "", dicDeformedPath: "",` after the existing `dvcDeformedPath` line.

- [ ] **Step 4:** In `configure.tsx`, add a `{ key: "computeDic", label: "DIC (digital image correlation)" }` entry to `TOGGLES`, and extend the conditional-rows logic (currently `config.computeDvc ? [...TOGGLES, dvc rows] : TOGGLES`) to also append `dicReferencePath`/`dicDeformedPath` rows when `config.computeDic` is true — read the existing conditional first and extend it rather than duplicating the whole rows-construction block.

- [ ] **Step 5:** In `review.tsx`, add a DIC line mirroring the existing DVC line.

- [ ] **Step 6:** Fix the two test fixture files (`bridge.test.ts`, `history.test.ts`) — add `computeDic: false, dicReferencePath: "", dicDeformedPath: "",` to the `AnalysisConfig` object literals there (same fix the DVC work needed).

- [ ] **Step 7:** `tsc --noEmit && bun test && bun run build`, then commit:

```bash
git add tui/
git commit -m "Wire DIC into the TUI"
```

---

### Task 8: Docs

**Files:**
- Modify: `docs/CAPABILITIES.md`, `ROADMAP.md`

- [ ] **Step 1:** In `CAPABILITIES.md`'s status matrix, flip `DIC (2D displacement/strain) | ❌` to `✅`, add the DIC switches to the switch-flow table (mirroring the DVC rows), and remove/update the "Research track (spike first)" DIC bullet since it's now shipped (mirror how the DVC bullet was moved to a "Shipped" section).

- [ ] **Step 2:** In `ROADMAP.md`, add a "Digital Image Correlation (epic #20)" section mirroring the existing DVC one, checking off the completed items.

- [ ] **Step 3:** Commit:

```bash
git add docs/CAPABILITIES.md ROADMAP.md
git commit -m "Update CAPABILITIES/ROADMAP for shipped DIC feature"
```

---

### Task 9: Final verification and GitHub tracking

- [ ] **Step 1:** Full sweep: `pytest tests/ -q`, `black --check .`, `ruff check src/ tests/`, `mypy src/fiber_tracer`, `cd tui && tsc --noEmit && bun test`.
- [ ] **Step 2:** Manual end-to-end run via the real `fiber-tracer run --config ...` CLI (not just the Python API), confirming `dic_summary.json`/`dic_report.csv`/`dic_report.html` are produced — this is exactly the check that caught the `dvc` config-reconstruction bug; do not skip it.
- [ ] **Step 3:** Push branch, open a PR (mirror PR #25's structure: summary, what shipped, test plan).
- [ ] **Step 4:** Confirm with the user before posting to epic #20 (checklist + closing comment), same as epic #19's convention.
- [ ] **Step 5:** Watch CI; merge (squash) once all checks are green.

## Self-Review

**Spec coverage:** module restructure → Task 1-2; config → Task 2; accuracy tests → Task 3; pipeline → Task 4; reporting → Task 5; CLI fix (proactive) → Task 6; TUI → Task 7; docs → Task 8; verification/PR/tracking → Task 9. All spec sections covered.

**Placeholder scan:** one explicitly-flagged unknown (2D `node_positions` axis indexing in Task 3) is marked for empirical verification during execution, not assumed — consistent with this project's established "verify, don't assume" pattern from the DVC work, not a placeholder in the "TBD" sense.

**Consistency:** `run_local_dic`, `DICConfig`, `dic_windows`, `dic_summary.json`/`dic_report.csv`/`dic_report.html`, `computeDic`/`dicReferencePath`/`dicDeformedPath` are used identically across all tasks that reference them.
